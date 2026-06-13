"""
End-to-end pipeline orchestrator for the Neonatal Sepsis project.
Runs: preprocess → create splits → local train → split clients → FL simulation → evaluate → plot.

Each step is skipped automatically when its output already exists.
Use --force_* flags to re-run specific steps.

Usage:
  python scripts/run_pipeline.py                          # full run with defaults
  python scripts/run_pipeline.py --force_eval             # re-run evaluation only
  python scripts/run_pipeline.py --skip_local_train       # FL only (no local baseline)
  python scripts/run_pipeline.py --skip_fl                # local training only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _run(cmd, desc, env):
    print(f"\n{'=' * 60}")
    print(f"[PIPELINE] {desc}")
    print(f"[CMD] {' '.join(str(c) for c in cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n[PIPELINE] ERROR: '{desc}' failed (rc={result.returncode}). Stopping.")
        sys.exit(result.returncode)


def _find_latest_local_ckpt(runs_dir: Path):
    candidates = sorted(
        runs_dir.glob("*/checkpoints/model_best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser(description="End-to-end pipeline orchestrator")
    ap.add_argument("--raw_folder", default=str(PROJECT_ROOT / "data" / "raw"))
    ap.add_argument("--model", choices=["transformer", "grud"], default="transformer")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--fl_rounds", type=int, default=5)
    ap.add_argument("--fl_local_epochs", type=int, default=1)
    ap.add_argument("--n_clients", type=int, default=3)
    ap.add_argument("--n_features", type=int, default=40)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--nprocs", type=int, default=4)
    ap.add_argument("--run_name", default="pipeline_local")
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    # Skip flags
    ap.add_argument("--skip_local_train", action="store_true")
    ap.add_argument("--skip_fl", action="store_true")
    # Force flags
    ap.add_argument("--force_preprocess", action="store_true")
    ap.add_argument("--force_splits", action="store_true")
    ap.add_argument("--force_train", action="store_true")
    ap.add_argument("--force_split", action="store_true")
    ap.add_argument("--force_fl", action="store_true")
    ap.add_argument("--force_eval", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    processed_dir = PROJECT_ROOT / "data" / "processed" / "patients"
    index_path = processed_dir / "index_with_labels.pt"
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_index = splits_dir / "train_index.pt"
    val_index = splits_dir / "val_index.pt"
    test_index = splits_dir / "test_index.pt"
    clients_dir = PROJECT_ROOT / "data" / "processed" / "clients"
    test_client_index = clients_dir / f"client{args.n_clients}" / "index.pt"
    runs_dir = PROJECT_ROOT / "runs"
    global_best = PROJECT_ROOT / "server_out" / "global_best.pt"
    eval_fed = PROJECT_ROOT / "eval_results_federated.json"
    eval_local = PROJECT_ROOT / "eval_results_local.json"
    plot_out = PROJECT_ROOT / "model_comparison_plot.png"

    # ── Step 1: Preprocessing ──────────────────────────────────────────────────
    if not index_path.exists() or args.force_preprocess:
        _run(
            [
                py, str(SRC_DIR / "parallel_preprocess.py"),
                "--raw_folder", args.raw_folder,
                "--out_folder", str(processed_dir),
                "--seq_len", str(args.seq_len),
                "--nprocs", str(args.nprocs),
            ],
            "Step 1/7 — Preprocessing raw PSV data",
            env,
        )
    else:
        print(f"\n[PIPELINE] Step 1/7 — Skipping preprocessing (index exists: {index_path})")

    # ── Step 2: Create frozen 70/15/15 splits ─────────────────────────────────
    if not test_index.exists() or args.force_splits:
        _run(
            [
                py, str(SCRIPTS_DIR / "create_splits.py"),
                "--index", str(index_path),
                "--out_dir", str(splits_dir),
                "--train_ratio", str(args.train_ratio),
                "--val_ratio", str(args.val_ratio),
            ],
            "Step 2/7 — Creating frozen 70/15/15 patient splits",
            env,
        )
    else:
        print(f"\n[PIPELINE] Step 2/7 — Skipping splits (already frozen: {splits_dir})")

    # Determine which index to use for local training (train split if available)
    local_train_index = str(train_index) if train_index.exists() else str(index_path)

    # ── Step 3: Local Training ─────────────────────────────────────────────────
    local_ckpt = None
    if not args.skip_local_train:
        existing = _find_latest_local_ckpt(runs_dir)
        if existing and not args.force_train:
            print(f"\n[PIPELINE] Step 3/7 — Skipping local training (checkpoint: {existing})")
            local_ckpt = existing
        else:
            _run(
                [
                    py, str(SRC_DIR / "train_local.py"),
                    "--index", local_train_index,
                    "--model", args.model,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--patience", str(args.patience),
                    "--run_name", args.run_name,
                ],
                "Step 3/7 — Local training (on train split)",
                env,
            )
            local_ckpt = _find_latest_local_ckpt(runs_dir)
    else:
        print("\n[PIPELINE] Step 3/7 — Skipping local training (--skip_local_train)")

    # ── Step 4: Split Clients (excluding test patients) ────────────────────────
    if not test_client_index.exists() or args.force_split:
        _run(
            [
                py, str(SRC_DIR / "split_clients.py"),
                "--processed_folder", str(processed_dir),
                "--out_root", str(clients_dir),
                "--n_clients", str(args.n_clients),
                "--splits_dir", str(splits_dir),
            ],
            "Step 4/7 — Splitting into federated client folders (test excluded)",
            env,
        )
    else:
        print(f"\n[PIPELINE] Step 4/7 — Skipping client split (folders exist)")

    # ── Step 5: Federated Learning ─────────────────────────────────────────────
    if not args.skip_fl:
        if not global_best.exists() or args.force_fl:
            client_indexes = [
                str(clients_dir / f"client{i + 1}" / "index.pt")
                for i in range(args.n_clients - 1)
            ]
            _run(
                [
                    py, str(SCRIPTS_DIR / "run_fl_sim.py"),
                    "--client_indexes", *client_indexes,
                    "--model", args.model,
                    "--rounds", str(args.fl_rounds),
                    "--local_epochs", str(args.fl_local_epochs),
                    "--n_features", str(args.n_features),
                    "--seq_len", str(args.seq_len),
                ],
                "Step 5/7 — Federated learning simulation",
                env,
            )
        else:
            print(f"\n[PIPELINE] Step 5/7 — Skipping FL (global_best.pt exists: {global_best})")
    else:
        print("\n[PIPELINE] Step 5/7 — Skipping federated learning (--skip_fl)")

    # ── Step 6: Evaluation (on frozen test set) ────────────────────────────────
    eval_index = str(test_index) if test_index.exists() else str(test_client_index)
    evaluated = False

    if global_best.exists() and (test_index.exists() or test_client_index.exists()):
        if not eval_fed.exists() or args.force_eval:
            _run(
                [
                    py, str(SRC_DIR / "evaluate.py"),
                    "--index", eval_index,
                    "--ckpt", str(global_best),
                    "--model", args.model,
                    "--n_features", str(args.n_features),
                    "--seq_len", str(args.seq_len),
                    "--out_file", str(eval_fed),
                ],
                "Step 6a/7 — Evaluating federated model on test set",
                env,
            )
            evaluated = True
        else:
            print(f"\n[PIPELINE] Step 6a/7 — Skipping federated eval (file exists: {eval_fed})")

    if local_ckpt and (test_index.exists() or test_client_index.exists()):
        if not eval_local.exists() or args.force_eval:
            _run(
                [
                    py, str(SRC_DIR / "evaluate.py"),
                    "--index", eval_index,
                    "--ckpt", str(local_ckpt),
                    "--model", args.model,
                    "--n_features", str(args.n_features),
                    "--seq_len", str(args.seq_len),
                    "--out_file", str(eval_local),
                ],
                "Step 6b/7 — Evaluating local model on test set",
                env,
            )
            evaluated = True
        else:
            print(f"\n[PIPELINE] Step 6b/7 — Skipping local eval (file exists: {eval_local})")

    # ── Step 7: Plot ───────────────────────────────────────────────────────────
    result_files = [str(p) for p in [eval_fed, eval_local] if p.exists()]
    if result_files:
        _run(
            [
                py, str(SRC_DIR / "plot_results.py"),
                "--results", *result_files,
                "--out_file", str(plot_out),
            ],
            "Step 7/7 — Generating ROC/PRC comparison plots",
            env,
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[PIPELINE] All steps complete!")
    print(f"  Frozen splits      : {splits_dir}")
    print(f"  Global best model  : {global_best}")
    print(f"  Local checkpoint   : {local_ckpt or 'N/A'}")
    print(f"  Eval (federated)   : {eval_fed}")
    print(f"  Eval (local)       : {eval_local}")
    print(f"  Plots              : {plot_out}")
    print("\n  Launch dashboard:")
    print("    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
