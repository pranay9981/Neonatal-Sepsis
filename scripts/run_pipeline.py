"""
End-to-end pipeline orchestrator for the Neonatal Sepsis project.
Runs: preprocess → local train → split clients → FL simulation → evaluate → plot.

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
    print(f"\n{'='*60}")
    print(f"[PIPELINE] {desc}")
    print(f"[CMD] {' '.join(str(c) for c in cmd)}")
    print("="*60)
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n[PIPELINE] ERROR: '{desc}' failed (rc={result.returncode}). Stopping.")
        sys.exit(result.returncode)


def _find_latest_local_ckpt(runs_dir: Path):
    """Return the most recently modified model_best.pt across all run folders."""
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
    ap.add_argument("--patience", type=int, default=5, help="Early stopping patience for local training")
    ap.add_argument("--fl_rounds", type=int, default=5)
    ap.add_argument("--fl_local_epochs", type=int, default=1)
    ap.add_argument("--n_clients", type=int, default=3)
    ap.add_argument("--n_features", type=int, default=40)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--nprocs", type=int, default=4)
    ap.add_argument("--run_name", default="pipeline_local")
    # Skip flags
    ap.add_argument("--skip_local_train", action="store_true")
    ap.add_argument("--skip_fl", action="store_true")
    # Force flags (re-run even if output exists)
    ap.add_argument("--force_preprocess", action="store_true")
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
    clients_dir = PROJECT_ROOT / "data" / "processed" / "clients"
    test_index = clients_dir / f"client{args.n_clients}" / "index.pt"
    runs_dir = PROJECT_ROOT / "runs"
    global_best = PROJECT_ROOT / "server_out" / "global_best.pt"
    eval_fed = PROJECT_ROOT / "eval_results_federated.json"
    eval_local = PROJECT_ROOT / "eval_results_local.json"
    plot_out = PROJECT_ROOT / "model_comparison_plot.png"

    # ── Step 1: Preprocessing ──────────────────────────────────────────────────
    if not index_path.exists() or args.force_preprocess:
        _run([
            py, str(SRC_DIR / "parallel_preprocess.py"),
            "--raw_folder", args.raw_folder,
            "--out_folder", str(processed_dir),
            "--seq_len", str(args.seq_len),
            "--nprocs", str(args.nprocs),
        ], "Step 1/6 — Preprocessing raw PSV data", env)
    else:
        print(f"\n[PIPELINE] Step 1/6 — Skipping preprocessing (index exists: {index_path})")

    # ── Step 2: Local Training ─────────────────────────────────────────────────
    local_ckpt = None
    if not args.skip_local_train:
        existing = _find_latest_local_ckpt(runs_dir)
        if existing and not args.force_train:
            print(f"\n[PIPELINE] Step 2/6 — Skipping local training (checkpoint: {existing})")
            local_ckpt = existing
        else:
            _run([
                py, str(SRC_DIR / "train_local.py"),
                "--index", str(index_path),
                "--model", args.model,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--patience", str(args.patience),
                "--run_name", args.run_name,
            ], "Step 2/6 — Local training", env)
            local_ckpt = _find_latest_local_ckpt(runs_dir)
    else:
        print("\n[PIPELINE] Step 2/6 — Skipping local training (--skip_local_train)")

    # ── Step 3: Split Clients ──────────────────────────────────────────────────
    if not test_index.exists() or args.force_split:
        _run([
            py, str(SRC_DIR / "split_clients.py"),
            "--processed_folder", str(processed_dir),
            "--out_root", str(clients_dir),
            "--n_clients", str(args.n_clients),
        ], "Step 3/6 — Splitting data into federated client folders", env)
    else:
        print(f"\n[PIPELINE] Step 3/6 — Skipping client split (folders exist)")

    # ── Step 4: Federated Learning ─────────────────────────────────────────────
    if not args.skip_fl:
        if not global_best.exists() or args.force_fl:
            client_indexes = [
                str(clients_dir / f"client{i+1}" / "index.pt")
                for i in range(args.n_clients - 1)
            ]
            _run([
                py, str(SCRIPTS_DIR / "run_fl_sim.py"),
                "--client_indexes", *client_indexes,
                "--model", args.model,
                "--rounds", str(args.fl_rounds),
                "--local_epochs", str(args.fl_local_epochs),
                "--n_features", str(args.n_features),
                "--seq_len", str(args.seq_len),
            ], "Step 4/6 — Federated learning simulation", env)
        else:
            print(f"\n[PIPELINE] Step 4/6 — Skipping FL (global_best.pt exists: {global_best})")
    else:
        print("\n[PIPELINE] Step 4/6 — Skipping federated learning (--skip_fl)")

    # ── Step 5: Evaluation ─────────────────────────────────────────────────────
    evaluated = False
    if global_best.exists() and test_index.exists():
        if not eval_fed.exists() or args.force_eval:
            _run([
                py, str(SRC_DIR / "evaluate.py"),
                "--index", str(test_index),
                "--ckpt", str(global_best),
                "--model", args.model,
                "--n_features", str(args.n_features),
                "--seq_len", str(args.seq_len),
                "--out_file", str(eval_fed),
            ], "Step 5a/6 — Evaluating federated model", env)
            evaluated = True
        else:
            print(f"\n[PIPELINE] Step 5a/6 — Skipping federated eval (file exists: {eval_fed})")

    if local_ckpt and test_index.exists():
        if not eval_local.exists() or args.force_eval:
            _run([
                py, str(SRC_DIR / "evaluate.py"),
                "--index", str(test_index),
                "--ckpt", str(local_ckpt),
                "--model", args.model,
                "--n_features", str(args.n_features),
                "--seq_len", str(args.seq_len),
                "--out_file", str(eval_local),
            ], "Step 5b/6 — Evaluating local model", env)
            evaluated = True
        else:
            print(f"\n[PIPELINE] Step 5b/6 — Skipping local eval (file exists: {eval_local})")

    # ── Step 6: Plot ───────────────────────────────────────────────────────────
    result_files = [str(p) for p in [eval_fed, eval_local] if p.exists()]
    if result_files:
        _run([
            py, str(SRC_DIR / "plot_results.py"),
            "--results", *result_files,
            "--out_file", str(plot_out),
        ], "Step 6/6 — Generating ROC/PRC comparison plots", env)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[PIPELINE] All steps complete!")
    print(f"  Global best model : {global_best}")
    print(f"  Local checkpoint  : {local_ckpt or 'N/A'}")
    print(f"  Eval (federated)  : {eval_fed}")
    print(f"  Eval (local)      : {eval_local}")
    print(f"  Plots             : {plot_out}")
    print("\n  Launch dashboard:")
    print("    streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
