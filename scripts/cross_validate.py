"""
5-fold stratified cross-validation with bootstrap confidence intervals.

Each fold trains on 4/5 of train+val patients and evaluates on the remaining 1/5.
After all folds: reports mean ± std AUROC/AUPRC and 95% bootstrap CIs.

Usage:
  python scripts/cross_validate.py \\
    --index data/splits/train_index.pt \\
    --model transformer --epochs 10 --n_folds 5
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import PatientDataset
from train_local import train, seed_everything
from logging_config import get_logger


def _fit_fold_scaler(x_paths, indices, base_scaler_path, fold_scaler_path):
    """Fit a StandardScaler on the fold training patients only and write it to fold_scaler_path.

    Reads each patient's X tensor, accumulates per-feature mean/std, and writes a
    scaler.json in the same format as the project-wide scaler.  Falls back to the
    project-wide scaler when a per-patient file cannot be loaded.
    """
    x_sum = None
    x_sq_sum = None
    x_cnt = None

    for i in indices:
        try:
            data = torch.load(x_paths[i], weights_only=False)
        except Exception:
            continue
        X = data["X"].numpy().astype(np.float64)  # (T, F)
        mask = (~np.isnan(X)).astype(np.float64)
        X_valid = np.where(mask, X, 0.0)
        if x_sum is None:
            F = X.shape[1]
            x_sum = np.zeros(F, dtype=np.float64)
            x_sq_sum = np.zeros(F, dtype=np.float64)
            x_cnt = np.zeros(F, dtype=np.float64)
        x_sum += X_valid.sum(axis=0)
        x_sq_sum += (X_valid ** 2).sum(axis=0)
        x_cnt += mask.sum(axis=0)

    if x_sum is None or x_cnt is None or (x_cnt == 0).all():
        # Nothing to fit — fall back to the global scaler
        if base_scaler_path and os.path.exists(base_scaler_path):
            import shutil
            shutil.copy2(base_scaler_path, fold_scaler_path)
        return

    mean = x_sum / np.where(x_cnt > 0, x_cnt, 1.0)
    var = x_sq_sum / np.where(x_cnt > 0, x_cnt, 1.0) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    # Where we have no observations fall back to global scaler values if available
    if base_scaler_path and os.path.exists(base_scaler_path):
        with open(base_scaler_path) as f:
            global_sc = json.load(f)
        global_mean = np.array(global_sc.get("mean", [0.0] * len(mean)))
        global_std = np.array(global_sc.get("std", [1.0] * len(std)))
        mean = np.where(x_cnt > 0, mean, global_mean)
        std = np.where(x_cnt > 0, std, global_std)

    scaler_dict = {"mean": mean.tolist(), "std": std.tolist()}
    with open(fold_scaler_path, "w") as f:
        json.dump(scaler_dict, f)


logger = get_logger(__name__)


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, yp))
        except Exception:
            pass
    if not scores:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(scores, 100 * alpha)), float(np.percentile(scores, 100 * (1 - alpha)))


def run_cv(
    index_path: str,
    model_name: str = "transformer",
    n_folds: int = 5,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
    out_file: str | None = None,
    scaler_path: str | None = None,
):
    seed_everything(seed)
    d = torch.load(index_path, weights_only=False)
    x_paths = d["x_paths"]
    labels = np.array([int(float(y)) for y in d.get("y", [])])
    n = len(x_paths)

    logger.info("Starting %d-fold CV on %d patients ...", n_folds, n)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for fold, (train_sub, val_sub) in enumerate(skf.split(np.arange(n), labels), start=1):
            logger.info("Fold %d/%d  train=%d  val=%d", fold, n_folds, len(train_sub), len(val_sub))

            # Write temporary index files for this fold
            train_idx_path = os.path.join(tmpdir, f"fold{fold}_train.pt")
            val_idx_path = os.path.join(tmpdir, f"fold{fold}_val.pt")
            torch.save(
                {"x_paths": [x_paths[i] for i in train_sub], "y": labels[train_sub].tolist()},
                train_idx_path,
            )
            torch.save(
                {"x_paths": [x_paths[i] for i in val_sub], "y": labels[val_sub].tolist()},
                val_idx_path,
            )

            # C-12 / W-45: Fit scaler ONLY on fold train patients to prevent data leakage.
            # The fold val patients must not influence the scaler at any point.
            fold_scaler_path = os.path.join(tmpdir, f"fold{fold}_scaler.json")
            _fit_fold_scaler(x_paths, train_sub, scaler_path, fold_scaler_path)
            logger.info("Fold %d: scaler fitted on %d train patients -> %s", fold, len(train_sub), fold_scaler_path)

            run_folder_root = os.path.join(tmpdir, f"runs_fold{fold}")
            train(
                index_path=train_idx_path,
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed + fold,
                run_name=f"cv_fold{fold}",
                checkpoint_root=run_folder_root,
                patience=patience,
                scaler_path=fold_scaler_path,
            )

            # Find best checkpoint from this fold
            ckpts = sorted(
                Path(run_folder_root).glob("*/checkpoints/model_best.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not ckpts:
                logger.warning("Fold %d: no checkpoint found, skipping.", fold)
                continue
            best_ckpt = ckpts[0]

            # W-45: Pass fold-specific scaler to evaluate_single_ckpt so val features
            # are normalised with the same fold-train statistics (not raw/unnormalised).
            from evaluate import evaluate_single_ckpt
            result = evaluate_single_ckpt(
                index_path=val_idx_path,
                ckpt_path=str(best_ckpt),
                model_name=model_name,
                scaler_path=fold_scaler_path,
            )
            auroc = result.get("auroc") or 0.0
            auprc = result.get("auprc") or 0.0
            fold_entry = {"fold": fold, "auroc": auroc, "auprc": auprc, "n_val": len(val_sub)}
            if "y_true" in result and "y_prob" in result:
                fold_entry["y_true"] = result["y_true"]
                fold_entry["y_prob"] = result["y_prob"]
            fold_results.append(fold_entry)
            logger.info("Fold %d:  AUROC=%.4f  AUPRC=%.4f", fold, auroc, auprc)

    if not fold_results:
        logger.error("No fold results collected.")
        return

    aucs = np.array([r["auroc"] for r in fold_results])
    aps = np.array([r["auprc"] for r in fold_results])

    # Bootstrap CI on aggregated fold predictions (if available)
    agg_yt = np.concatenate([r["y_true"] for r in fold_results if "y_true" in r]) if any("y_true" in r for r in fold_results) else None
    agg_yp = np.concatenate([r["y_prob"] for r in fold_results if "y_prob" in r]) if any("y_prob" in r for r in fold_results) else None
    if agg_yt is not None and len(np.unique(agg_yt)) >= 2:
        auroc_lo, auroc_hi = bootstrap_ci(agg_yt, agg_yp, roc_auc_score)
        auprc_lo, auprc_hi = bootstrap_ci(agg_yt, agg_yp, average_precision_score)
    else:
        auroc_lo = auroc_hi = auprc_lo = auprc_hi = float("nan")

    # Strip y_true/y_prob from fold_results before serialising (large arrays)
    for r in fold_results:
        r.pop("y_true", None)
        r.pop("y_prob", None)

    summary = {
        "n_folds": n_folds,
        "model": model_name,
        "auroc_mean": float(aucs.mean()),
        "auroc_std": float(aucs.std()),
        "auroc_ci_95_lo": auroc_lo,
        "auroc_ci_95_hi": auroc_hi,
        "auprc_mean": float(aps.mean()),
        "auprc_std": float(aps.std()),
        "auprc_ci_95_lo": auprc_lo,
        "auprc_ci_95_hi": auprc_hi,
        "folds": fold_results,
    }

    print("\n" + "=" * 50)
    print(f"Cross-Validation Results ({n_folds} folds)")
    print(f"  AUROC:  {aucs.mean():.4f} ± {aucs.std():.4f}  [95% CI: {auroc_lo:.4f}–{auroc_hi:.4f}]")
    print(f"  AUPRC:  {aps.mean():.4f} ± {aps.std():.4f}  [95% CI: {auprc_lo:.4f}–{auprc_hi:.4f}]")
    print("=" * 50)

    if out_file:
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("CV summary saved to %s", out_file)

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="5-fold stratified cross-validation.")
    ap.add_argument(
        "--index",
        default=str(PROJECT_ROOT / "data" / "splits" / "train_index.pt"),
    )
    ap.add_argument("--model", choices=["transformer", "grud"], default="transformer")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_file", default=str(PROJECT_ROOT / "cv_results.json"))
    ap.add_argument("--scaler_path", default=None)
    args = ap.parse_args()

    run_cv(
        index_path=args.index,
        model_name=args.model,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        out_file=args.out_file,
        scaler_path=args.scaler_path,
    )
