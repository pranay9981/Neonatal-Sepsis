"""
Optuna hyperparameter optimisation for neonatal sepsis models.

Replaces the grid-search in hyperparam_search.py with Bayesian optimisation.
Searches: lr, hidden_size (d_model for Transformer), dropout, n_heads.
Uses median pruning to kill poor trials early.

Usage:
  python scripts/optuna_search.py \\
    --index data/splits/train_index.pt --model transformer --n_trials 50
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    _OPTUNA_OK = True
except ImportError:
    _OPTUNA_OK = False

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from dataset import PatientDataset
from model import TimeSeriesTransformer
from model_grud import GRUD
from train_local import (
    seed_everything, stratified_train_val_split, collate_grud,
    get_num_workers, make_warmup_cosine_scheduler, safe_metrics,
    FocalLoss, _load_grud_empirical_mean,
)
from logging_config import get_logger

logger = get_logger(__name__)

import torch.nn as nn
import torch.optim as optim


def _build_model(trial, model_name: str, n_features: int, seq_len: int):
    if model_name == "transformer":
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        n_heads_choices = [h for h in [2, 4, 8] if d_model % h == 0]
        n_heads = trial.suggest_categorical("n_heads", n_heads_choices)
        dropout = trial.suggest_float("dropout", 0.05, 0.4)
        return TimeSeriesTransformer(
            n_features=n_features, seq_len=seq_len,
            d_model=d_model, n_heads=n_heads, dropout=dropout,
        )
    else:
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.05, 0.4)
        return GRUD(n_features=n_features, hidden_size=hidden_size, dropout=dropout)


def objective(trial, index_path: str, model_name: str, epochs: int, device: str):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    seed_everything(42 + trial.number)
    mode = "grud" if model_name == "grud" else "transformer"
    ds = PatientDataset(index_path, mode=mode)
    train_ds, val_ds, all_labels = stratified_train_val_split(ds, val_ratio=0.2)

    collate = collate_grud if model_name == "grud" else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)

    sample_x = ds[0][0]
    n_features, seq_len = sample_x.shape[1], sample_x.shape[0]
    model = _build_model(trial, model_name, n_features, seq_len).to(device)

    if model_name == "grud":
        xm = _load_grud_empirical_mean(index_path, model.n_features)
        if xm is not None:
            model.set_empirical_mean(xm)

    pos = sum(all_labels)
    neg = len(all_labels) - pos
    pos_weight = torch.tensor(
        [(neg / (pos + 1e-6)) if pos > 0 else 1.0], dtype=torch.float32
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = make_warmup_cosine_scheduler(opt, warmup_epochs=2, total_epochs=epochs)

    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            if model_name == "transformer":
                Xb, pm, yb = batch
                logits = model(Xb.to(device), src_key_padding_mask=pm.to(device))
            else:
                Xb, Mb, Db, yb = batch
                logits = model(Xb.to(device), Mb.to(device), Db.to(device))
            loss = loss_fn(logits, yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()

        model.eval()
        logits_all, y_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                if model_name == "transformer":
                    Xb, pm, yb = batch
                    logits_all.extend(model(Xb.to(device), src_key_padding_mask=pm.to(device)).cpu().tolist())
                else:
                    Xb, Mb, Db, yb = batch
                    logits_all.extend(model(Xb.to(device), Mb.to(device), Db.to(device)).cpu().tolist())
                y_all.extend(yb.tolist())

        auc, _ = safe_metrics(y_all, logits_all)
        best_auc = max(best_auc, auc)
        trial.report(auc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_auc


def run_optuna(
    index_path: str,
    model_name: str = "transformer",
    n_trials: int = 50,
    epochs: int = 10,
    device: str = "cpu",
    out_file: str | None = None,
    study_name: str = "sepsis_hpo",
):
    if not _OPTUNA_OK:
        print("optuna not installed. Run: pip install optuna"); return

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        lambda t: objective(t, index_path, model_name, epochs, device),
        n_trials=n_trials,
    )

    best = study.best_trial
    print(f"\nBest AUROC: {best.value:.4f}")
    print(f"Best params: {best.params}")

    result = {"best_auroc": best.value, "best_params": best.params,
              "n_trials": n_trials, "model": model_name}
    if out_file:
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_file}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(PROJECT_ROOT / "data" / "splits" / "train_index.pt"))
    ap.add_argument("--model", choices=["transformer", "grud"], default="transformer")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_file", default=str(PROJECT_ROOT / "optuna_results.json"))
    ap.add_argument("--study_name", default="sepsis_hpo")
    args = ap.parse_args()
    run_optuna(args.index, args.model, args.n_trials, args.epochs,
               args.device, args.out_file, args.study_name)
