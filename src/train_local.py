# src/train_local.py
"""
Local training script.
Key improvements over original:
- Uses patient-level stratified train/val split (no data leakage between patients).
- Computes pos_weight from all labels, not a 2000-sample estimate.
- Replaces print() with structured logging.
- Uses absolute default paths anchored to project root.
"""

import os
import json
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

from dataset import PatientDataset
from model import TimeSeriesTransformer
from model_grud import GRUD
from logging_config import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent


# -------------------------
# Module-level collate for GRU-D (must be picklable for Windows multiprocessing)
# -------------------------
def collate_grud(batch):
    Xs, masks, deltas, ys = zip(*batch)
    return torch.stack(Xs), torch.stack(masks), torch.stack(deltas), torch.stack(ys)


# -------------------------
# Utilities
# -------------------------
def get_num_workers(preferred: int = 4) -> int:
    if os.name == "nt":
        return 0
    return max(0, preferred)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_metrics(y_true, logits):
    y_true = np.array(y_true)
    if y_true.size == 0:
        return 0.0, 0.0
    try:
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        if len(np.unique(y_true)) == 1:
            return 0.0, 0.0
        auc = roc_auc_score(y_true, probs)
        ap = average_precision_score(y_true, probs)
        return float(auc), float(ap)
    except Exception:
        return 0.0, 0.0


def build_model_from_sample(sample_X, model_name):
    if model_name == "transformer":
        return TimeSeriesTransformer(n_features=sample_X.shape[1], seq_len=sample_X.shape[0])
    elif model_name == "grud":
        return GRUD(n_features=sample_X.shape[1], hidden_size=128)
    else:
        raise ValueError("Unknown model: " + model_name)


def stratified_train_val_split(ds: PatientDataset, val_ratio: float = 0.2, seed: int = 42):
    """
    Patient-level stratified split using labels stored in the index file.
    Falls back to random split if stratification is not possible.
    """
    labels_raw = ds.y_indexed if ds.y_indexed is not None else [float(ds[i][-1]) for i in range(len(ds))]
    labels = [int(float(l)) for l in labels_raw]

    n = len(ds)
    n_val = max(1, int(val_ratio * n))
    n_train = n - n_val

    unique = set(labels)
    if len(unique) < 2 or n_val < len(unique):
        logger.warning("Cannot stratify (classes=%d, n_val=%d) — falling back to random split.", len(unique), n_val)
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        return train_ds, val_ds, labels

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(range(n), labels))
    return Subset(ds, list(train_idx)), Subset(ds, list(val_idx)), labels


# -------------------------
# Training
# -------------------------
def train(
    index_path: str,
    model_name: str = "transformer",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 42,
    run_name: str = "run",
    device: str | None = None,
    checkpoint_root: str | None = None,
    preferred_workers: int = 4,
):
    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    if checkpoint_root is None:
        checkpoint_root = str(_PROJECT_ROOT / "runs")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_folder = os.path.join(checkpoint_root, f"{ts}__{run_name}")
    ckpt_dir = os.path.join(run_folder, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    run_info = {
        "index_path": index_path,
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "device": device,
        "timestamp_utc": ts,
        "preferred_workers": preferred_workers,
    }
    with open(os.path.join(run_folder, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=2)

    # Dataset
    mode = "grud" if model_name == "grud" else "transformer"
    ds = PatientDataset(index_path, mode=mode)
    n = len(ds)
    if n < 2:
        raise ValueError(f"Need at least 2 patients, found {n}")

    # Stratified split + full-dataset pos_weight
    train_ds, val_ds, all_labels = stratified_train_val_split(ds, val_ratio=0.2, seed=seed)
    pos = sum(all_labels)
    neg = n - pos
    logger.info("Dataset: %d patients | %d positive (%.1f%%) | %d negative", n, pos, 100 * pos / n, neg)
    logger.info("Split: %d train | %d val", len(train_ds), len(val_ds))

    num_workers = get_num_workers(preferred=preferred_workers)
    if model_name == "transformer":
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=min(2, num_workers))
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_grud)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=min(2, num_workers), collate_fn=collate_grud)

    # Model
    sample = ds[0]
    sample_X = sample[0]
    model = build_model_from_sample(sample_X, model_name).to(device)

    pos_weight = torch.tensor(
        [(neg / (pos + 1e-6)) if pos > 0 else 1.0],
        dtype=torch.float32,
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    metrics_csv = os.path.join(run_folder, "metrics.csv")
    with open(metrics_csv, "w") as fh:
        fh.write("epoch,train_loss,val_auc,val_ap,lr\n")

    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            if model_name == "transformer":
                Xb, yb = batch
                Xb, yb = Xb.to(device), yb.to(device)
                with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                    logits = model(Xb)
                    loss = loss_fn(logits, yb)
            else:
                Xb, Mb, Db, yb = batch
                Xb, Mb, Db, yb = Xb.to(device), Mb.to(device), Db.to(device), yb.to(device)
                with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                    logits = model(Xb, Mb, Db)
                    loss = loss_fn(logits, yb)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += float(loss.item()) * Xb.size(0)
            seen += Xb.size(0)

        train_loss = running_loss / max(1, seen)

        model.eval()
        val_logits, val_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                if model_name == "transformer":
                    Xb, yb = batch
                    with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                        logits = model(Xb.to(device)).cpu().numpy()
                    val_logits.extend(logits.tolist())
                    val_y.extend(yb.numpy().tolist())
                else:
                    Xb, Mb, Db, yb = batch
                    with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                        logits = model(Xb.to(device), Mb.to(device), Db.to(device)).cpu().numpy()
                    val_logits.extend(logits.tolist())
                    val_y.extend(yb.numpy().tolist())

        auc, ap = safe_metrics(val_y, val_logits)

        old_lr = opt.param_groups[0]["lr"]
        scheduler.step(auc)
        new_lr = opt.param_groups[0]["lr"]
        if new_lr != old_lr:
            logger.info("LR reduced: %.2e -> %.2e", old_lr, new_lr)

        torch.save(
            {"model_state": model.state_dict(), "epoch": epoch, "auc": auc},
            os.path.join(ckpt_dir, f"model_epoch{epoch}.pt"),
        )
        if auc > best_auc:
            best_auc = auc
            torch.save({"model_state": model.state_dict()}, os.path.join(ckpt_dir, "model_best.pt"))

        with open(metrics_csv, "a") as fh:
            fh.write(f"{epoch},{train_loss:.6f},{auc:.6f},{ap:.6f},{new_lr:.8f}\n")

        logger.info("Epoch %d/%d  train_loss=%.4f  val_auc=%.4f  val_ap=%.4f", epoch, epochs, train_loss, auc, ap)

    logger.info("Training complete. Best AUROC = %.4f", best_auc)
    logger.info("Run folder: %s", run_folder)


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--model", choices=["transformer", "grud"], default="transformer")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", type=str, default="train")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    train(
        index_path=args.index,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        run_name=args.run_name,
        device=args.device,
        preferred_workers=args.workers,
    )
