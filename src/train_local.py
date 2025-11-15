# src/train_local.py
"""
Training script updated for Windows-compatible dataloader workers.

Key fixes:
- collate_grud moved to module scope (picklable).
- num_workers defaults to 0 on Windows to avoid spawn/pickle issues.
- All previous features retained: transformer/grud, AMP, scheduler, run folders, logging.
"""

import os
import json
import math
import random
import argparse
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import PatientDataset
from model import TimeSeriesTransformer
from model_grud import GRUD

# -------------------------
# Module-level collate for GRU-D
# -------------------------
def collate_grud(batch):
    """
    Top-level collate function so it can be pickled on Windows.
    Expects batch items: (X (T,F), mask (T,F), deltas (T,F), y)
    Returns stacked tensors: X (B,T,F), mask (B,T,F), deltas (B,T,F), y (B,)
    """
    Xs, masks, deltas, ys = zip(*batch)
    X = torch.stack(Xs)
    mask = torch.stack(masks)
    delta = torch.stack(deltas)
    y = torch.stack(ys)
    return X, mask, delta, y

# -------------------------
# Utilities
# -------------------------
def get_num_workers(preferred: int = 4):
    """
    Return number of DataLoader workers. On Windows use 0 to avoid pickling problems
    unless user explicitly sets environment or platform supports forkserver.
    """
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
        n_features = sample_X.shape[1]
        seq_len = sample_X.shape[0]
        return TimeSeriesTransformer(n_features=n_features, seq_len=seq_len)
    elif model_name == "grud":
        n_features = sample_X.shape[1]
        return GRUD(n_features=n_features, hidden_size=128)
    else:
        raise ValueError("Unknown model: " + model_name)

# -------------------------
# Training
# -------------------------
def train(index_path: str,
          model_name: str = "transformer",
          epochs: int = 10,
          batch_size: int = 64,
          lr: float = 1e-4,
          seed: int = 42,
          run_name: str = "run",
          device: str | None = None,
          checkpoint_root: str = "runs",
          preferred_workers: int = 4):

    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    # timezone-aware timestamp
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # run folder
    run_folder = os.path.join(checkpoint_root, f"{ts}__{run_name}")
    ckpt_dir = os.path.join(run_folder, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # save run info
    run_info = {
        "index_path": index_path,
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "device": device,
        "timestamp_utc": ts,
        "preferred_workers": preferred_workers
    }
    os.makedirs(run_folder, exist_ok=True)
    with open(os.path.join(run_folder, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=2)

    # --- Dataset ---
    ds = PatientDataset(index_path, mode="grud" if model_name == "grud" else "transformer")
    n = len(ds)

    if n < 2:
        raise ValueError(f"Need at least 2 patients to train/validate, found {n}")

    n_train = int(0.8 * n)
    n_val = n - n_train
    if n_val == 0:
        n_train = max(1, n_train - 1)
        n_val = n - n_train

    train_ds, val_ds = random_split(ds, [n_train, n_val])

    # determine num_workers
    num_workers = get_num_workers(preferred=preferred_workers)

    # dataloaders
    if model_name == "transformer":
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=min(2, num_workers))
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_grud)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=min(2, num_workers), collate_fn=collate_grud)

    # --- Model ---
    if model_name == "transformer":
        sample_X, _ = ds[0]
    else:
        sample_X, _, _, _ = ds[0]
    model = build_model_from_sample(sample_X, model_name).to(device)

    # --- Class imbalance ---
    ys = []
    sample_n = min(n, 2000)
    for i in range(sample_n):
        try:
            it = ds[i]
            y = it[-1] if model_name == "grud" else it[1]
            ys.append(float(y))
        except Exception:
            pass
    pos = sum(ys)
    neg = len(ys) - pos
    pos_weight = torch.tensor([(neg / (pos + 1e-6)) if pos > 0 else 1.0], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # optimizer & scheduler
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5)

    # AMP scaler (compatible)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # metrics logging
    metrics_csv = os.path.join(run_folder, "metrics.csv")
    with open(metrics_csv, "w") as fh:
        fh.write("epoch,train_loss,val_auc,val_ap,lr\n")

    best_auc = 0.0

    # training loop
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

        # validation
        model.eval()
        val_logits = []
        val_y = []

        with torch.no_grad():
            for batch in val_loader:
                if model_name == "transformer":
                    Xb, yb = batch
                    Xb = Xb.to(device)
                    with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                        logits = model(Xb).cpu().numpy()
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
            print(f"[LR Scheduler] LR reduced: {old_lr} -> {new_lr}")

        # save checkpoint
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "auc": auc}, os.path.join(ckpt_dir, f"model_epoch{epoch}.pt"))
        if auc > best_auc:
            best_auc = auc
            torch.save({"model_state": model.state_dict()}, os.path.join(ckpt_dir, "model_best.pt"))

        with open(metrics_csv, "a") as fh:
            fh.write(f"{epoch},{train_loss:.6f},{auc:.6f},{ap:.6f},{new_lr:.8f}\n")

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_auc={auc:.4f} val_ap={ap:.4f}")

    print("Training complete. Best AUROC =", best_auc)
    print("Run folder:", run_folder)


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
    ap.add_argument("--workers", type=int, default=4, help="Preferred number of dataloader workers (0 on Windows)")
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
        checkpoint_root="runs",
        preferred_workers=args.workers
    )
