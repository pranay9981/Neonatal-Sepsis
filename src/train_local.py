# src/train_local.py
"""
Local training script.

Improvements in this version:
- Patient-level stratified train/val split (no data leakage between patients).
- Pos_weight computed from all labels, not a sample estimate.
- Optional Focal Loss for severe class imbalance (--use_focal).
- Gradient clipping (max_norm=1.0) for training stability.
- Linear warmup + cosine decay LR schedule instead of ReduceLROnPlateau.
- Saves best AUROC checkpoint (model_best.pt) and best AUPRC checkpoint (model_best_ap.pt).
- Threshold calibration via Youden's J after training; saved to threshold.json.
- GRU-D empirical mean loaded from scaler.json when available.
- Structured logging throughout.
"""

import json
import math
import os
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from calibration import TemperatureScaler
from dataset import PatientDataset
from model import TimeSeriesTransformer
from model_grud import GRUD
from logging_config import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

_PROJECT_ROOT = Path(__file__).parent.parent


# -------------------------
# Collate for GRU-D (must be picklable for Windows multiprocessing)
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


def calibrate_threshold(val_logits, val_y):
    """
    Find the decision threshold that maximises Youden's J = sensitivity + specificity - 1.
    Returns 0.5 if calibration is not possible (single class or degenerate predictions).

    sklearn's roc_curve prepends an artificial boundary point at threshold=max(score)+1
    to anchor the curve at (fpr=0, tpr=0). We exclude thresholds outside [0, 1] to avoid
    returning that boundary value.
    """
    y = np.array(val_y)
    if len(np.unique(y)) < 2:
        return 0.5
    probs = torch.sigmoid(torch.tensor(val_logits, dtype=torch.float32)).numpy()
    fpr, tpr, thresholds = roc_curve(y, probs)
    j = tpr - fpr
    # Restrict to thresholds that are valid probabilities in [0, 1].
    valid = np.isfinite(thresholds) & (thresholds >= 0.0) & (thresholds <= 1.0)
    if not valid.any():
        return 0.5
    j_valid = np.where(valid, j, -np.inf)
    return float(thresholds[int(np.argmax(j_valid))])


def build_model_from_sample(sample_X, model_name, hidden_size=128, dropout=0.1):
    if model_name == "transformer":
        return TimeSeriesTransformer(n_features=sample_X.shape[1], seq_len=sample_X.shape[0])
    elif model_name == "grud":
        return GRUD(n_features=sample_X.shape[1], hidden_size=hidden_size, dropout=dropout)
    else:
        raise ValueError("Unknown model: " + model_name)


def _load_grud_empirical_mean(index_path: str, n_features: int) -> torch.Tensor | None:
    """Load per-feature training mean from scaler.json adjacent to the index."""
    for candidate in [
        os.path.join(os.path.dirname(index_path), "scaler.json"),
        os.path.join(os.path.dirname(index_path), "..", "scaler.json"),
    ]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                s = json.load(f)
            mean = s.get("mean", [])
            if len(mean) == n_features:
                logger.info("Loaded GRU-D empirical mean from %s", candidate)
                return torch.tensor(mean, dtype=torch.float32)
            logger.warning(
                "scaler.json has %d features but model expects %d; skipping.", len(mean), n_features
            )
    logger.warning("scaler.json not found near %s; GRU-D will use zero mean.", index_path)
    return None


# -------------------------
# Loss functions
# -------------------------
class FocalLoss(nn.Module):
    """
    Focal loss for binary classification.
    Down-weights easy examples so training focuses on hard, minority-class samples.
    gamma=2 is the value from the original paper (Lin et al., 2017).
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        return ((1.0 - p_t) ** self.gamma * bce).mean()


# -------------------------
# LR schedule
# -------------------------
def make_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup for `warmup_epochs` then cosine decay to 0."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# -------------------------
# Early stopping
# -------------------------
class EarlyStopping:
    """Stop training when val_auc stops improving for `patience` consecutive epochs."""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best = -float("inf")
        self.wait = 0
        self.stopped_epoch = 0

    def step(self, metric: float, epoch: int) -> bool:
        if metric > self.best:
            self.best = metric
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False


# -------------------------
# Stratified split
# -------------------------
def stratified_train_val_split(ds: PatientDataset, val_ratio: float = 0.2, seed: int = 42):
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
    patience: int = 5,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    warmup_epochs: int = 3,
    clip_grad: float = 1.0,
    use_mlflow: bool = False,
    mlflow_experiment: str = "neonatal_sepsis",
    scaler_path: str | None = None,
    augment: bool = False,
    use_temperature_scaling: bool = False,
    hidden_size: int = 128,
    dropout: float = 0.1,
):
    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    if checkpoint_root is None:
        checkpoint_root = str(_PROJECT_ROOT / "runs")

    # MLflow setup (optional)
    mlflow_run = None
    if use_mlflow:
        if not _MLFLOW_AVAILABLE:
            logger.warning("mlflow not installed; skipping experiment tracking. pip install mlflow")
            use_mlflow = False
        else:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment(mlflow_experiment)
            mlflow_run = mlflow.start_run(run_name=run_name)

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
        "use_focal": use_focal,
        "focal_gamma": focal_gamma,
        "warmup_epochs": warmup_epochs,
        "clip_grad": clip_grad,
        "scaler_path": scaler_path,
    }
    with open(os.path.join(run_folder, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=2)

    if use_mlflow and mlflow_run:
        mlflow.log_params(run_info)

    mode = "grud" if model_name == "grud" else "transformer"
    ds = PatientDataset(index_path, mode=mode, scaler_path=scaler_path, augment=augment)
    n = len(ds)
    if n < 2:
        raise ValueError(f"Need at least 2 patients, found {n}")

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

    sample = ds[0]
    sample_X = sample[0]
    model = build_model_from_sample(sample_X, model_name, hidden_size=hidden_size, dropout=dropout).to(device)

    # For GRU-D: load empirical mean so the decay targets the training distribution.
    if model_name == "grud":
        x_mean = _load_grud_empirical_mean(index_path, model.n_features)
        if x_mean is not None:
            model.set_empirical_mean(x_mean)

    pos_weight = torch.tensor(
        [(neg / (pos + 1e-6)) if pos > 0 else 1.0],
        dtype=torch.float32,
    ).to(device)

    if use_focal:
        loss_fn = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
        logger.info("Using Focal Loss (gamma=%.1f)", focal_gamma)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = make_warmup_cosine_scheduler(opt, warmup_epochs=warmup_epochs, total_epochs=epochs)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    metrics_csv = os.path.join(run_folder, "metrics.csv")
    with open(metrics_csv, "w") as fh:
        fh.write("epoch,train_loss,val_auc,val_ap,lr\n")

    best_auc = -1.0
    best_ap = -1.0
    early_stop = EarlyStopping(patience=patience)

    # Track val predictions from the best-AUROC epoch for threshold calibration.
    best_val_logits: list = []
    best_val_y: list = []
    final_val_logits, final_val_y = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            opt.zero_grad()
            if model_name == "transformer":
                Xb, pad_mask_b, yb = batch
                Xb, pad_mask_b, yb = Xb.to(device), pad_mask_b.to(device), yb.to(device)
                with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                    logits = model(Xb, src_key_padding_mask=pad_mask_b)
                    loss = loss_fn(logits, yb)
            else:
                Xb, Mb, Db, yb = batch
                Xb, Mb, Db, yb = Xb.to(device), Mb.to(device), Db.to(device), yb.to(device)
                with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                    logits = model(Xb, Mb, Db)
                    loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
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
                    Xb, pad_mask_b, yb = batch
                    with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                        logits = model(Xb.to(device), src_key_padding_mask=pad_mask_b.to(device)).cpu().numpy()
                    val_logits.extend(logits.tolist())
                    val_y.extend(yb.numpy().tolist())
                else:
                    Xb, Mb, Db, yb = batch
                    with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                        logits = model(Xb.to(device), Mb.to(device), Db.to(device)).cpu().numpy()
                    val_logits.extend(logits.tolist())
                    val_y.extend(yb.numpy().tolist())

        auc, ap = safe_metrics(val_y, val_logits)

        scheduler.step()
        current_lr = opt.param_groups[0]["lr"]

        torch.save(
            {"model_state": model.state_dict(), "epoch": epoch, "auc": auc, "ap": ap},
            os.path.join(ckpt_dir, f"model_epoch{epoch}.pt"),
        )
        if auc > best_auc:
            best_auc = auc
            best_val_logits, best_val_y = val_logits[:], val_y[:]
            torch.save({"model_state": model.state_dict()}, os.path.join(ckpt_dir, "model_best.pt"))
        if ap > best_ap:
            best_ap = ap
            torch.save({"model_state": model.state_dict()}, os.path.join(ckpt_dir, "model_best_ap.pt"))

        # Keep last epoch's val predictions for threshold calibration.
        final_val_logits, final_val_y = val_logits, val_y

        with open(metrics_csv, "a") as fh:
            fh.write(f"{epoch},{train_loss:.6f},{auc:.6f},{ap:.6f},{current_lr:.8f}\n")

        if use_mlflow and mlflow_run:
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_auc": auc, "val_ap": ap, "lr": current_lr},
                step=epoch,
            )

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_auc=%.4f  val_ap=%.4f  lr=%.2e",
            epoch, epochs, train_loss, auc, ap, current_lr,
        )

        if early_stop.step(auc, epoch):
            logger.info(
                "Early stopping at epoch %d (no AUROC improvement for %d epochs). Best=%.4f",
                epoch, patience, best_auc,
            )
            break

    # Threshold calibration on the best-AUROC epoch's val predictions (matches model_best.pt).
    # Fall back to final epoch if best was never recorded (e.g., no improvement at all).
    cal_logits = best_val_logits if best_val_logits else final_val_logits
    cal_y = best_val_y if best_val_y else final_val_y
    threshold = calibrate_threshold(cal_logits, cal_y)
    threshold_path = os.path.join(run_folder, "threshold.json")
    with open(threshold_path, "w") as fh:
        json.dump({"threshold": threshold, "method": "youden_j"}, fh, indent=2)

    if use_temperature_scaling and len(cal_logits) >= 2 and len(np.unique(cal_y)) >= 2:
        ts = TemperatureScaler()
        ts.fit(np.array(cal_logits), np.array(cal_y))
        ts.save(threshold_path)
        logger.info("Temperature scaling fitted: T=%.4f", ts.temperature)

    logger.info("Calibrated decision threshold: %.4f (saved to %s)", threshold, threshold_path)
    logger.info("Training complete. Best AUROC=%.4f  Best AUPRC=%.4f", best_auc, best_ap)
    logger.info("Run folder: %s", run_folder)

    if use_mlflow and mlflow_run:
        mlflow.log_metrics({"best_auroc": best_auc, "best_auprc": best_ap, "threshold": threshold})
        try:
            mlflow.pytorch.log_model(model, "model")
        except Exception as e:
            logger.warning("Could not log model to MLflow: %s", e)
        mlflow.end_run()


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
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--use_focal", action="store_true", help="Use Focal Loss instead of BCE")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--warmup_epochs", type=int, default=3, help="Linear LR warmup epochs")
    ap.add_argument("--clip_grad", type=float, default=1.0, help="Max gradient norm (0 to disable)")
    ap.add_argument("--use_mlflow", action="store_true", help="Enable MLflow experiment tracking")
    ap.add_argument("--mlflow_experiment", type=str, default="neonatal_sepsis")
    ap.add_argument("--scaler_path", type=str, default=None, help="Path to scaler.json for feature normalisation")
    ap.add_argument("--augment", action="store_true", help="Enable on-the-fly Gaussian jitter augmentation")
    ap.add_argument("--use_temperature_scaling", action="store_true", help="Apply temperature scaling after training")
    ap.add_argument("--hidden_size", type=int, default=128, help="GRU-D hidden size (ignored for transformer)")
    ap.add_argument("--dropout", type=float, default=0.1, help="GRU-D dropout rate (ignored for transformer)")
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
        patience=args.patience,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        warmup_epochs=args.warmup_epochs,
        clip_grad=args.clip_grad,
        use_mlflow=args.use_mlflow,
        mlflow_experiment=args.mlflow_experiment,
        scaler_path=args.scaler_path,
        augment=args.augment,
        use_temperature_scaling=args.use_temperature_scaling,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )
