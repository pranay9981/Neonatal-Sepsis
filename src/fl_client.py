# src/fl_client.py
"""
Flower client with:
- FedProx proximal regularisation (--mu, default 0.01; set 0.0 to revert to plain FedAvg).
- Train/val split on the client's local data (80/20 stratified) — evaluation now uses
  a held-out set rather than the training set.
- Gradient clipping for training stability.
- AUROC and AUPRC metrics reported per round.

Usage:
  python src/fl_client.py \
    --index data/processed/clients/client1/index.pt \
    --server_address 127.0.0.1:8080 \
    --model transformer \
    --device cpu \
    --local_epochs 1 \
    --batch_size 32 \
    --mu 0.01
"""
import argparse
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

import flwr as fl

try:
    from dataset import PatientDataset
except Exception:
    PatientDataset = None

try:
    from model import TimeSeriesTransformer
except Exception:
    TimeSeriesTransformer = None

try:
    from model_grud import GRUD
except Exception:
    GRUD = None

from logging_config import get_logger

logger = get_logger(__name__)


# -----------------------
# State-dict ↔ ndarray helpers
# -----------------------
def state_dict_to_ndarrays(sd: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    return [v.cpu().numpy() for v in sd.values()]


def ndarrays_to_state_dict_by_order(
    model: torch.nn.Module, arrays: List[np.ndarray]
) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    keys = list(sd.keys())
    map_len = min(len(keys), len(arrays))
    new_sd = {}
    for k, arr in zip(keys[:map_len], arrays[:map_len]):
        t = torch.tensor(arr)
        if t.shape != sd[k].shape:
            try:
                t = t.view(sd[k].shape)
            except Exception:
                if t.ndim == 2 and tuple(t.T.shape) == tuple(sd[k].shape):
                    t = t.T
                else:
                    raise RuntimeError(
                        f"Cannot map array for key {k}: got {tuple(t.shape)}, expected {tuple(sd[k].shape)}"
                    )
        new_sd[k] = t
    sd.update(new_sd)
    return sd


# -----------------------
# FedProx proximal term
# -----------------------
def proximal_term(
    model: torch.nn.Module,
    global_tensors: List[torch.Tensor],
    mu: float,
) -> torch.Tensor:
    """(mu/2) * sum ||w_local - w_global||^2 across all parameter tensors."""
    prox = sum(
        torch.sum((p - g.to(p.device)) ** 2)
        for p, g in zip(model.parameters(), global_tensors)
    )
    return (mu / 2.0) * prox


# -----------------------
# Model factory
# -----------------------
def build_model(model_name: str, n_features: int, seq_len: int, device: str):
    if model_name == "transformer":
        assert TimeSeriesTransformer is not None
        return TimeSeriesTransformer(n_features=n_features, seq_len=seq_len).to(device)
    elif model_name == "grud":
        assert GRUD is not None
        return GRUD(n_features=n_features).to(device)
    else:
        raise ValueError("Unknown model: " + str(model_name))


def _load_grud_empirical_mean(index_path: str, n_features: int) -> Optional[torch.Tensor]:
    for candidate in [
        os.path.join(os.path.dirname(index_path), "scaler.json"),
        os.path.join(os.path.dirname(index_path), "..", "scaler.json"),
    ]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                s = json.load(f)
            mean = s.get("mean", [])
            if len(mean) == n_features:
                return torch.tensor(mean, dtype=torch.float32)
    return None


# -----------------------
# Flower client
# -----------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        index_path: str,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        lr: float = 1e-3,
        local_epochs: int = 1,
        n_features: Optional[int] = None,
        seq_len: Optional[int] = None,
        mu: float = 0.01,
        clip_grad: float = 1.0,
    ):
        assert PatientDataset is not None
        self.index_path = index_path
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs
        self.mu = mu
        self.clip_grad = clip_grad
        self.global_tensors: Optional[List[torch.Tensor]] = None

        ds_mode = "transformer" if model_name == "transformer" else "grud"
        ds = PatientDataset(index_path, mode=ds_mode)

        # Infer n_features / seq_len from the first sample.
        if n_features is None:
            try:
                x0 = ds[0][0]
                arr = np.asarray(x0)
                if arr.ndim == 2:
                    seq_len = seq_len or arr.shape[0]
                    n_features = arr.shape[1]
                else:
                    n_features = arr.shape[-1]
                    seq_len = seq_len or 48
            except Exception:
                n_features = n_features or 40
                seq_len = seq_len or 48
        if seq_len is None:
            seq_len = 48

        self.n_features = int(n_features)
        self.seq_len = int(seq_len)

        # Stratified 80/20 train/val split on the client's data.
        train_loader, eval_loader = self._make_loaders(ds)
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.model = build_model(self.model_name, self.n_features, self.seq_len, self.device)

        # Load empirical mean for GRU-D.
        if model_name == "grud":
            x_mean = _load_grud_empirical_mean(index_path, self.n_features)
            if x_mean is not None:
                self.model.set_empirical_mean(x_mean)

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

        logger.info(
            "FlowerClient ready: n_features=%d, seq_len=%d, train=%d, eval=%d, mu=%.3f",
            self.n_features, self.seq_len,
            len(self.train_loader.dataset), len(self.eval_loader.dataset), mu,
        )

    def _make_loaders(self, ds: PatientDataset):
        """Stratified 80/20 split; fall back to random if not enough samples."""
        n = len(ds)
        labels = [int(float(ds.y_indexed[i])) if ds.y_indexed else 0 for i in range(n)]
        n_val = max(1, int(0.2 * n))
        n_train = n - n_val

        use_grud = self.model_name == "grud"

        def _collate_grud(batch):
            Xs, masks, deltas, ys = zip(*batch)
            return torch.stack(Xs), torch.stack(masks), torch.stack(deltas), torch.stack(ys)

        collate = _collate_grud if use_grud else None

        unique = set(labels)
        if len(unique) >= 2 and n_val >= len(unique):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(range(n), labels))
        else:
            import random as _rnd
            idx = list(range(n))
            _rnd.shuffle(idx)
            train_idx, val_idx = idx[:n_train], idx[n_train:]

        train_ds = Subset(ds, list(train_idx))
        val_ds = Subset(ds, list(val_idx))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
        eval_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        return train_loader, eval_loader

    # ----- Flower NumPyClient interface -----

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return state_dict_to_ndarrays(self.model.state_dict())

    def set_parameters(self, arrays: List[np.ndarray]):
        sd = ndarrays_to_state_dict_by_order(self.model, arrays)
        self.model.load_state_dict(sd)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        if parameters is not None:
            try:
                self.set_parameters(parameters)
            except Exception as e:
                logger.warning("Could not set incoming parameters: %s", e)

        # Store global params as fixed tensors for the proximal term.
        # Filter to only trainable parameter positions (skip buffers like x_mean)
        # so the zip in proximal_term stays aligned with model.parameters().
        if self.mu > 0 and parameters is not None:
            param_names = {name for name, _ in self.model.named_parameters()}
            sd_keys = list(self.model.state_dict().keys())
            self.global_tensors = [
                torch.tensor(arr, dtype=torch.float32)
                for key, arr in zip(sd_keys, parameters)
                if key in param_names
            ]
        else:
            self.global_tensors = None

        self.model.train()
        device = torch.device(self.device)

        for epoch in range(self.local_epochs):
            running_loss = 0.0
            n_samples = 0
            for batch in self.train_loader:
                if self.model_name == "transformer":
                    Xb, pad_mask_b, yb = batch
                    Xb = Xb.to(device).float()
                    pad_mask_b = pad_mask_b.to(device)
                    yb = yb.to(device).float().view(-1)
                    logits = self.model(Xb, src_key_padding_mask=pad_mask_b)
                else:
                    Xb, Mb, Db, yb = batch
                    Xb = Xb.to(device).float()
                    Mb = Mb.to(device).float()
                    Db = Db.to(device).float()
                    yb = yb.to(device).float().view(-1)
                    logits = self.model(Xb, Mb, Db)

                loss = self.loss_fn(logits.view(-1), yb)

                # FedProx proximal regularisation.
                if self.global_tensors is not None and self.mu > 0:
                    loss = loss + proximal_term(self.model, self.global_tensors, self.mu)

                self.opt.zero_grad()
                loss.backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.opt.step()

                running_loss += float(loss.item()) * len(yb)
                n_samples += len(yb)

            if n_samples > 0:
                logger.info(
                    "[TRAIN] epoch %d/%d  loss=%.4f", epoch + 1, self.local_epochs, running_loss / n_samples
                )

        metrics = self.evaluate_local()
        return state_dict_to_ndarrays(self.model.state_dict()), len(self.train_loader.dataset), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        if parameters is not None:
            try:
                self.set_parameters(parameters)
            except Exception as e:
                logger.warning("Could not set parameters in evaluate: %s", e)

        metrics = self.evaluate_local()
        n = len(self.eval_loader.dataset)
        loss = metrics.pop("loss", float("nan"))
        return loss, n, metrics

    def evaluate_local(self) -> Dict[str, float]:
        self.model.eval()
        device = torch.device(self.device)
        ys, preds = [], []
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                if self.model_name == "transformer":
                    Xb, pad_mask_b, yb = batch
                    Xb = Xb.to(device).float()
                    pad_mask_b = pad_mask_b.to(device)
                    logits = self.model(Xb, src_key_padding_mask=pad_mask_b).view(-1)
                    yb = yb.to(device).float().view(-1)
                else:
                    Xb, Mb, Db, yb = batch
                    Xb, Mb, Db = Xb.to(device).float(), Mb.to(device).float(), Db.to(device).float()
                    logits = self.model(Xb, Mb, Db).view(-1)
                    yb = yb.to(device).float().view(-1)

                loss = self.loss_fn(logits, yb)
                ys.append(yb.cpu().numpy().reshape(-1))
                preds.append(logits.cpu().numpy().reshape(-1))
                total_loss += float(loss.item()) * len(yb)
                n_samples += len(yb)

        if n_samples == 0:
            return {"loss": float("nan"), "auroc": float("nan"), "auprc": float("nan")}

        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        probs = 1.0 / (1.0 + np.exp(-preds))
        metrics = {"loss": total_loss / n_samples}

        if len(set(ys.tolist())) > 1:
            try:
                metrics["auroc"] = float(roc_auc_score(ys, probs))
            except Exception:
                metrics["auroc"] = float("nan")
            try:
                metrics["auprc"] = float(average_precision_score(ys, probs))
            except Exception:
                metrics["auprc"] = float("nan")
        else:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")

        return metrics


# -----------------------
# CLI entry point
# -----------------------
def start_client(
    index_path: str,
    server_address: str,
    model_name: str,
    device: str = "cpu",
    batch_size: int = 32,
    lr: float = 1e-3,
    local_epochs: int = 1,
    n_features: Optional[int] = None,
    seq_len: Optional[int] = None,
    mu: float = 0.01,
    clip_grad: float = 1.0,
    max_retries: int = 20,
    retry_delay: float = 2.0,
):
    client = FlowerClient(
        index_path=index_path,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        lr=lr,
        local_epochs=local_epochs,
        n_features=n_features,
        seq_len=seq_len,
        mu=mu,
        clip_grad=clip_grad,
    )
    client_obj = client.to_client()
    attempt = 0
    while True:
        attempt += 1
        try:
            logger.info("Connecting to %s (attempt %d)...", server_address, attempt)
            fl.client.start_client(server_address=server_address, client=client_obj)
            break
        except Exception as e:
            logger.warning("start_client attempt %d failed: %s", attempt, e)
            if attempt >= max_retries:
                raise
            time.sleep(retry_delay)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--server_address", required=True)
    ap.add_argument("--model", choices=["transformer", "grud"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--n_features", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--mu", type=float, default=0.01, help="FedProx proximal term weight (0=plain FedAvg)")
    ap.add_argument("--clip_grad", type=float, default=1.0, help="Max gradient norm (0 to disable)")
    args = ap.parse_args()

    start_client(
        index_path=args.index,
        server_address=args.server_address,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        lr=args.lr,
        local_epochs=args.local_epochs,
        n_features=args.n_features,
        seq_len=args.seq_len,
        mu=args.mu,
        clip_grad=args.clip_grad,
    )
