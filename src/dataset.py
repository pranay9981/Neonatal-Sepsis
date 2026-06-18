# src/dataset.py
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import lmdb
except ImportError:
    lmdb = None


def _compute_deltas_fallback(mask: torch.Tensor) -> torch.Tensor:
    """Fallback delta computation for .pt files that predate the preprocessing fix."""
    T, F = mask.shape
    deltas = torch.zeros(T, F)
    for t in range(1, T):
        deltas[t] = (deltas[t - 1] + 1.0) * (1.0 - mask[t])
    return deltas


class PatientDataset(Dataset):
    """
    Supports:
    - index file with 'x_paths' and optional 'y'
    - LMDB shards: index entries pointing to lmdb://<path>#<key>
    - mode='transformer': yields (X, pad_mask, y)
        pad_mask: (T,) bool tensor — True at front-padded (invalid) positions.
    - mode='grud': yields (X_filled, mask, deltas, actual_len, y)
        Uses pre-computed mask/deltas stored by parallel_preprocess.py when available;
        falls back to NaN-detection for older .pt files.

    scaler_path: optional path to scaler.json (mean/std per feature).
        When provided, X is normalised as (X - mean) / std in __getitem__.
        Padded positions beyond actual_len are zeroed out after normalisation
        so the scaler shift does not corrupt padding zeros.

    augment: if True, apply on-the-fly time-series augmentation (jitter + window slicing).
        Should only be enabled during training, not evaluation.
    """

    # Per-process LMDB environment cache. Keyed by (pid, lmdb_path) to avoid
    # pickling issues with DataLoader workers (each worker has its own pid).
    _lmdb_envs: dict = {}

    def __init__(self, index_path, mode="transformer", scaler_path=None, augment=False):
        d = torch.load(index_path, weights_only=False)
        self.x_paths = d.get("x_paths", [])
        self.y_indexed = None
        if "y" in d:
            self.y_indexed = d["y"]
            if hasattr(self.y_indexed, "tolist"):
                self.y_indexed = self.y_indexed.tolist()
        self.mode = mode
        self.augment = augment

        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        if scaler_path is not None and os.path.exists(scaler_path):
            with open(scaler_path) as f:
                s = json.load(f)
            self._mean = torch.tensor(s["mean"], dtype=torch.float32)
            self._std = torch.tensor(s["std"], dtype=torch.float32).clamp(min=1e-8)

    def __len__(self):
        return len(self.x_paths)

    def _load_pt(self, path):
        return torch.load(path, weights_only=True)

    def _get_lmdb_env(self, lmdb_path: str):
        """Return a cached lmdb.Environment for lmdb_path, one per process.

        A new environment is opened when the process id changes (e.g. DataLoader
        worker fork) so we never share file descriptors across processes.
        """
        import os as _os
        pid = _os.getpid()
        key = (pid, lmdb_path)
        if key not in self.__class__._lmdb_envs:
            self.__class__._lmdb_envs[key] = lmdb.open(
                lmdb_path, readonly=True, lock=False, readahead=False
            )
        return self.__class__._lmdb_envs[key]

    def _load_lmdb(self, lmdb_spec):
        assert lmdb_spec.startswith("lmdb://")
        s = lmdb_spec[len("lmdb://"):]
        path, key = s.split("#", 1)
        env = self._get_lmdb_env(path)
        with env.begin() as txn:
            raw = txn.get(key.encode("utf-8"))
        if raw is None:
            raise KeyError(f"Key {key!r} not found in LMDB at {path}")
        return pickle.loads(raw)

    def _apply_scaler(self, X: torch.Tensor) -> torch.Tensor:
        if self._mean is None:
            return X
        if self._mean.shape[0] != X.shape[-1]:
            raise ValueError(
                f"Scaler has {self._mean.shape[0]} features but tensor has {X.shape[-1]} — "
                "check that scaler.json matches the preprocessed data."
            )
        return (X - self._mean.to(X.device)) / self._std.to(X.device)

    def _augment(self, X: torch.Tensor) -> torch.Tensor:
        """Light on-the-fly augmentation: Gaussian jitter."""
        if not self.augment:
            return X
        X = X + torch.randn_like(X) * 0.05
        return X

    def __getitem__(self, idx):
        spec = self.x_paths[idx]
        if isinstance(spec, str) and spec.startswith("lmdb://"):
            data = self._load_lmdb(spec)
        else:
            data = self._load_pt(spec)

        X = data["X"].float()  # (T, F)
        T = X.shape[0]
        y = float(data.get("y", 0))
        if self.y_indexed is not None:
            y = float(self.y_indexed[idx])

        if self.mode == "transformer":
            actual_len = int(data.get("actual_len", T))
            pad_mask = torch.zeros(T, dtype=torch.bool)
            if actual_len < T:
                pad_mask[: T - actual_len] = True

            # Scaler is applied before augmentation to ensure augmented features remain in z-score space.
            X = self._apply_scaler(X)
            if actual_len < T:
                X[: T - actual_len] = 0.0  # restore zeros after normalization shifts them
            X = self._augment(X)
            return X, pad_mask, torch.tensor(y, dtype=torch.float32)

        elif self.mode == "grud":
            actual_len = int(data.get("actual_len", T))
            if "mask" in data and "deltas" in data:
                mask = data["mask"].float()
                deltas = data["deltas"].float()
                X_filled = X.clone()
                X_filled[torch.isnan(X_filled)] = 0.0
            else:
                mask = (~torch.isnan(X)).float()
                X_filled = X.clone()
                X_filled[torch.isnan(X_filled)] = 0.0
                deltas = _compute_deltas_fallback(mask)

            # Scaler applied before augmentation; padded positions zeroed AFTER
            # normalisation so the scaler mean-shift does not corrupt padding zeros.
            X_filled = self._apply_scaler(X_filled)
            if actual_len < T:
                X_filled[: T - actual_len] = 0.0
                mask[: T - actual_len] = 0.0
                deltas[: T - actual_len] = 0.0
            X_filled = self._augment(X_filled)
            return X_filled, mask, deltas, actual_len, torch.tensor(y, dtype=torch.float32)

        else:
            raise ValueError("Unknown dataset mode: " + self.mode)
