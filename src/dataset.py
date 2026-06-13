# src/dataset.py
import torch
from torch.utils.data import Dataset
import os, glob
import pickle
import numpy as np

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
    - mode='grud': yields (X_filled, mask, deltas, y)
        Uses pre-computed mask/deltas stored by parallel_preprocess.py when available;
        falls back to NaN-detection for older .pt files.
    """

    def __init__(self, index_path, mode='transformer'):
        d = torch.load(index_path, weights_only=False)
        self.x_paths = d.get('x_paths', [])
        self.y_indexed = None
        if 'y' in d:
            self.y_indexed = d['y']
            if hasattr(self.y_indexed, 'tolist'):
                self.y_indexed = self.y_indexed.tolist()
        self.mode = mode

    def __len__(self):
        return len(self.x_paths)

    def _load_pt(self, path):
        return torch.load(path, weights_only=False)

    def _load_lmdb(self, lmdb_spec):
        assert lmdb_spec.startswith("lmdb://")
        s = lmdb_spec[len("lmdb://"):]
        path, key = s.split("#", 1)
        env = lmdb.open(path, readonly=True, lock=False)
        with env.begin() as txn:
            raw = txn.get(key.encode('utf-8'))
            obj = pickle.loads(raw)
        env.close()
        return obj

    def __getitem__(self, idx):
        spec = self.x_paths[idx]
        if isinstance(spec, str) and spec.startswith("lmdb://"):
            data = self._load_lmdb(spec)
        else:
            data = self._load_pt(spec)

        X = data['X'].float()  # (T, F)
        T = X.shape[0]
        y = float(data.get('y', 0))
        if self.y_indexed is not None:
            y = float(self.y_indexed[idx])

        if self.mode == 'transformer':
            # Build padding mask: True = ignore (front-padded zeros).
            actual_len = int(data.get('actual_len', T))
            pad_mask = torch.zeros(T, dtype=torch.bool)
            if actual_len < T:
                pad_mask[:T - actual_len] = True
            return X, pad_mask, torch.tensor(y, dtype=torch.float32)

        elif self.mode == 'grud':
            if 'mask' in data and 'deltas' in data:
                # Fast path: pre-computed by preprocessing pipeline.
                mask = data['mask'].float()
                deltas = data['deltas'].float()
                X_filled = X.clone()
                X_filled[torch.isnan(X_filled)] = 0.0
            else:
                # Fallback for older .pt files: derive from NaN positions.
                mask = (~torch.isnan(X)).float()
                X_filled = X.clone()
                X_filled[torch.isnan(X_filled)] = 0.0
                deltas = _compute_deltas_fallback(mask)
            return X_filled, mask, deltas, torch.tensor(y, dtype=torch.float32)

        else:
            raise ValueError("Unknown dataset mode: " + self.mode)
