# src/dataset.py
import torch
from torch.utils.data import Dataset
import os, glob
import lmdb
import pickle
import numpy as np

class PatientDataset(Dataset):
    """
    Supports:
    - index file with 'x_paths' and optional 'y'
    - LMDB shards: index entries pointing to lmdb://<path>#<key>
    - If model='grud' the dataset yields (X, mask, deltas, y)
    """
    def __init__(self, index_path, mode='transformer'):
        d = torch.load(index_path)
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
        return torch.load(path)

    def _load_lmdb(self, lmdb_spec):
        # lmdb_spec: "lmdb://path/to/shard.lmdb#key"
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
        y = float(data.get('y', 0))
        if self.y_indexed is not None:
            y = float(self.y_indexed[idx])
        if self.mode == 'transformer':
            return X, torch.tensor(y, dtype=torch.float32)
        elif self.mode == 'grud':
            # create mask and deltas
            mask = (~torch.isnan(X)).float()
            X_filled = X.clone()
            # forward/backward fill for imputations used by model (we keep X_filled for GRU-D)
            # simple fill with 0 for NaNs (GRU-D handles mask)
            X_filled[torch.isnan(X_filled)] = 0.0
            # compute deltas: time since last observation per feature
            # we don't have timestamps here; assume uniform hourly spacing -> delta 1 where mask=1 else accumulate
            T, F = X.shape
            deltas = torch.zeros_like(X_filled)
            last_seen = torch.zeros(X_filled.size(0), F) if False else None
            # simple per-step delta: 1 when observation present else increase
            dt = torch.ones(T, F)
            for t in range(T):
                if t == 0:
                    deltas[t] = torch.zeros(F)
                else:
                    deltas[t] = deltas[t-1] + 1.0
                    deltas[t] = deltas[t] * (1.0 - mask[t]) + 0.0 * mask[t]
            # reshape deltas to (T,F)
            # Make shapes (B,T,F) at collate time; here return T x F
            return X_filled, mask, deltas, torch.tensor(y, dtype=torch.float32)
        else:
            raise ValueError("Unknown dataset mode: " + self.mode)
