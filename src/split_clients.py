# src/split_clients.py
"""
Split processed patients into federated client folders.
Uses stratified splitting to ensure balanced class distribution across clients.
"""
import os
import shutil
import argparse
from math import floor
from pathlib import Path

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils import ensure_dir
from logging_config import get_logger

logger = get_logger(__name__)


def split_into_clients(processed_folder, out_root, n_clients=3, seed=42):
    ensure_dir(out_root)
    idx_path = os.path.join(processed_folder, "index_with_labels.pt")
    d = torch.load(idx_path, weights_only=False)
    x_paths = d["x_paths"]
    labels = [int(float(y)) for y in d.get("y", [])]

    n = len(x_paths)
    if len(labels) != n:
        logger.warning("Labels length %d != paths length %d; falling back to random split.", len(labels), n)
        rng = np.random.default_rng(seed)
        order = rng.permutation(n).tolist()
        x_paths = [x_paths[i] for i in order]
        labels = [0] * n
        splits = _even_split(list(range(n)), n_clients)
    else:
        splits = _stratified_split(x_paths, labels, n_clients, seed)

    clients = []
    for i, indices in enumerate(splits):
        client_folder = os.path.join(out_root, f"client{i+1}")
        ensure_dir(client_folder)
        selected = [x_paths[j] for j in indices]
        for p in selected:
            shutil.copy(p, client_folder)
        new_paths = [os.path.join(client_folder, os.path.basename(p)) for p in selected]
        client_labels = [labels[j] for j in indices]
        torch.save({"x_paths": new_paths, "y": client_labels}, os.path.join(client_folder, "index.pt"))
        pos = sum(client_labels)
        logger.info("Client %d: %d patients | %d positive (%.1f%%)", i+1, len(selected), pos, 100*pos/max(1,len(selected)))
        clients.append(client_folder)

    logger.info("Split %d patients into %d clients.", n, n_clients)
    return clients


def _stratified_split(x_paths, labels, n_clients, seed):
    """Return list of n_clients index lists, stratified by label."""
    arr = np.arange(len(x_paths))
    lbl = np.array(labels)

    unique_cls = np.unique(lbl)
    if len(unique_cls) < 2:
        logger.warning("Only one class present — cannot stratify, using random split.")
        return _even_split(arr.tolist(), n_clients, seed=seed)

    # Use StratifiedKFold as an easy way to get n balanced folds
    if n_clients < 2:
        return [arr.tolist()]

    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=seed)
    splits = [[] for _ in range(n_clients)]

    # StratifiedKFold produces n_clients (train, test) pairs; we use the test splits
    for fold_idx, (_, test_idx) in enumerate(skf.split(arr, lbl)):
        splits[fold_idx] = test_idx.tolist()

    return splits


def _even_split(indices, n_clients, seed=42):
    """Randomly split indices into n_clients roughly equal groups."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(indices)).tolist()
    shuffled = [indices[i] for i in order]
    size = len(shuffled) // n_clients
    splits = []
    cur = 0
    for i in range(n_clients):
        count = size + (1 if i < len(shuffled) % n_clients else 0)
        splits.append(shuffled[cur: cur + count])
        cur += count
    return splits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_folder", required=True)
    ap.add_argument("--out_root", default="data/processed/clients")
    ap.add_argument("--n_clients", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split_into_clients(args.processed_folder, args.out_root, args.n_clients, args.seed)
