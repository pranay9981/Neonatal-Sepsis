# src/split_clients.py
"""
Split processed patients into federated client folders.
Uses stratified splitting to ensure balanced class distribution across clients.

Pass --splits_dir to exclude the frozen test set from client partitions,
preventing data leakage between FL training and the held-out test evaluation.
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from logging_config import get_logger
from utils import ensure_dir

logger = get_logger(__name__)


def split_into_clients(
    processed_folder: str,
    out_root: str,
    n_clients: int = 3,
    seed: int = 42,
    splits_dir: str | None = None,
    heterogeneous: bool = False,
) -> list:
    ensure_dir(out_root)
    idx_path = os.path.join(processed_folder, "index_with_labels.pt")
    # I-04: These index files contain plain Python lists/strings, not arbitrary
    # objects, so weights_only=False is required (weights_only=True only handles
    # tensors and a small set of primitives — lists of strings are excluded).
    d = torch.load(idx_path, weights_only=False)  # nosec: index file, no tensors
    x_paths = d["x_paths"]
    labels = [int(float(y)) for y in d.get("y", [])]
    n_total = len(x_paths)

    # Exclude test patients when frozen splits are available.
    if splits_dir is not None:
        test_idx_path = os.path.join(splits_dir, "test_index.pt")
        if os.path.exists(test_idx_path):
            test_d = torch.load(test_idx_path, weights_only=False)  # nosec: index file
            # W-11: Normalise paths via pathlib so mixed-slash paths on Windows
            # (e.g. from different OS origins) compare equal and test patients
            # are never silently leaked into the FL partition.
            test_set = {str(Path(p).resolve()) for p in test_d.get("x_paths", [])}
            filtered = [(p, l) for p, l in zip(x_paths, labels)
                        if str(Path(p).resolve()) not in test_set]
            if filtered:
                x_paths, labels = zip(*filtered)
                x_paths, labels = list(x_paths), list(labels)
                logger.info(
                    "Excluded %d test patients; %d remain for FL partitioning.",
                    n_total - len(x_paths),
                    len(x_paths),
                )
        else:
            logger.warning("splits_dir provided but test_index.pt not found: %s", test_idx_path)

    n = len(x_paths)
    if len(labels) != n:
        logger.warning(
            "Labels length %d != paths length %d; falling back to random split.", len(labels), n
        )
        rng = np.random.default_rng(seed)
        order = rng.permutation(n).tolist()
        x_paths = [x_paths[i] for i in order]
        labels = [0] * n
        splits = _even_split(list(range(n)), n_clients)
    elif heterogeneous:
        splits = _heterogeneous_split(x_paths, labels, n_clients, seed)
    else:
        splits = _stratified_split(x_paths, labels, n_clients, seed)

    clients = []
    for i, indices in enumerate(splits):
        client_folder = os.path.join(out_root, f"client{i + 1}")
        ensure_dir(client_folder)
        selected = [x_paths[j] for j in indices]
        new_paths = []
        for p in selected:
            from pathlib import Path as _Path
            dst = os.path.join(client_folder, os.path.basename(p))
            if os.path.exists(dst):
                stem, suffix = _Path(p).stem, _Path(p).suffix
                counter = 1
                while os.path.exists(dst):
                    dst = os.path.join(client_folder, f"{stem}_{counter}{suffix}")
                    counter += 1
            shutil.copy(p, dst)
            new_paths.append(dst)
        client_labels = [labels[j] for j in indices]
        torch.save({"x_paths": new_paths, "y": client_labels}, os.path.join(client_folder, "index.pt"))
        pos = sum(client_labels)
        logger.info(
            "Client %d: %d patients | %d positive (%.1f%%)",
            i + 1, len(selected), pos, 100 * pos / max(1, len(selected)),
        )
        clients.append(client_folder)

    logger.info("Split %d patients into %d clients.", n, n_clients)
    return clients


def _stratified_split(x_paths, labels, n_clients, seed):
    arr = np.arange(len(x_paths))
    lbl = np.array(labels)
    unique_cls = np.unique(lbl)
    if len(unique_cls) < 2:
        logger.warning("Only one class present — cannot stratify, using random split.")
        return _even_split(arr.tolist(), n_clients, seed=seed)
    if n_clients < 2:
        return [arr.tolist()]
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=seed)
    splits = [[] for _ in range(n_clients)]
    for fold_idx, (_, test_idx) in enumerate(skf.split(arr, lbl)):
        splits[fold_idx] = test_idx.tolist()
    return splits


def _heterogeneous_split(x_paths, labels, n_clients, seed):
    """
    Simulate non-IID hospital data by sorting patients by label prevalence bucket,
    then assigning each contiguous block to a client.
    This creates clients with skewed class distributions — a more realistic
    simulation of real-world federated hospital data.
    """
    rng = np.random.default_rng(seed)
    n = len(x_paths)
    arr = np.arange(n)
    lbl = np.array(labels)

    # Sort: all positives first in shuffled order, then negatives
    pos_idx = rng.permutation(arr[lbl == 1]).tolist()
    neg_idx = rng.permutation(arr[lbl == 0]).tolist()
    ordered = pos_idx + neg_idx  # non-IID ordering

    size = n // n_clients
    splits = []
    for i in range(n_clients):
        start = i * size
        end = start + size if i < n_clients - 1 else n
        splits.append(ordered[start:end])
    return splits


def _even_split(indices, n_clients, seed=42):
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
    ap.add_argument(
        "--splits_dir",
        default=None,
        help="Path to data/splits/ folder; if provided, test patients are excluded from FL partitions.",
    )
    ap.add_argument(
        "--heterogeneous",
        action="store_true",
        help="Simulate non-IID hospital distributions (positives skewed across clients).",
    )
    args = ap.parse_args()
    split_into_clients(
        args.processed_folder,
        args.out_root,
        args.n_clients,
        args.seed,
        splits_dir=args.splits_dir,
        heterogeneous=args.heterogeneous,
    )
