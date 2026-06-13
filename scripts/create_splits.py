"""
Create frozen 3-way patient-level stratified split: 70% train / 15% val / 15% test.
Saves index files to data/splits/: train_index.pt, val_index.pt, test_index.pt.

Run once after preprocessing. The test set is frozen and must never be used
for training, hyperparameter selection, or threshold calibration.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

_PROJECT_ROOT = Path(__file__).parent.parent


def create_splits(
    index_path: str,
    out_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = torch.load(index_path, weights_only=False)
    x_paths = d["x_paths"]
    labels = [int(float(y)) for y in d.get("y", [])]
    n = len(x_paths)

    if len(labels) != n:
        raise ValueError(f"x_paths ({n}) and labels ({len(labels)}) length mismatch")

    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    print(f"Splitting {n} patients -> train={train_ratio:.0%} / val={val_ratio:.0%} / test={test_ratio:.0%}")

    # Step 1: carve out the test set
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.arange(n), labels))

    # Step 2: split remaining into train / val
    trainval_labels = [labels[i] for i in trainval_idx]
    val_frac = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_sub, val_sub = next(sss2.split(np.arange(len(trainval_idx)), trainval_labels))

    train_idx = [trainval_idx[i] for i in train_sub]
    val_idx = [trainval_idx[i] for i in val_sub]

    for name, idx_list in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_paths = [x_paths[i] for i in idx_list]
        split_labels = [labels[i] for i in idx_list]
        out_path = out_dir / f"{name}_index.pt"
        torch.save({"x_paths": split_paths, "y": split_labels}, out_path)
        pos = sum(split_labels)
        print(f"  {name:5s}: {len(split_paths):5d} patients | {pos:4d} positive ({100 * pos / max(1, len(split_labels)):.1f}%)")

    print(f"\nFrozen splits saved to: {out_dir}")
    print("  IMPORTANT: test_index.pt must never be used during training or model selection.")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create frozen 70/15/15 patient-level stratified split.")
    ap.add_argument(
        "--index",
        default=str(_PROJECT_ROOT / "data" / "processed" / "patients" / "index_with_labels.pt"),
    )
    ap.add_argument("--out_dir", default=str(_PROJECT_ROOT / "data" / "splits"))
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    create_splits(args.index, args.out_dir, args.train_ratio, args.val_ratio, args.seed)
