"""
Evaluate the Ensemble model (Transformer + GRU-D blended) on a patient index.

Usage:
  python scripts/eval_ensemble.py \
    --index data/splits/test_index.pt \
    --transformer_ckpt runs/20260613T153606Z__local_transformer/checkpoints/model_best.pt \
    --grud_ckpt runs/20260614T064920Z__local_grud/checkpoints/model_best.pt \
    --out_file eval_results_ensemble.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from ensemble import load_ensemble


class EnsembleDataset(Dataset):
    """
    Returns (X, pad_mask, mask, deltas, y) in one pass — everything
    both the Transformer and GRU-D need simultaneously.
    """

    def __init__(self, index_path: str):
        d = torch.load(index_path, weights_only=True)
        self.x_paths = d.get("x_paths", [])
        self.y_indexed = d.get("y", None)
        if self.y_indexed is not None and hasattr(self.y_indexed, "tolist"):
            self.y_indexed = self.y_indexed.tolist()

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        data = torch.load(self.x_paths[idx], weights_only=True)
        X = data["X"].float()
        T, F = X.shape
        y = float(self.y_indexed[idx]) if self.y_indexed is not None else float(data.get("y", 0))

        # Transformer pad_mask: True at front-padded (invalid) timesteps
        actual_len = int(data.get("actual_len", T))
        pad_mask = torch.zeros(T, dtype=torch.bool)
        if actual_len < T:
            pad_mask[: T - actual_len] = True

        # GRU-D mask + deltas
        if "mask" in data and "deltas" in data:
            mask = data["mask"].float()
            deltas = data["deltas"].float()
        else:
            mask = (~torch.isnan(X)).float()
            deltas = torch.zeros_like(X)

        X_filled = X.clone()
        X_filled[torch.isnan(X_filled)] = 0.0

        return X_filled, pad_mask, mask, deltas, torch.tensor(y, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description="Evaluate Transformer+GRU-D ensemble on a patient index")
    ap.add_argument("--index",            required=True,  help="Path to index .pt file (e.g. data/splits/test_index.pt)")
    ap.add_argument("--transformer_ckpt", required=True,  help="Path to Transformer model_best.pt")
    ap.add_argument("--grud_ckpt",        required=True,  help="Path to GRU-D model_best.pt")
    ap.add_argument("--out_file",         default="eval_results_ensemble.json")
    ap.add_argument("--alpha",            type=float, default=0.5, help="Transformer blend weight (default 0.5)")
    ap.add_argument("--device",           default="cpu")
    args = ap.parse_args()

    print(f"[ENSEMBLE] Loading models (alpha={args.alpha}) ...")
    model = load_ensemble(
        transformer_ckpt=args.transformer_ckpt,
        grud_ckpt=args.grud_ckpt,
        alpha=args.alpha,
        device=args.device,
    )
    model.eval()

    ds = EnsembleDataset(args.index)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"[ENSEMBLE] Evaluating {len(ds)} patients ...")

    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for Xb, pad_mask_b, Mb, Db, yb in loader:
            Xb         = Xb.to(args.device).float()
            pad_mask_b = pad_mask_b.to(args.device)
            Mb         = Mb.to(args.device).float()
            Db         = Db.to(args.device).float()
            probs = model(Xb, pad_mask=pad_mask_b, mask=Mb, deltas=Db)
            y_true_all.append(yb.numpy())
            y_prob_all.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)

    if len(np.unique(y_true)) < 2:
        print("[ENSEMBLE] Warning: test set has only one class — skipping AUC metrics.")
        auroc = auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    print(f"[ENSEMBLE] samples={len(y_true)}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    result = {
        "model_name": "ensemble",
        "auroc": auroc,
        "auprc": auprc,
        "n": int(len(y_true)),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
    }
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ENSEMBLE] Saved -> {args.out_file}")


if __name__ == "__main__":
    main()
