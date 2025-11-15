# src/evaluate.py
"""
Evaluate script that accepts either a .pt checkpoint or a .npz produced by the FL server.
Saves detailed metrics and predictions to a JSON file for plotting.

Usage examples:
  python src/evaluate.py --index data/processed/clients/client3/index.pt --ckpt server_out/global_best.pt --model transformer --n_features 40 --seq_len 48 --out_file eval_results.json
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import json

from dataset import PatientDataset

# Try importing models (must exist in repo)
try:
    from model import TimeSeriesTransformer
except Exception:
    TimeSeriesTransformer = None

try:
    from model_grud import GRUD
except Exception:
    GRUD = None

# -------------------------
# Model builder
# -------------------------
def build_model_for_eval(model_name: str, n_features: int = None, seq_len: int = None):
    if model_name == "transformer":
        assert TimeSeriesTransformer is not None, "TimeSeriesTransformer not importable"
        kwargs = {}
        if n_features is not None:
            kwargs["n_features"] = n_features
        if seq_len is not None:
            kwargs["seq_len"] = seq_len
        return TimeSeriesTransformer(**kwargs)
    elif model_name == "grud":
        assert GRUD is not None, "GRUD not importable"
        kwargs = {}
        if n_features is not None:
            kwargs["n_features"] = n_features
        return GRUD(**kwargs)
    else:
        raise ValueError("Unknown model: " + str(model_name))

# -------------------------
# NPZ loading helpers
# (These functions remain unchanged)
# -------------------------
def load_npz_to_arrays(npz_path: str):
    """Return ordered list of arrays and dict file->array for npz file."""
    npz = np.load(npz_path)
    arrays = [npz[f] for f in npz.files]
    files = list(npz.files)
    return arrays, files

def try_ordered_map(model, arrays):
    """
    Attempt to map arrays -> model.state_dict() keys in order.
    Returns resulting state_dict (or raises).
    """
    sd = model.state_dict()
    keys = list(sd.keys())
    map_len = min(len(keys), len(arrays))
    new_sd = {}
    for k, arr in zip(keys[:map_len], arrays[:map_len]):
        t = torch.tensor(arr)
        target = sd[k]
        if t.shape != target.shape:
            # If number of elements match, reshape
            if t.numel() == target.numel():
                try:
                    t = t.view(target.shape)
                except Exception:
                    raise RuntimeError(f"reshape failed for key {k}: {t.shape} -> {target.shape}")
            else:
                # shapes mismatch and number elements differ; raise to let caller try other strategies
                raise RuntimeError(f"shape mismatch for key {k}: array {t.shape} vs target {tuple(target.shape)}")
        new_sd[k] = t
    sd.update(new_sd)
    return sd

def resample_1d_along_axis(src, target_len):
    """
    Resample 2D array src (T_src, D) to (T_tgt, D) by linear interpolation
    along axis 0. src can be 1D (T,) or 2D (T, D). Returns float64 array.
    """
    src = np.asarray(src)
    if src.ndim == 1:
        src = src[:, None]
    T_src, D = src.shape
    if T_src == target_len:
        return src.squeeze() if src.shape[1] == 1 else src
    # new positions
    src_x = np.linspace(0.0, 1.0, T_src)
    tgt_x = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, D), dtype=src.dtype)
    for d in range(D):
        out[:, d] = np.interp(tgt_x, src_x, src[:, d])
    if out.shape[1] == 1:
        return out.squeeze()
    return out

def try_smart_map(model, arrays, files):
    """
    Try to map arrays -> model.state_dict by heuristics:
      - If ordered mapping failed, attempt best-effort mapping:
        - If an array can be reshaped to a key (matching element count), use it.
        - If a key is a positional embedding-like tensor (first dim mismatch),
          attempt to resample along first dimension.
    Returns state_dict on success or raises.
    """
    sd = model.state_dict()
    keys = list(sd.keys())

    # Build available arrays metadata
    arr_meta = []
    for idx, a in enumerate(arrays):
        arr_meta.append({"idx": idx, "shape": tuple(a.shape), "nelems": int(np.prod(a.shape)), "file": files[idx] if idx < len(files) else f"arr_{idx}"})

    new_sd = {}
    used = set()

    # First pass: exact shape matches preferred
    for k in keys:
        tgt_shape = tuple(sd[k].shape)
        found = None
        for m in arr_meta:
            if m["idx"] in used:
                continue
            if m["shape"] == tgt_shape:
                found = m; break
        if found is not None:
            new_sd[k] = torch.tensor(arrays[found["idx"]])
            used.add(found["idx"])

    # Second pass: same element count -> reshape
    for k in keys:
        if k in new_sd:
            continue
        tgt_shape = tuple(sd[k].shape)
        tgt_nelems = int(np.prod(tgt_shape))
        cand = None
        for m in arr_meta:
            if m["idx"] in used:
                continue
            if m["nelems"] == tgt_nelems:
                cand = m; break
        if cand is not None:
            a = arrays[cand["idx"]]
            try:
                t = torch.tensor(a).view(tgt_shape)
                new_sd[k] = t
                used.add(cand["idx"])
            except Exception:
                # cannot reshape, continue
                pass

    # Third pass: positional embedding style handling (first-dim mismatch)
    for k in keys:
        if k in new_sd:
            continue
        tgt = sd[k]
        tgt_shape = tuple(tgt.shape)
        # consider only if target has first dim > 1
        if len(tgt_shape) >= 1:
            tgt_len = tgt_shape[0]
            # find an unused array with same trailing dims
            for m in arr_meta:
                if m["idx"] in used:
                    continue
                arr = arrays[m["idx"]]
                # check if trailing dims match
                if arr.ndim >= 1 and arr.shape[1:] == tgt_shape[1:]:
                    # try resampling along axis 0
                    try:
                        res = resample_1d_along_axis(arr, tgt_len)
                        t = torch.tensor(res).view(tgt_shape)
                        new_sd[k] = t
                        used.add(m["idx"])
                        break
                    except Exception:
                        continue

    # Final: if new_sd empty or very small, raise
    if len(new_sd) == 0:
        raise RuntimeError("smart mapping produced no matches")

    # Fill any remaining keys with existing sd (so load_state_dict won't fail for missing keys)
    final_sd = sd.copy()
    final_sd.update(new_sd)
    return final_sd

def npz_to_pt_and_state_dict(npz_path: str, model):
    """
    Try to convert npz -> state_dict and save corresponding .pt file.
    Returns tuple (state_dict, pt_path) on success, or raises on failure.
    """
    arrays, files = load_npz_to_arrays(npz_path)
    # Strategy 1: ordered map
    try:
        sd = try_ordered_map(model, arrays)
        pt_path = Path(npz_path).with_suffix(".pt")
        torch.save(sd, pt_path)
        return sd, str(pt_path)
    except Exception as ex_order:
        # Try smart mapping
        try:
            sd = try_smart_map(model, arrays, files)
            pt_path = Path(npz_path).with_suffix(".pt")
            torch.save(sd, pt_path)
            return sd, str(pt_path)
        except Exception as ex_smart:
            # raise combined error for debugging
            raise RuntimeError(f"npz->pt mapping failed: ordered_error={ex_order}; smart_error={ex_smart}")

# -------------------------
# Evaluate helper
# -------------------------
def evaluate_single_ckpt(index_path, ckpt_path, model_name, device="cpu", n_features=None, seq_len=None, out_file=None):
    """
    Evaluate one checkpoint (pt or npz). If npz, attempt conversion->pt (and save .pt).
    Returns dict { 'ckpt': path, 'auroc': val, 'auprc': val, 'n_samples': n } or raises.
    """
    ckpt_path = str(ckpt_path)
    
    # --- START OF MODIFICATION ---
    # We must determine n_features and seq_len for the model
    # If not provided, infer them from the dataset *before* loading the model
    
    ds = PatientDataset(index_path, mode="transformer" if model_name=="transformer" else "grud")
    if n_features is None:
        try:
            x0, *_ = ds[0]
            arr = np.asarray(x0)
            if arr.ndim == 2:
                seq_len_inf, n_features_inf = arr.shape
            else:
                n_features_inf = arr.shape[-1]
                seq_len_inf = arr.shape[0] if arr.ndim >= 2 else 48
            
            if n_features is None:
                n_features = n_features_inf
            if seq_len is None:
                seq_len = seq_len_inf
            print(f"[EVAL] Inferred n_features={n_features}, seq_len={seq_len} from dataset.")
        except Exception:
            if n_features is None or seq_len is None:
                raise ValueError("Could not infer n_features/seq_len. Please provide --n_features and --seq_len arguments.")
    # --- END OF MODIFICATION ---
    
    model = build_model_for_eval(model_name, n_features=n_features, seq_len=seq_len)

    # load checkpoint - accept .pt or .npz
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        
        # Handle different .pt save formats
        if "model_state" in sd and isinstance(sd["model_state"], dict):
             sd = sd["model_state"] # From train_local.py
        elif "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"] # Common alternative
        # else assume sd *is* the state_dict (from fl_server.py)
        
        try:
            model.load_state_dict(sd)
            print(f"[EVAL] Loaded .pt checkpoint: {ckpt_path}")
        except Exception as e:
            raise RuntimeError(f"Failed loading .pt state_dict: {e}")

    elif ckpt_path.endswith(".npz"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"NPZ not found: {ckpt_path}")
        # attempt conversions (ordered -> smart)
        try:
            sd, pt_path = npz_to_pt_and_state_dict(ckpt_path, model)
            model.load_state_dict(sd)
            print(f"[EVAL] Converted NPZ -> PT: saved {pt_path}")
            ckpt_path = pt_path  # use converted .pt for reporting
        except Exception as e:
            raise RuntimeError(f"Failed to convert NPZ to PT: {e}")
    else:
        raise ValueError("Unsupported checkpoint extension. Provide .pt/.pth or .npz")

    # Run evaluation using the dataset
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model.eval()
    model.to(device)

    ys = []
    preds = []
    import numpy as _np
    from sklearn.metrics import roc_auc_score, average_precision_score

    with torch.no_grad():
        for batch in loader:
            if model_name == "transformer":
                Xb, yb = batch
                Xb = Xb.to(device).float()
                logits = model(Xb)
            else:
                Xb, Mb, Db, yb = batch
                Xb, Mb, Db = Xb.to(device).float(), Mb.to(device).float(), Db.to(device).float()
                logits = model(Xb, Mb, Db)

            if isinstance(logits, torch.Tensor):
                arr = logits.detach().cpu().numpy().reshape(-1)
            else:
                arr = _np.asarray(logits).reshape(-1)

            yb = _np.asarray(yb).reshape(-1)

            ys.append(yb)
            preds.append(arr)

    if len(ys) == 0:
        raise RuntimeError("No evaluation samples found in dataset.")

    ys = _np.concatenate(ys)
    preds = _np.concatenate(preds)

    # convert logits to probabilities
    try:
        from scipy.special import expit
        probs = expit(preds)
    except Exception:
        probs = 1.0 / (1.0 + _np.exp(-preds))

    auc = None
    ap = None
    if len(set(ys)) > 1:
        auc = float(roc_auc_score(ys, probs))
        ap = float(average_precision_score(ys, probs))
    else:
        print("[EVAL] Warning: single-class labels; cannot compute AUC/AP")

    print(f"[EVAL] {os.path.basename(ckpt_path)} -> samples={len(ys)} AUROC={auc} AUPRC={ap}")
    
    # --- START OF MODIFICATION ---
    # Create results dictionary and save to JSON if out_file is provided
    results_dict = {
        "ckpt": ckpt_path,
        "model_name": Path(ckpt_path).stem,
        "auroc": auc,
        "auprc": ap,
        "n": int(len(ys)),
        "y_true": ys.tolist(),  # Save full lists for plotting
        "y_prob": probs.tolist() # Save full lists for plotting
    }
    
    if out_file:
        try:
            with open(out_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"[EVAL] Saved detailed results to: {out_file}")
        except Exception as e:
            print(f"[EVAL][WARN] Failed to save results to {out_file}: {e}")
            
    return results_dict
    # --- END OF MODIFICATION ---

# -------------------------
# Main entrypoint
# -------------------------
def evaluate(index_path, ckpt_input, model_name, device="cpu", n_features=None, seq_len=None, out_file=None):
    """
    ckpt_input can be:
      - path to a single .pt or .npz
      - path to a directory containing many .npz/.pt files
    Returns best metrics info if directory mode used; otherwise returns single result.
    """
    ckpt_path = Path(ckpt_input)
    results = []

    if ckpt_path.is_dir():
        # collect all .npz/.pt/.pth in directory
        cands = sorted(list(ckpt_path.glob("*.npz")) + list(ckpt_path.glob("*.pt")) + list(ckpt_path.glob("*.pth")))
        if not cands:
            raise FileNotFoundError("No .npz/.pt/.pth files found in directory: " + str(ckpt_path))
        for c in cands:
            try:
                # Note: out_file is not passed in directory mode, just print results
                r = evaluate_single_ckpt(index_path, str(c), model_name, device=device, n_features=n_features, seq_len=seq_len, out_file=None)
                results.append(r)
            except Exception as e:
                print(f"[EVAL][WARN] Skipping {c}: {e}")
    else:
        # single file, pass the out_file argument
        results.append(evaluate_single_ckpt(index_path, str(ckpt_path), model_name, device=device, n_features=n_features, seq_len=seq_len, out_file=out_file))

    # If multiple results, pick best (AUROC, fallback AUPRC)
    if len(results) > 1:
        best = None
        for r in results:
            if r["auroc"] is not None:
                score = r["auroc"]
            elif r["auprc"] is not None:
                score = r["auprc"]
            else:
                score = None
            r["_score"] = score
            if score is not None:
                if best is None or score > best["_score"]:
                    best = r
        print("\n[EVAL] Summary of evaluated checkpoints:")
        for r in results:
            print(f"  {os.path.basename(r['ckpt']):40s}  AUROC={r['auroc']}  AUPRC={r['auprc']}  samples={r['n']}")
        if best is not None:
            print(f"\n[EVAL] Best checkpoint: {os.path.basename(best['ckpt'])}  score={best['_score']}")
        else:
            print("\n[EVAL] No checkpoint reported numeric AUROC/AUPRC; cannot select best.")
        return best
    else:
        return results[0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--ckpt", required=True, help="path to .pt/.pth or .npz checkpoint OR a directory containing them")
    ap.add_argument("--model", choices=["transformer", "grud"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_features", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    # --- START OF MODIFICATION ---
    ap.add_argument("--out_file", type=str, default=None, help="Optional: path to save detailed results JSON for plotting")
    # --- END OF MODIFICATION ---
    
    args = ap.parse_args()
    evaluate(args.index, args.ckpt, args.model, device=args.device, n_features=args.n_features, seq_len=args.seq_len, out_file=args.out_file)