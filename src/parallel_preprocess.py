# src/parallel_preprocess.py
import os
import glob
import json
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from logging_config import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_FEATURES = [
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2",
    "BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN",
    "Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
    "Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total",
    "TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets","Age","Gender","Unit1","Unit2",
    "HospAdmTime","ICULOS"
]
LABEL_CANDIDATES = ["SepsisLabel","sepsislabel","sepsis_label","sepsis"]


def detect_sep_from_file(path):
    import itertools
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = ''.join(itertools.islice(f, 5))
    if '|' in sample:
        return '|'
    if ',' in sample:
        return ','
    if '\t' in sample:
        return '\t'
    return None


def safe_read(path):
    sep = detect_sep_from_file(path)
    if sep is None:
        df = pd.read_csv(path, engine='python')
    else:
        df = pd.read_csv(path, sep=sep, engine='python')
    df.columns = [c.strip() for c in df.columns]
    return df


def make_datetime_index(df):
    for cand in ['timestamp', 'time', 'datetime', 'date', 'record_time']:
        if cand in df.columns:
            try:
                df[cand] = pd.to_datetime(df[cand], errors='coerce')
                if df[cand].notna().any():
                    return df.set_index(cand)
            except Exception:
                pass
    if 'ICULOS' in df.columns:
        try:
            horas = pd.to_numeric(df['ICULOS'], errors='coerce')
            df = df.assign(_iculos=horas)
            df = df.sort_values('_iculos')
            base = pd.Timestamp("1970-01-01")
            idx = base + pd.to_timedelta(df['_iculos'].fillna(0), unit='h')
            df.index = idx
            df.index.name = 'time_index'
            df = df.drop(columns=['_iculos'])
            return df
        except Exception:
            pass
    base = pd.Timestamp("1970-01-01")
    idx = pd.date_range(start=base, periods=len(df), freq='h')
    df.index = idx
    df.index.name = 'time_index'
    return df


def ensure_unique_index(df):
    if df.index.duplicated().any():
        # Keep the FIRST occurrence (earliest timestamp) to preserve the earliest
        # clinical events including sepsis onset rather than discarding them.
        df = df[~df.index.duplicated(keep='first')]
    return df


def _compute_deltas(mask_arr: np.ndarray) -> np.ndarray:
    """
    Per-feature time-since-last-observation.
    mask_arr: (T, F) float32, 1=observed 0=missing.
    Returns delta: (T, F) float32 in units of timesteps.
    """
    T, F = mask_arr.shape
    delta = np.zeros((T, F), dtype=np.float32)
    for t in range(1, T):
        delta[t] = (delta[t - 1] + 1.0) * (1.0 - mask_arr[t])
    return delta


def process_file(fp, out_folder, seq_len=48, freq='h'):
    try:
        df = safe_read(fp)
    except Exception as e:
        return (fp, False, f"read_error: {e}")
    df.columns = [c.strip() for c in df.columns]
    if df.shape[0] == 0:
        return (fp, False, "empty_file")
    df = make_datetime_index(df)
    df = ensure_unique_index(df)
    present_features = [c for c in DEFAULT_FEATURES if c in df.columns]
    if not present_features:
        lmap = {c.lower(): c for c in df.columns}
        present_features = [lmap[f.lower()] for f in DEFAULT_FEATURES if f.lower() in lmap]
    if not present_features:
        return (fp, False, "no_features")

    X_df_raw = df[present_features].apply(pd.to_numeric, errors='coerce')

    # Resample to hourly buckets — aggregate by mean, NaN where no data in that hour.
    try:
        X_df_resampled = X_df_raw.resample(freq).mean()
    except Exception:
        start, end = X_df_raw.index.min(), X_df_raw.index.max()
        new_idx = pd.date_range(start=start, end=end, freq=freq)
        X_df_resampled = X_df_raw.reindex(new_idx)

    # Compute mask and deltas BEFORE imputation so missingness signal is preserved.
    mask_np = X_df_resampled.notna().values.astype(np.float32)   # (T_res, F)
    delta_np = _compute_deltas(mask_np)                           # (T_res, F)

    # Impute for the input tensor (used by both Transformer and GRU-D as the "filled" X).
    # NOTE: per-patient column mean is used here because population mean (from scaler.json)
    # is not yet available at preprocessing time. This is intentional — scaler.json is
    # produced by recompute_scaler_from_index() after create_splits.py.
    logger.debug(
        "Patient %s: imputing missing values with per-patient mean; "
        "population mean from scaler.json is applied at training time via dataset.py.",
        os.path.splitext(os.path.basename(fp))[0],
    )
    col_mean = X_df_resampled.mean()
    X_df = X_df_resampled.ffill().bfill().fillna(col_mean).fillna(0.0)

    # Window to seq_len. Front-pad with zeros so padded positions are identifiable.
    n_rows = len(X_df)
    F = len(present_features)
    if n_rows >= seq_len:
        X_seq    = X_df.iloc[-seq_len:].to_numpy(dtype=np.float32)
        mask_seq = mask_np[-seq_len:]
        delta_seq = delta_np[-seq_len:]
        actual_len = seq_len
    else:
        pad_len = seq_len - n_rows
        actual_len = n_rows
        X_seq    = np.vstack([np.zeros((pad_len, F), dtype=np.float32),
                               X_df.to_numpy(dtype=np.float32)])
        mask_seq  = np.vstack([np.zeros((pad_len, F), dtype=np.float32), mask_np])
        delta_seq = np.vstack([np.zeros((pad_len, F), dtype=np.float32), delta_np])

    y = 0
    y_seq_raw = None  # per-timestep SepsisLabel (before resampling window)
    y_seq_full = None  # full resampled label sequence (set inside label loop below)
    onset_hour = None
    for cand in LABEL_CANDIDATES:
        for col in df.columns:
            if col.lower() == cand.lower():
                try:
                    y = int(df[col].max())
                except Exception:
                    try:
                        y = int(float(df[col].max()))
                    except Exception:
                        y = 0
                # Extract per-timestep labels for early-warning windowing.
                try:
                    raw_labels = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    df_tmp = df.assign(_lbl=raw_labels)
                    lbl_resampled = X_df_resampled.copy()
                    lbl_resampled['_lbl'] = df_tmp.resample(freq)['_lbl'].max().reindex(lbl_resampled.index).fillna(0).astype(int)
                    y_seq_full = lbl_resampled['_lbl'].values.astype(np.int8)
                    # Find first onset hour index in the resampled sequence
                    onset_indices = np.where(y_seq_full == 1)[0]
                    if len(onset_indices) > 0:
                        onset_hour = int(onset_indices[0])
                    # Window or front-pad y_seq the same way as X
                    n_rows_full = len(y_seq_full)
                    if n_rows_full >= seq_len:
                        y_seq_raw = y_seq_full[-seq_len:]
                    else:
                        pad_len_y = seq_len - n_rows_full
                        y_seq_raw = np.concatenate([np.zeros(pad_len_y, dtype=np.int8), y_seq_full])
                except Exception as _e:
                    patient_id_tmp = os.path.splitext(os.path.basename(fp))[0]
                    logger.warning(
                        "Patient %s: y_seq extraction failed (%s); patient-level label y=%d unchanged",
                        patient_id_tmp, _e, y,
                    )
                    y_seq_raw = None
                break
        if y:
            break

    patient_id = os.path.splitext(os.path.basename(fp))[0]
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{patient_id}.pt")
    payload = {
        'X': torch.tensor(X_seq),
        'mask': torch.tensor(mask_seq),
        'deltas': torch.tensor(delta_seq),
        'actual_len': actual_len,
        'y': int(y),
        'meta': {'patient_id': patient_id, 'features': present_features},
    }
    if y_seq_raw is not None:
        payload['y_seq'] = torch.tensor(y_seq_raw)
    if onset_hour is not None:
        # Adjust for windowing: if full sequence was cropped to seq_len, onset_hour shifts
        n_rows_full_final = len(y_seq_full) if y_seq_full is not None else seq_len
        if n_rows_full_final > seq_len:
            onset_hour = max(0, onset_hour - (n_rows_full_final - seq_len))
        payload['onset_hour'] = onset_hour
    torch.save(payload, out_path)
    return (fp, True, out_path, int(y))


def recompute_scaler_from_index(train_index_path: str, out_folder: str):
    """Recompute scaler.json using ONLY training patients to avoid test-set leakage.
    Call this after create_splits.py with the train_index.pt path."""
    d = torch.load(train_index_path, weights_only=False)
    x_paths = d.get("x_paths", [])
    if not x_paths:
        logger.warning("No paths in %s; skipping scaler recomputation.", train_index_path)
        return
    logger.info("Recomputing scaler from %d train-only patients (no test leakage)...", len(x_paths))
    _compute_and_save_scaler(x_paths, out_folder)


def _compute_and_save_scaler(x_paths, out_folder):
    """Compute per-feature mean/std across the provided patients and save as scaler.json.
    NOTE: call recompute_scaler_from_index() after create_splits.py to fit on train-only data."""
    logger.info("Computing feature scaler from %d patients...", len(x_paths))
    all_arrays = []
    for xp in x_paths:
        try:
            d = torch.load(xp, weights_only=False)
            arr = d['X'].numpy()
            actual_len = int(d.get('actual_len', arr.shape[0]))
            # Only use real (non-padded) rows to avoid bias from padding zeros
            all_arrays.append(arr[-actual_len:] if actual_len < arr.shape[0] else arr)
        except Exception:
            pass
    if not all_arrays:
        logger.warning("No patient tensors could be loaded; skipping scaler computation.")
        return
    flat = np.concatenate(all_arrays, axis=0)        # (sum_actual_lens, F)
    feature_mean = flat.mean(axis=0).tolist()
    feature_std = np.maximum(flat.std(axis=0), 1e-8).tolist()
    scaler = {
        "mean": feature_mean,
        "std": feature_std,
        "n_features": len(feature_mean),
        "n_patients": len(all_arrays),
    }
    scaler_path = os.path.join(out_folder, "scaler.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler, f, indent=2)
    logger.info("Saved feature scaler to %s", scaler_path)


def main(raw_folder, out_folder, seq_len=48, nprocs=None):
    files = sorted(
        glob.glob(os.path.join(raw_folder, "*.psv")) +
        glob.glob(os.path.join(raw_folder, "*.csv"))
    )
    if not files:
        logger.warning("No files found in %s", raw_folder)
        return
    nprocs = nprocs or max(1, cpu_count() - 1)
    logger.info("Processing %d files with %d processes...", len(files), nprocs)
    worker = partial(process_file, out_folder=out_folder, seq_len=seq_len)
    results = []
    with Pool(processes=nprocs) as p:
        for r in tqdm(p.imap_unordered(worker, files), total=len(files)):
            try:
                results.append(r)
            except Exception as e:
                logger.error("Worker error collecting result: %s", e)
    x_paths = []
    ys = []
    failures = []
    for result in results:
        fp, ok, info = result[0], result[1], result[2]
        if ok:
            x_paths.append(info)
            ys.append(float(result[3]) if len(result) > 3 else 0.0)
        else:
            failures.append((fp, info))
    idx_path = os.path.join(out_folder, "index_with_labels.pt")
    torch.save({'x_paths': x_paths, 'y': ys}, idx_path)
    logger.info("Wrote %d patient .pt files to %s", len(x_paths), out_folder)
    if failures:
        logger.warning("%d files failed to process:", len(failures))
        for fp, reason in failures[:10]:
            logger.warning("  - %s: %s", fp, reason)
    logger.info("Index written to %s", idx_path)
    logger.warning(
        "Scaler NOT computed here to prevent test-set leakage. "
        "Call recompute_scaler_from_index(train_index_path, out_folder) after create_splits.py."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_folder", default=str(_PROJECT_ROOT / "data" / "raw"))
    ap.add_argument("--out_folder", default=str(_PROJECT_ROOT / "data" / "processed" / "patients"))
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--nprocs", type=int, default=None)
    args = ap.parse_args()
    main(args.raw_folder, args.out_folder, seq_len=args.seq_len, nprocs=args.nprocs)
