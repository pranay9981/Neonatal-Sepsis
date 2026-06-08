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
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = ''.join([next(f) for _ in range(5)])
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
        df = df[~df.index.duplicated(keep='last')]
    return df


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
    X_df = df[present_features].apply(pd.to_numeric, errors='coerce')
    X_df = X_df.ffill().bfill()
    X_df = X_df.fillna(X_df.mean()).fillna(0.0)
    try:
        X_df = X_df.resample(freq).ffill().bfill()
    except Exception:
        start, end = X_df.index.min(), X_df.index.max()
        new_idx = pd.date_range(start=start, end=end, freq=freq)
        X_df = X_df.reindex(new_idx).ffill().bfill().fillna(0.0)
    if len(X_df) >= seq_len:
        X_seq = X_df.iloc[-seq_len:].to_numpy(dtype=np.float32)
    else:
        pad_len = seq_len - len(X_df)
        if len(X_df) > 0:
            pad_arr = np.vstack([X_df.iloc[0].to_numpy(dtype=np.float32)] * pad_len)
            X_seq = np.vstack([pad_arr, X_df.to_numpy(dtype=np.float32)])
        else:
            return (fp, False, "no_numeric_rows")
    y = 0
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
                break
        if y:
            break
    patient_id = os.path.splitext(os.path.basename(fp))[0]
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{patient_id}.pt")
    torch.save(
        {'X': torch.tensor(X_seq), 'y': int(y), 'meta': {'patient_id': patient_id, 'features': present_features}},
        out_path,
    )
    return (fp, True, out_path)


def _compute_and_save_scaler(x_paths, out_folder):
    """Compute per-feature mean/std across all processed patients and save as scaler.json."""
    logger.info("Computing feature scaler from %d processed patients...", len(x_paths))
    all_arrays = []
    for xp in x_paths:
        try:
            d = torch.load(xp, weights_only=False)
            all_arrays.append(d['X'].numpy())
        except Exception:
            pass
    if not all_arrays:
        logger.warning("No patient tensors could be loaded; skipping scaler computation.")
        return
    stacked = np.stack(all_arrays, axis=0)          # (N, T, F)
    flat = stacked.reshape(-1, stacked.shape[-1])    # (N*T, F)
    feature_mean = flat.mean(axis=0).tolist()
    feature_std = np.maximum(flat.std(axis=0), 1e-6).tolist()
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
            results.append(r)
    x_paths = []
    ys = []
    failures = []
    for (fp, ok, info) in results:
        if ok:
            x_paths.append(info)
            d = torch.load(info, weights_only=False)
            ys.append(float(d.get('y', 0)))
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
    _compute_and_save_scaler(x_paths, out_folder)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_folder", default=str(_PROJECT_ROOT / "data" / "raw"))
    ap.add_argument("--out_folder", default=str(_PROJECT_ROOT / "data" / "processed" / "patients"))
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--nprocs", type=int, default=None)
    args = ap.parse_args()
    main(args.raw_folder, args.out_folder, seq_len=args.seq_len, nprocs=args.nprocs)
