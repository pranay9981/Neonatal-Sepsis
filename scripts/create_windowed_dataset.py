"""
Create a sliding-window early-warning dataset from processed patient .pt files.

For each patient the script creates one window per ICU hour with a prospective
(forward-looking) label: does sepsis onset occur within the next `horizon` hours?

This reframes the task from "did this patient ever get sepsis?" (retrospective,
one label per patient) to "at hour T, will sepsis onset within the next H hours?"
(prospective early-warning — the clinically meaningful question).

Usage:
  python scripts/create_windowed_dataset.py \\
    --index data/processed/patients/index_with_labels.pt \\
    --out_dir data/processed/windows \\
    --seq_len 48 --stride 1 --horizon 6
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).parent.parent


def _compute_deltas(mask_arr: np.ndarray) -> np.ndarray:
    T, F = mask_arr.shape
    delta = np.zeros((T, F), dtype=np.float32)
    for t in range(1, T):
        delta[t] = (delta[t - 1] + 1.0) * (1.0 - mask_arr[t])
    return delta


def create_windowed_dataset(
    index_path: str,
    out_dir: str,
    seq_len: int = 48,
    stride: int = 1,
    horizon: int = 6,
) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = torch.load(index_path, weights_only=False)
    x_paths = d["x_paths"]
    print(f"Processing {len(x_paths)} patients -> seq_len={seq_len}, stride={stride}, horizon={horizon}h")

    all_window_paths = []
    all_labels = []
    skipped = 0

    for pt_path in tqdm(x_paths):
        try:
            data = torch.load(pt_path, weights_only=False)
        except Exception:
            skipped += 1
            continue

        X = data["X"].numpy().astype(np.float32)  # (T, F) — already padded/truncated
        actual_len = int(data.get("actual_len", X.shape[0]))

        # Use pre-computed per-timestep labels if available; fall back to patient-level.
        if "y_seq" in data:
            y_seq = data["y_seq"].numpy().astype(np.int8)
        else:
            y_pat = int(data.get("y", 0))
            y_seq = np.full(X.shape[0], y_pat, dtype=np.int8)

        onset_hour = data.get("onset_hour", None)

        # Use pre-computed mask/deltas if available.
        # Note: delta values are patient-relative (hours since last observation for that feature
        # within this patient's full ICU stay), NOT window-relative.
        if "mask" in data and "deltas" in data:
            mask_full = data["mask"].numpy().astype(np.float32)
            delta_full = data["deltas"].numpy().astype(np.float32)
        else:
            mask_full = (~np.isnan(X)).astype(np.float32)
            delta_full = _compute_deltas(mask_full)

        T, F = X.shape
        patient_id = str(data.get("meta", {}).get("patient_id", Path(pt_path).stem))

        # Slide window over the actual (non-padded) portion of the sequence.
        # Padded prefix is seq_len - actual_len positions long.
        pad_offset = T - actual_len  # index where real data begins

        window_count = 0
        t_start = pad_offset  # first window must end at exactly seq_len (full window)
        if actual_len < seq_len:
            # Patient shorter than window — emit one window (the whole thing)
            t_starts = [0]
        else:
            # Slide window over real data
            t_starts = list(range(0, actual_len - seq_len + 1, stride))
            # Map back to absolute index
            t_starts = [pad_offset + i for i in t_starts]

        for t_end_excl in [ts + seq_len for ts in t_starts]:
            if t_end_excl > T:
                break

            X_win = X[t_end_excl - seq_len: t_end_excl]
            mask_win = mask_full[t_end_excl - seq_len: t_end_excl]
            delta_win = delta_full[t_end_excl - seq_len: t_end_excl]

            # Prospective label: 1 if onset is within the next `horizon` hours.
            window_end_abs = t_end_excl - pad_offset  # position in real-time axis
            if onset_hour is not None:
                label = int(onset_hour > window_end_abs and onset_hour <= window_end_abs + horizon)
            else:
                # Fallback: use per-timestep labels for next horizon hours
                next_horizon = y_seq[t_end_excl: t_end_excl + horizon]
                label = int(next_horizon.max() == 1) if len(next_horizon) > 0 else 0

            win_id = f"{patient_id}_w{t_end_excl - seq_len:04d}"
            win_path = out_dir / f"{win_id}.pt"
            torch.save(
                {
                    "X": torch.tensor(X_win),
                    "mask": torch.tensor(mask_win),
                    "deltas": torch.tensor(delta_win),
                    "actual_len": seq_len,
                    "y": label,
                    "meta": {"patient_id": patient_id, "window_end": int(t_end_excl)},
                },
                win_path,
            )
            all_window_paths.append(str(win_path))
            all_labels.append(label)
            window_count += 1

    index_out = out_dir / "index_with_labels.pt"
    torch.save({"x_paths": all_window_paths, "y": all_labels}, index_out)

    pos = sum(all_labels)
    print(f"\nCreated {len(all_window_paths)} windows from {len(x_paths) - skipped} patients")
    print(f"  Positive (early warning): {pos} ({100 * pos / max(1, len(all_labels)):.1f}%)")
    print(f"  Skipped patients: {skipped}")
    print(f"  Index saved to: {index_out}")
    return str(index_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create sliding-window early-warning dataset.")
    ap.add_argument(
        "--index",
        default=str(_PROJECT_ROOT / "data" / "processed" / "patients" / "index_with_labels.pt"),
    )
    ap.add_argument("--out_dir", default=str(_PROJECT_ROOT / "data" / "processed" / "windows"))
    ap.add_argument("--seq_len", type=int, default=48, help="Window length in hours")
    ap.add_argument("--stride", type=int, default=1, help="Stride between windows (1=every hour)")
    ap.add_argument("--horizon", type=int, default=6, help="Early-warning horizon in hours")
    args = ap.parse_args()
    create_windowed_dataset(args.index, args.out_dir, args.seq_len, args.stride, args.horizon)
