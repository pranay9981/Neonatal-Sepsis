"""
Tests for scripts/create_windowed_dataset.py and parallel_preprocess onset_hour/y_seq fields.
"""
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

SRC_DIR = Path(__file__).parent.parent / "src"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from create_windowed_dataset import create_windowed_dataset
from dataset import PatientDataset


def _make_patient_pt(folder: Path, patient_id: str, T: int = 60, n_features: int = 40, onset: Optional[int] = None):
    """Create a full-length patient .pt with optional onset_hour."""
    X = torch.randn(T, n_features)
    mask = torch.ones(T, n_features)
    deltas = torch.zeros(T, n_features)
    y = 1 if onset is not None else 0
    y_seq = torch.zeros(T, dtype=torch.int8)
    if onset is not None:
        y_seq[onset:] = 1

    payload = {
        "X": X, "mask": mask, "deltas": deltas, "actual_len": T,
        "y": y, "y_seq": y_seq,
        "meta": {"patient_id": patient_id},
    }
    if onset is not None:
        payload["onset_hour"] = onset

    p = folder / f"{patient_id}.pt"
    torch.save(payload, p)
    return str(p)


def _make_index(folder: Path, paths: list, labels: list) -> str:
    idx = folder / "index_with_labels.pt"
    torch.save({"x_paths": paths, "y": labels}, idx)
    return str(idx)


class TestCreateWindowedDataset:
    def test_creates_index(self, tmp_path):
        p1 = _make_patient_pt(tmp_path, "p001", T=55, onset=50)
        idx = _make_index(tmp_path, [p1], [1])
        out = tmp_path / "windows"
        create_windowed_dataset(str(idx), str(out), seq_len=48, stride=1, horizon=6)
        assert (out / "index_with_labels.pt").exists()

    def test_window_shapes(self, tmp_path):
        p1 = _make_patient_pt(tmp_path, "p001", T=55)
        idx = _make_index(tmp_path, [p1], [0])
        out = tmp_path / "windows"
        create_windowed_dataset(str(idx), str(out), seq_len=48, stride=1, horizon=6)
        d = torch.load(out / "index_with_labels.pt", weights_only=True)
        for pt_path in d["x_paths"]:
            w = torch.load(pt_path, weights_only=True)
            assert w["X"].shape[0] == 48
            assert w["X"].shape[1] == 40

    def test_early_warning_label_before_onset(self, tmp_path):
        """Window ending just before onset+horizon should get label 1."""
        # T=60, onset=50 → windows ending between 44 and 50 (before onset, within horizon=6) → label=1
        p1 = _make_patient_pt(tmp_path, "p001", T=60, onset=50)
        idx = _make_index(tmp_path, [p1], [1])
        out = tmp_path / "windows"
        create_windowed_dataset(str(idx), str(out), seq_len=48, stride=1, horizon=6)
        d = torch.load(out / "index_with_labels.pt", weights_only=True)
        labels = d["y"]
        # At least one window should have label=1 (early warning)
        assert 1 in labels, "Expected at least one early-warning positive window"

    def test_non_sepsis_patient_all_zeros(self, tmp_path):
        p1 = _make_patient_pt(tmp_path, "p001", T=55, onset=None)
        idx = _make_index(tmp_path, [p1], [0])
        out = tmp_path / "windows"
        create_windowed_dataset(str(idx), str(out), seq_len=48, stride=1, horizon=6)
        d = torch.load(out / "index_with_labels.pt", weights_only=True)
        assert all(l == 0 for l in d["y"]), "Non-sepsis patient should produce all-zero windows"

    def test_stride_controls_window_count(self, tmp_path):
        p1 = _make_patient_pt(tmp_path, "p001", T=60)
        idx = _make_index(tmp_path, [p1], [0])

        out1 = tmp_path / "w_stride1"
        create_windowed_dataset(str(idx), str(out1), seq_len=48, stride=1, horizon=6)
        d1 = torch.load(out1 / "index_with_labels.pt", weights_only=True)

        out4 = tmp_path / "w_stride4"
        create_windowed_dataset(str(idx), str(out4), seq_len=48, stride=4, horizon=6)
        d4 = torch.load(out4 / "index_with_labels.pt", weights_only=True)

        assert len(d1["x_paths"]) > len(d4["x_paths"]), "Larger stride should produce fewer windows"

    def test_windowed_dataset_loadable(self, tmp_path):
        """Windows should be loadable via PatientDataset in both modes."""
        p1 = _make_patient_pt(tmp_path, "p001", T=55, onset=50)
        idx = _make_index(tmp_path, [p1], [1])
        out = tmp_path / "windows"
        win_idx = create_windowed_dataset(str(idx), str(out), seq_len=48, stride=4, horizon=6)

        ds_t = PatientDataset(win_idx, mode="transformer")
        X, pad_mask, y = ds_t[0]
        assert X.shape == (48, 40)

        ds_g = PatientDataset(win_idx, mode="grud")
        X, mask, deltas, actual_len, y = ds_g[0]
        assert X.shape == (48, 40)


class TestPreprocessYSeq:
    """Test that parallel_preprocess saves y_seq and onset_hour fields."""

    def test_process_file_saves_y_seq(self, tmp_path):
        import pandas as pd
        sys.path.insert(0, str(SRC_DIR))
        from parallel_preprocess import process_file

        # Build a synthetic CSV with SepsisLabel
        T = 60
        df = pd.DataFrame({
            "HR": np.random.uniform(60, 100, T),
            "O2Sat": np.random.uniform(95, 100, T),
            "ICULOS": np.arange(1, T + 1, dtype=float),
            "SepsisLabel": [0] * 50 + [1] * 10,
        })
        csv_path = tmp_path / "test_patient.csv"
        df.to_csv(csv_path, index=False)

        result = process_file(str(csv_path), str(tmp_path), seq_len=48)
        ok, out_path = result[1], result[2]
        assert ok, "process_file should succeed"

        d = torch.load(out_path, weights_only=True)
        assert "y" in d
        assert d["y"] == 1, "Patient should be labeled positive"
        assert "y_seq" in d, "y_seq key missing from saved window"
        assert d["y_seq"] is not None
