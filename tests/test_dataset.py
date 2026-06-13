"""
Tests for PatientDataset (src/dataset.py).
Covers both transformer and grud modes, index loading, and edge cases.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import PatientDataset


def _make_patient(folder: Path, name: str, y: int = 0, seq_len: int = 48, n_features: int = 40) -> str:
    X = torch.randn(seq_len, n_features)
    p = folder / name
    torch.save({"X": X, "y": y, "meta": {}}, p)
    return str(p)


def _make_index(folder: Path, paths: list, labels: list) -> str:
    idx_path = folder / "index.pt"
    torch.save({"x_paths": paths, "y": labels}, idx_path)
    return str(idx_path)


class TestPatientDatasetTransformer:
    def test_len(self, tmp_path):
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt", y=i % 2) for i in range(6)]
        idx = _make_index(tmp_path, paths, [i % 2 for i in range(6)])
        ds = PatientDataset(idx, mode="transformer")
        assert len(ds) == 6

    def test_output_shape(self, tmp_path):
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt", y=0) for i in range(3)]
        idx = _make_index(tmp_path, paths, [0] * 3)
        ds = PatientDataset(idx, mode="transformer")
        X, y = ds[0]
        assert X.shape == (48, 40), f"Expected (48, 40), got {X.shape}"
        assert y.shape == torch.Size([])

    def test_labels(self, tmp_path):
        labels = [0, 1, 0, 1, 1]
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt", y=labels[i]) for i in range(5)]
        idx = _make_index(tmp_path, paths, labels)
        ds = PatientDataset(idx, mode="transformer")
        for i, (_, y) in enumerate(ds):
            assert int(y.item()) == labels[i]

    def test_all_finite(self, tmp_path):
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt") for i in range(4)]
        idx = _make_index(tmp_path, paths, [0] * 4)
        ds = PatientDataset(idx, mode="transformer")
        for i in range(len(ds)):
            X, _ = ds[i]
            assert torch.isfinite(X).all(), f"Non-finite values in sample {i}"

    def test_index_with_labels_attribute(self, tmp_path):
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt", y=i % 2) for i in range(4)]
        idx = _make_index(tmp_path, paths, [i % 2 for i in range(4)])
        ds = PatientDataset(idx, mode="transformer")
        assert ds.y_indexed is not None
        assert len(ds.y_indexed) == 4


class TestPatientDatasetGRUD:
    def test_output_shape(self, tmp_path):
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt") for i in range(3)]
        idx = _make_index(tmp_path, paths, [0] * 3)
        ds = PatientDataset(idx, mode="grud")
        X, mask, delta, y = ds[0]
        assert X.shape == (48, 40)
        assert mask.shape == (48, 40)
        assert delta.shape == (48, 40)
        assert y.shape == torch.Size([])

    def test_mask_binary(self, tmp_path):
        """Mask values should be 0 or 1."""
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt") for i in range(3)]
        idx = _make_index(tmp_path, paths, [0] * 3)
        ds = PatientDataset(idx, mode="grud")
        X, mask, delta, _ = ds[0]
        unique_mask = torch.unique(mask)
        for v in unique_mask:
            assert v.item() in (0.0, 1.0), f"Unexpected mask value: {v.item()}"

    def test_delta_non_negative(self, tmp_path):
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt") for i in range(3)]
        idx = _make_index(tmp_path, paths, [0] * 3)
        ds = PatientDataset(idx, mode="grud")
        _, _, delta, _ = ds[0]
        assert (delta >= 0).all()

    def test_grud_with_nan_features(self, tmp_path):
        """PatientDataset should handle NaN in X without error."""
        X = torch.randn(48, 40)
        X[5, 3] = float("nan")
        X[10, 15] = float("nan")
        p = tmp_path / "p_nan.pt"
        torch.save({"X": X, "y": 0, "meta": {}}, p)
        idx = _make_index(tmp_path, [str(p)], [0])
        ds = PatientDataset(idx, mode="grud")
        X_out, mask, delta, y = ds[0]
        assert mask[5, 3].item() == 0.0
        assert mask[10, 15].item() == 0.0


class TestPatientDatasetEdgeCases:
    def test_single_sample(self, tmp_path):
        paths = [_make_patient(tmp_path, "p000.pt", y=1)]
        idx = _make_index(tmp_path, paths, [1])
        ds = PatientDataset(idx, mode="transformer")
        assert len(ds) == 1
        X, y = ds[0]
        assert y.item() == 1.0

    def test_custom_seq_len(self, tmp_path):
        """Dataset should handle tensors with seq_len != 48."""
        X = torch.randn(24, 40)
        p = tmp_path / "short.pt"
        torch.save({"X": X, "y": 0, "meta": {}}, p)
        idx = _make_index(tmp_path, [str(p)], [0])
        ds = PatientDataset(idx, mode="transformer")
        X_out, _ = ds[0]
        assert X_out.shape[1] == 40
