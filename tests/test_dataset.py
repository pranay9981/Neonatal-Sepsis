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


def _make_patient(
    folder: Path,
    name: str,
    y: int = 0,
    seq_len: int = 48,
    n_features: int = 40,
    include_new_fields: bool = True,
) -> str:
    """
    Create a synthetic patient .pt file.
    include_new_fields=True adds mask/deltas/actual_len (new preprocessing format).
    include_new_fields=False mimics older .pt files (backward-compat fallback path).
    """
    X = torch.randn(seq_len, n_features)
    d = {"X": X, "y": y, "meta": {}}
    if include_new_fields:
        mask = torch.ones(seq_len, n_features)
        deltas = torch.zeros(seq_len, n_features)
        d["mask"] = mask
        d["deltas"] = deltas
        d["actual_len"] = seq_len
    p = folder / name
    torch.save(d, p)
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
        X, pad_mask, y = ds[0]
        assert X.shape == (48, 40), f"Expected (48, 40), got {X.shape}"
        assert pad_mask.shape == (48,), f"Expected pad_mask (48,), got {pad_mask.shape}"
        assert pad_mask.dtype == torch.bool
        assert y.shape == torch.Size([])

    def test_labels(self, tmp_path):
        labels = [0, 1, 0, 1, 1]
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt", y=labels[i]) for i in range(5)]
        idx = _make_index(tmp_path, paths, labels)
        ds = PatientDataset(idx, mode="transformer")
        for i, (_, __, y) in enumerate(ds):
            assert int(y.item()) == labels[i]

    def test_all_finite(self, tmp_path):
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt") for i in range(4)]
        idx = _make_index(tmp_path, paths, [0] * 4)
        ds = PatientDataset(idx, mode="transformer")
        for i in range(len(ds)):
            X, _, _ = ds[i]
            assert torch.isfinite(X).all(), f"Non-finite values in sample {i}"

    def test_index_with_labels_attribute(self, tmp_path):
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt", y=i % 2) for i in range(4)]
        idx = _make_index(tmp_path, paths, [i % 2 for i in range(4)])
        ds = PatientDataset(idx, mode="transformer")
        assert ds.y_indexed is not None
        assert len(ds.y_indexed) == 4

    def test_pad_mask_all_false_when_no_padding(self, tmp_path):
        """When actual_len == seq_len, pad_mask should be all False."""
        paths = [_make_patient(tmp_path, f"p{i:03d}.pt", seq_len=48) for i in range(2)]
        idx = _make_index(tmp_path, paths, [0] * 2)
        ds = PatientDataset(idx, mode="transformer")
        _, pad_mask, _ = ds[0]
        assert not pad_mask.any(), "Expected no padding for full-length sequence"

    def test_pad_mask_front_padded(self, tmp_path):
        """When actual_len < seq_len, first (seq_len - actual_len) positions should be True."""
        X = torch.randn(24, 40)
        # Simulate: padded to 48 by prepending 24 zero rows
        X_padded = torch.cat([torch.zeros(24, 40), X], dim=0)
        p = tmp_path / "padded.pt"
        torch.save({"X": X_padded, "y": 0, "meta": {}, "actual_len": 24}, p)
        idx = _make_index(tmp_path, [str(p)], [0])
        ds = PatientDataset(idx, mode="transformer")
        _, pad_mask, _ = ds[0]
        assert pad_mask[:24].all(), "First 24 positions should be masked"
        assert not pad_mask[24:].any(), "Last 24 positions should be unmasked"


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
        """Fallback path: NaN in X triggers isnan-based mask (no pre-computed mask in .pt)."""
        X = torch.randn(48, 40)
        X[5, 3] = float("nan")
        X[10, 15] = float("nan")
        p = tmp_path / "p_nan.pt"
        # Intentionally no 'mask' key to exercise fallback path.
        torch.save({"X": X, "y": 0, "meta": {}}, p)
        idx = _make_index(tmp_path, [str(p)], [0])
        ds = PatientDataset(idx, mode="grud")
        X_out, mask, delta, y = ds[0]
        assert mask[5, 3].item() == 0.0
        assert mask[10, 15].item() == 0.0

    def test_grud_uses_precomputed_mask(self, tmp_path):
        """When .pt has pre-computed mask, it should be used directly."""
        X = torch.randn(48, 40)
        mask = torch.ones(48, 40)
        mask[3, 7] = 0.0  # mark one position as missing
        deltas = torch.zeros(48, 40)
        deltas[4:, 7] = torch.arange(1, 45, dtype=torch.float)  # delta builds up after missing
        p = tmp_path / "precomputed.pt"
        torch.save({"X": X, "mask": mask, "deltas": deltas, "actual_len": 48, "y": 1, "meta": {}}, p)
        idx = _make_index(tmp_path, [str(p)], [1])
        ds = PatientDataset(idx, mode="grud")
        _, m_out, d_out, _ = ds[0]
        assert m_out[3, 7].item() == 0.0, "Should use pre-computed mask"
        assert d_out[4, 7].item() == 1.0, "Should use pre-computed deltas"


class TestPatientDatasetEdgeCases:
    def test_single_sample(self, tmp_path):
        paths = [_make_patient(tmp_path, "p000.pt", y=1)]
        idx = _make_index(tmp_path, paths, [1])
        ds = PatientDataset(idx, mode="transformer")
        assert len(ds) == 1
        X, pad_mask, y = ds[0]
        assert y.item() == 1.0

    def test_custom_seq_len(self, tmp_path):
        """Dataset should handle tensors with seq_len != 48."""
        X = torch.randn(24, 40)
        p = tmp_path / "short.pt"
        torch.save({"X": X, "y": 0, "meta": {}}, p)
        idx = _make_index(tmp_path, [str(p)], [0])
        ds = PatientDataset(idx, mode="transformer")
        X_out, _, _ = ds[0]
        assert X_out.shape[1] == 40
