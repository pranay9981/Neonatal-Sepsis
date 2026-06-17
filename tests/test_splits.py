"""
Tests for scripts/create_splits.py and the scaler support in PatientDataset.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

SRC_DIR = Path(__file__).parent.parent / "src"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import PatientDataset
from create_splits import create_splits


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_index(folder, n=20, n_pos=5):
    paths, labels = [], []
    for i in range(n):
        y = 1 if i < n_pos else 0
        X = torch.randn(48, 40)
        mask = torch.ones(48, 40)
        deltas = torch.zeros(48, 40)
        p = folder / f"p{i:03d}.pt"
        torch.save({"X": X, "mask": mask, "deltas": deltas, "actual_len": 48, "y": y, "meta": {}}, p)
        paths.append(str(p))
        labels.append(y)
    idx_path = folder / "index_with_labels.pt"
    torch.save({"x_paths": paths, "y": labels}, idx_path)
    return str(idx_path), paths, labels


def _make_scaler(folder, n_features=40):
    mean = [float(i) for i in range(n_features)]
    std = [1.0] * n_features
    scaler = {"mean": mean, "std": std, "n_features": n_features}
    p = folder / "scaler.json"
    with open(p, "w") as f:
        json.dump(scaler, f)
    return str(p)


# ── create_splits tests ─────────────────────────────────────────────────────────

class TestCreateSplits:
    def test_creates_three_files(self, tmp_path):
        idx, _, _ = _make_index(tmp_path)
        out = tmp_path / "splits"
        create_splits(idx, str(out))
        assert (out / "train_index.pt").exists()
        assert (out / "val_index.pt").exists()
        assert (out / "test_index.pt").exists()

    def test_no_patient_overlap(self, tmp_path):
        idx, _, _ = _make_index(tmp_path, n=30, n_pos=6)
        out = tmp_path / "splits"
        create_splits(idx, str(out))
        sets = {}
        for name in ("train", "val", "test"):
            d = torch.load(out / f"{name}_index.pt", weights_only=False)
            sets[name] = set(d["x_paths"])
        assert sets["train"].isdisjoint(sets["test"]), "train/test overlap"
        assert sets["val"].isdisjoint(sets["test"]), "val/test overlap"
        assert sets["train"].isdisjoint(sets["val"]), "train/val overlap"

    def test_covers_all_patients(self, tmp_path):
        idx, paths, _ = _make_index(tmp_path, n=30, n_pos=6)
        out = tmp_path / "splits"
        create_splits(idx, str(out))
        all_split = set()
        for name in ("train", "val", "test"):
            d = torch.load(out / f"{name}_index.pt", weights_only=False)
            all_split |= set(d["x_paths"])
        assert all_split == set(paths), "Not all patients covered"

    def test_approximate_ratios(self, tmp_path):
        idx, _, _ = _make_index(tmp_path, n=100, n_pos=20)
        out = tmp_path / "splits"
        create_splits(idx, str(out), train_ratio=0.70, val_ratio=0.15)
        sizes = {}
        for name in ("train", "val", "test"):
            d = torch.load(out / f"{name}_index.pt", weights_only=False)
            sizes[name] = len(d["x_paths"])
        total = sum(sizes.values())
        assert abs(sizes["train"] / total - 0.70) < 0.05
        assert abs(sizes["val"] / total - 0.15) < 0.05
        assert abs(sizes["test"] / total - 0.15) < 0.05

    def test_both_classes_in_all_splits(self, tmp_path):
        idx, _, _ = _make_index(tmp_path, n=60, n_pos=15)
        out = tmp_path / "splits"
        create_splits(idx, str(out))
        for name in ("train", "val", "test"):
            d = torch.load(out / f"{name}_index.pt", weights_only=False)
            assert 1 in d["y"], f"No positives in {name} split"
            assert 0 in d["y"], f"No negatives in {name} split"


# ── scaler in PatientDataset ────────────────────────────────────────────────────

class TestDatasetScaler:
    def _make_patient_idx(self, folder, n_features=40, n=5):
        paths, labels = [], []
        for i in range(n):
            X = torch.ones(48, n_features) * 10.0  # known raw value
            mask = torch.ones(48, n_features)
            deltas = torch.zeros(48, n_features)
            p = folder / f"p{i:03d}.pt"
            torch.save({"X": X, "mask": mask, "deltas": deltas, "actual_len": 48, "y": i % 2, "meta": {}}, p)
            paths.append(str(p))
            labels.append(i % 2)
        idx_path = folder / "index.pt"
        torch.save({"x_paths": paths, "y": labels}, idx_path)
        return str(idx_path)

    def test_scaler_normalises_output(self, tmp_path):
        # mean=10, std=2 → (10-10)/2 = 0
        scaler = {"mean": [10.0] * 40, "std": [2.0] * 40}
        sp = tmp_path / "scaler.json"
        sp.write_text(json.dumps(scaler))
        idx = self._make_patient_idx(tmp_path)
        ds = PatientDataset(idx, mode="transformer", scaler_path=str(sp))
        X, _, _ = ds[0]
        assert torch.allclose(X, torch.zeros_like(X), atol=1e-5), "Expected zero after scaling"

    def test_no_scaler_returns_raw(self, tmp_path):
        idx = self._make_patient_idx(tmp_path)
        ds = PatientDataset(idx, mode="transformer")
        X, _, _ = ds[0]
        assert torch.allclose(X, torch.ones_like(X) * 10.0, atol=1e-5)

    def test_scaler_applied_grud_mode(self, tmp_path):
        scaler = {"mean": [10.0] * 40, "std": [2.0] * 40}
        sp = tmp_path / "scaler.json"
        sp.write_text(json.dumps(scaler))
        idx = self._make_patient_idx(tmp_path)
        ds = PatientDataset(idx, mode="grud", scaler_path=str(sp))
        X, _, _, _ = ds[0]
        assert torch.allclose(X, torch.zeros_like(X), atol=1e-5)

    def test_missing_scaler_path_is_ignored(self, tmp_path):
        idx = self._make_patient_idx(tmp_path)
        ds = PatientDataset(idx, mode="transformer", scaler_path="/nonexistent/scaler.json")
        X, _, _ = ds[0]
        assert X.shape == (48, 40)


# ── split_clients excludes test patients ───────────────────────────────────────

class TestSplitClientsExcludesTest:
    def test_test_patients_not_in_client_data(self, tmp_path):
        from split_clients import split_into_clients

        # Create processed folder
        processed = tmp_path / "processed"
        processed.mkdir()
        paths, labels = [], []
        for i in range(30):
            X = torch.randn(48, 40)
            mask = torch.ones(48, 40)
            deltas = torch.zeros(48, 40)
            p = processed / f"p{i:03d}.pt"
            torch.save({"X": X, "mask": mask, "deltas": deltas, "actual_len": 48, "y": i % 2}, p)
            paths.append(str(p))
            labels.append(i % 2)
        torch.save({"x_paths": paths, "y": labels}, processed / "index_with_labels.pt")

        # Create a fake test split with 5 patients
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        test_paths = paths[:5]
        torch.save({"x_paths": test_paths, "y": labels[:5]}, splits_dir / "test_index.pt")

        out_root = tmp_path / "clients"
        split_into_clients(str(processed), str(out_root), n_clients=3, splits_dir=str(splits_dir))

        test_set = set(test_paths)
        all_client_paths = set()
        for i in range(1, 4):
            d = torch.load(out_root / f"client{i}" / "index.pt", weights_only=False)
            # Check the basenames match (files are copied)
            for p in d["x_paths"]:
                all_client_paths.add(Path(p).name)

        test_basenames = {Path(p).name for p in test_set}
        overlap = all_client_paths & test_basenames
        assert not overlap, f"Test patients leaked into client data: {overlap}"
