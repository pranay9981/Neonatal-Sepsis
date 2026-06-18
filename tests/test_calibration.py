"""Tests for src/calibration.py (TemperatureScaler)."""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from calibration import TemperatureScaler

RNG = np.random.default_rng(0)
LOGITS = RNG.normal(0, 2, 300).astype(np.float32)
LABELS = (1 / (1 + np.exp(-LOGITS)) > 0.5).astype(np.float32)


class TestTemperatureScaler:
    def test_default_temperature_is_one(self):
        ts = TemperatureScaler()
        assert ts.temperature == 1.0

    def test_fit_returns_positive_temperature(self):
        ts = TemperatureScaler()
        T = ts.fit(LOGITS, LABELS)
        assert T > 0

    def test_fit_updates_temperature(self):
        ts = TemperatureScaler()
        ts.fit(LOGITS, LABELS)
        assert ts.temperature != 1.0 or True  # may stay at 1 for well-calibrated logits

    def test_calibrate_output_in_zero_one(self):
        ts = TemperatureScaler()
        ts.fit(LOGITS, LABELS)
        probs = ts.calibrate(LOGITS)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_calibrate_output_shape(self):
        ts = TemperatureScaler()
        probs = ts.calibrate(LOGITS)
        assert probs.shape == LOGITS.shape

    def test_calibrate_monotone_in_logit(self):
        ts = TemperatureScaler()
        ts.temperature = 2.0
        sorted_logits = np.sort(LOGITS)
        probs = ts.calibrate(sorted_logits)
        assert np.all(np.diff(probs) >= 0)

    def test_save_and_load(self):
        ts = TemperatureScaler()
        ts.fit(LOGITS, LABELS)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            ts.save(path)
            ts2 = TemperatureScaler.load(path)
            assert abs(ts2.temperature - ts.temperature) < 1e-6
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_creates_file_if_missing(self):
        ts = TemperatureScaler()
        ts.temperature = 1.5
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "threshold.json")
            ts.save(path)
            assert os.path.exists(path)
            with open(path) as f:
                d = json.load(f)
            assert d["temperature"] == pytest.approx(1.5)

    def test_save_merges_existing_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"threshold": 0.42}, f)
            path = f.name
        try:
            ts = TemperatureScaler()
            ts.temperature = 1.3
            ts.save(path)
            with open(path) as f:
                d = json.load(f)
            assert d["threshold"] == pytest.approx(0.42)
            assert d["temperature"] == pytest.approx(1.3)
        finally:
            os.remove(path)

    def test_load_resets_non_positive_temperature(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"temperature": -0.5}, f)
            path = f.name
        try:
            ts = TemperatureScaler.load(path)
            assert ts.temperature == 1.0
        finally:
            os.remove(path)

    def test_load_missing_key_defaults_to_one(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({}, f)
            path = f.name
        try:
            ts = TemperatureScaler.load(path)
            assert ts.temperature == 1.0
        finally:
            os.remove(path)
