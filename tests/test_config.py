"""Tests for src/config.py (environment-driven constants)."""
import os
import sys
import importlib

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestConfigDefaults:
    def test_n_features_default(self):
        import config as cfg
        assert cfg.N_FEATURES == int(os.environ.get("SEPSIS_N_FEATURES", "40"))

    def test_seq_len_default(self):
        import config as cfg
        assert cfg.SEQ_LEN == int(os.environ.get("SEPSIS_SEQ_LEN", "48"))

    def test_project_root_exists(self):
        import config as cfg
        assert cfg.PROJECT_ROOT.exists()

    def test_model_path_is_string(self):
        import config as cfg
        assert isinstance(cfg.MODEL_PATH, str)

    def test_scaler_path_is_string(self):
        import config as cfg
        assert isinstance(cfg.SCALER_PATH, str)

    def test_n_features_positive(self):
        import config as cfg
        assert cfg.N_FEATURES > 0

    def test_seq_len_positive(self):
        import config as cfg
        assert cfg.SEQ_LEN > 0

    def test_env_override_model_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SEPSIS_MODEL_PATH", str(tmp_path / "my_model.pt"))
        if "config" in sys.modules:
            del sys.modules["config"]
        import config as cfg
        assert str(tmp_path / "my_model.pt") in cfg.MODEL_PATH
        # cleanup: reload with original env
        del sys.modules["config"]

    def test_validation_raises_for_zero_n_features(self, monkeypatch):
        monkeypatch.setenv("SEPSIS_N_FEATURES", "0")
        if "config" in sys.modules:
            del sys.modules["config"]
        with pytest.raises(ValueError, match="N_FEATURES"):
            import config  # noqa: F401
        # cleanup
        if "config" in sys.modules:
            del sys.modules["config"]

    def test_validation_raises_for_zero_seq_len(self, monkeypatch):
        monkeypatch.setenv("SEPSIS_N_FEATURES", "40")
        monkeypatch.setenv("SEPSIS_SEQ_LEN", "0")
        if "config" in sys.modules:
            del sys.modules["config"]
        with pytest.raises(ValueError, match="SEQ_LEN"):
            import config  # noqa: F401
        # cleanup
        if "config" in sys.modules:
            del sys.modules["config"]
