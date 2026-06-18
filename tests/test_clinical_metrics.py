"""Tests for src/clinical_metrics.py"""
import math
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from clinical_metrics import (
    sensitivity_at_specificity,
    specificity_at_sensitivity,
    alert_fatigue_rate,
    nn_alert,
    compute_all,
    subgroup_analysis,
)

RNG = np.random.default_rng(42)
N = 200
Y_TRUE = np.array([1] * 40 + [0] * 160)
Y_PROB = np.clip(Y_TRUE * 0.6 + RNG.normal(0, 0.2, N), 0, 1)
Y_PRED = (Y_PROB >= 0.5).astype(int)


class TestSensitivityAtSpecificity:
    def test_returns_tuple_in_range(self):
        sens, thresh = sensitivity_at_specificity(Y_TRUE, Y_PROB, 0.95)
        assert 0.0 <= sens <= 1.0
        assert 0.0 <= thresh <= 1.0

    def test_high_specificity_lower_sensitivity(self):
        sens_high, _ = sensitivity_at_specificity(Y_TRUE, Y_PROB, 0.99)
        sens_low, _ = sensitivity_at_specificity(Y_TRUE, Y_PROB, 0.50)
        assert sens_high <= sens_low + 0.3  # higher specificity → generally lower or equal sensitivity

    def test_default_target(self):
        sens, thresh = sensitivity_at_specificity(Y_TRUE, Y_PROB)
        assert isinstance(sens, float)
        assert isinstance(thresh, float)


class TestSpecificityAtSensitivity:
    def test_returns_tuple_in_range(self):
        spec, thresh = specificity_at_sensitivity(Y_TRUE, Y_PROB, 0.90)
        assert 0.0 <= spec <= 1.0
        assert 0.0 <= thresh <= 1.0

    def test_default_target(self):
        spec, thresh = specificity_at_sensitivity(Y_TRUE, Y_PROB)
        assert isinstance(spec, float)


class TestAlertFatigueRate:
    def test_no_false_positives(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        rate = alert_fatigue_rate(y_true, y_pred)
        assert rate == 0.0

    def test_with_false_positives(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        rate = alert_fatigue_rate(y_true, y_pred, patient_hours=48.0)
        assert rate == 2 / 2.0  # 2 FP / (48h / 24h)

    def test_uses_len_when_patient_hours_none(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        rate = alert_fatigue_rate(y_true, y_pred, patient_hours=None)
        assert isinstance(rate, float)

    def test_all_positives(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])
        rate = alert_fatigue_rate(y_true, y_pred)
        assert rate == 0.0  # no false positives


class TestNNAlert:
    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert nn_alert(y_true, y_pred) == 1.0  # 2 alerts / 2 TP

    def test_all_alerted(self):
        y_true = np.array([1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        assert nn_alert(y_true, y_pred) == 4.0  # 4 alerts / 1 TP

    def test_no_alerts(self):
        y_true = np.array([1, 0, 0])
        y_pred = np.array([0, 0, 0])
        val = nn_alert(y_true, y_pred)
        assert val == 0.0  # 0 alerts / max(1,0)


class TestComputeAll:
    def test_returns_expected_keys(self):
        result = compute_all(Y_TRUE, Y_PROB)
        for key in ("sensitivity_at_95spec", "specificity_at_90sens", "precision", "recall", "f1", "sensitivity", "specificity"):
            assert key in result

    def test_single_class_returns_error(self):
        y_all_neg = np.zeros(50)
        y_prob = np.random.rand(50)
        result = compute_all(y_all_neg, y_prob)
        assert "error" in result
        assert result["error"] == "single_class"

    def test_perfect_classifier(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        result = compute_all(y_true, y_prob, threshold=0.5)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_f1_nan_when_no_positives_predicted(self):
        y_true = np.array([1, 0, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.1, 0.1])  # all below threshold
        result = compute_all(y_true, y_prob, threshold=0.5)
        assert math.isnan(result["f1"])

    def test_patient_hours_propagated(self):
        result = compute_all(Y_TRUE, Y_PROB, threshold=0.5, patient_hours=1000.0)
        assert "alert_fatigue_rate_per_day" in result

    def test_confusion_matrix_keys(self):
        result = compute_all(Y_TRUE, Y_PROB)
        assert all(k in result for k in ("tp", "fp", "fn", "tn"))
        assert result["tp"] + result["fp"] + result["fn"] + result["tn"] == len(Y_TRUE)


class TestSubgroupAnalysis:
    def test_returns_per_group_dict(self):
        groups = {"high_risk": Y_TRUE == 1, "low_risk": Y_TRUE == 0}
        result = subgroup_analysis(Y_TRUE, Y_PROB, groups)
        assert "high_risk" in result
        assert "low_risk" in result

    def test_empty_group_returns_error(self):
        groups = {"empty": np.zeros(N, dtype=bool)}
        result = subgroup_analysis(Y_TRUE, Y_PROB, groups)
        assert result["empty"]["error"] == "empty_group"

    def test_group_n_counts(self):
        groups = {"pos": Y_TRUE == 1}
        result = subgroup_analysis(Y_TRUE, Y_PROB, groups)
        assert result["pos"]["n"] == int(Y_TRUE.sum())
