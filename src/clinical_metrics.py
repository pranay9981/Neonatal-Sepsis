# src/clinical_metrics.py
"""
Clinical evaluation metrics for early-warning sepsis detection.

Beyond AUROC/AUPRC, clinicians care about:
  - Sensitivity at fixed specificity (and vice versa)
  - Time-to-alert: hours of warning before clinical diagnosis
  - Alert fatigue rate: false alerts per patient-day
  - NNAlert: alerts needed to catch one true case
  - Subgroup performance (gestational age, birth weight, ICULOS quartile)
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def sensitivity_at_specificity(y_true, y_prob, target_specificity: float = 0.95):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Exclude sklearn's artificial boundary point (threshold = max_score+1, not a valid probability).
    valid = np.isfinite(thresholds) & (thresholds >= 0.0) & (thresholds <= 1.0)
    if valid.any():
        fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
    specificity = 1.0 - fpr
    idx = np.argmin(np.abs(specificity - target_specificity))
    return float(tpr[idx]), float(thresholds[idx])


def specificity_at_sensitivity(y_true, y_prob, target_sensitivity: float = 0.90):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.isfinite(thresholds) & (thresholds >= 0.0) & (thresholds <= 1.0)
    if valid.any():
        fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
    idx = np.argmin(np.abs(tpr - target_sensitivity))
    return float(1.0 - fpr[idx]), float(thresholds[idx])


def alert_fatigue_rate(y_true, y_pred_binary, patient_hours: float | None = None):
    """False alerts per patient-hour (or per 24h if patient_hours=None uses len as hours)."""
    fp = int(((y_pred_binary == 1) & (y_true == 0)).sum())
    hours = patient_hours if patient_hours is not None else float(len(y_true))
    return fp / max(1.0, hours / 24.0)


def nna_lert(y_true, y_pred_binary):
    """Number Needed to Alert: alerts per true positive caught."""
    tp = int(((y_pred_binary == 1) & (y_true == 1)).sum())
    alerts = int((y_pred_binary == 1).sum())
    return float(alerts) / max(1, tp)


def compute_all(y_true, y_prob, threshold: float = 0.5, patient_hours: float | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    results = {}
    if len(np.unique(y_true)) < 2:
        return {"error": "single_class"}

    sens_at_95spec, _ = sensitivity_at_specificity(y_true, y_prob, 0.95)
    spec_at_90sens, _ = specificity_at_sensitivity(y_true, y_prob, 0.90)
    results["sensitivity_at_95spec"] = sens_at_95spec
    results["specificity_at_90sens"] = spec_at_90sens
    results["alert_fatigue_rate_per_day"] = alert_fatigue_rate(y_true, y_pred, patient_hours)
    results["nn_alert"] = nna_lert(y_true, y_pred)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    results["tp"], results["fp"], results["fn"], results["tn"] = tp, fp, fn, tn
    results["precision"] = tp / max(1, tp + fp)
    results["recall"] = tp / max(1, tp + fn)
    results["f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
    return results


def subgroup_analysis(y_true, y_prob, groups: dict, threshold: float = 0.5) -> dict:
    """
    Compute clinical metrics per subgroup.
    groups: dict of {group_name: boolean_mask_array}
    Returns: dict of {group_name: metrics_dict}
    """
    results = {}
    for name, mask in groups.items():
        mask = np.asarray(mask, dtype=bool)
        yt = np.asarray(y_true)[mask]
        yp = np.asarray(y_prob)[mask]
        if len(yt) == 0:
            results[name] = {"error": "empty_group"}
            continue
        results[name] = compute_all(yt, yp, threshold)
        results[name]["n"] = int(mask.sum())
    return results
