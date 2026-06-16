# Clinical Validation — Neonatal Sepsis Detection

## Overview

This document reports clinical operating-point metrics beyond AUROC/AUPRC, threshold selection rationale,
known failure modes, subgroup analysis guidance, and the prospective-validation disclaimer required before
any clinical deployment.

All results are on the **frozen held-out test set** (6,049 patients, 440 sepsis-positive; never seen
during training, hyperparameter search, or federated rounds).

---

## 1. Clinical Operating-Point Summary

### Recommended model for clinical use: GRU-D local (threshold = 0.3539)

| Metric | Value |
|---|---|
| AUROC | 0.9190 |
| AUPRC | 0.6913 |
| Sensitivity @ 0.95 specificity | 0.693 |
| Specificity @ 0.90 sensitivity | 0.754 |
| **Sensitivity at calibrated threshold (0.3539)** | **0.805** |
| **Specificity at calibrated threshold (0.3539)** | **0.885** |
| PPV (precision) at threshold | 0.354 |
| False alerts per patient-day | 0.05 |
| NNAlert (alerts per true positive caught) | 2.8 |
| TP / FP / FN / TN | 354 / 646 / 86 / 4 963 |

**Interpretation:** At the calibrated threshold, the model alerts on 80.5% of sepsis cases
(354/440) while correctly clearing 88.5% of non-septic patients. For every true positive caught,
approximately 2.8 alerts are issued — clinically acceptable for an ICU early-warning tool where
missed sepsis is more costly than a false alert.

---

## 2. All-Model Clinical Comparison

| Model | AUROC | Threshold | Sensitivity | Specificity | PPV | NNAlert | FA/day |
|---|---|---|---|---|---|---|---|
| GRU-D (local) | 0.9190 | 0.3539 | 0.805 | 0.885 | 0.354 | 2.8 | 0.05 |
| Transformer (local) | 0.9092 | 0.3397 | 0.795 | 0.857 | 0.303 | 3.3 | 0.07 |
| GRU-D FL FedAvg IID | 0.9238 | 0.3500 | 0.600 | 0.977 | 0.670 | 1.5 | 0.01 |
| Ensemble (Transformer + GRU-D) | 0.9293 | 0.3500 | 0.827 | 0.884 | 0.358 | 2.8 | 0.05 |

**Key observations:**
- **GRU-D FL FedAvg** at threshold 0.35 achieves the highest precision (PPV=0.670) and lowest
  alert fatigue (0.01 FA/day) at the cost of sensitivity (0.600). Best for settings where alert
  overload is the primary concern.
- **Ensemble** achieves the highest sensitivity (0.827) with equivalent alert fatigue to GRU-D local
  — preferred if both models are available at inference time.
- **Transformer** has the highest alert fatigue and lowest PPV — use GRU-D or Ensemble instead.

---

## 3. Threshold Selection Rationale

Thresholds were calibrated on the **validation set** using the best-AUROC checkpoint
(not the last epoch) to avoid threshold leakage from training dynamics.

The reported threshold for GRU-D (0.3539) was chosen as the decision boundary from Temperature
Scaling calibration. In a real deployment, the operating threshold should be re-selected
based on the clinical trade-off:

| Operating point | Target | Resulting threshold | Use case |
|---|---|---|---|
| High sensitivity | Sens ≥ 0.90 | ~0.1187 (GRU-D) | Screen to miss no case (ICU triage) |
| Balanced | Val-calibrated | 0.3539 (GRU-D) | Standard bedside alerting |
| High specificity | Spec ≥ 0.95 | ~0.6775 (GRU-D) | Reduce alert fatigue in busy units |

A clinical deployment would work with bedside nursing staff to select the target operating point
before go-live.

---

## 4. Sensitivity / Specificity Trade-off Reference

For the GRU-D local model (recommended):

| Clinical operating point | Sensitivity | Specificity | Threshold |
|---|---|---|---|
| Sensitivity @ 0.95 specificity | 0.693 | 0.950 | 0.6775 |
| Calibrated (default) | 0.805 | 0.885 | 0.3539 |
| Specificity @ 0.90 sensitivity | 0.900 | 0.754 | 0.1187 |

---

## 5. Subgroup Analysis

### What is available

`src/clinical_metrics.py` includes `subgroup_analysis(y_true, y_prob, groups, threshold)` which
accepts arbitrary boolean group masks and returns per-group clinical metrics.

### How to run it

```python
import json, torch, numpy as np
from src.clinical_metrics import subgroup_analysis

# Load eval results
with open("eval_results_grud.json") as f:
    d = json.load(f)
y_true = np.array(d["y_true"])
y_prob  = np.array(d["y_prob"])

# Load test index (has demographics) and align with eval order
test_index = torch.load("data/splits/test_index.pt")

# Define group masks — example by ICULOS quartile
# (requires loading actual patient .pt files to extract demographics)
groups = {
    "short_stay":  iculos < 24,
    "long_stay":   iculos >= 24,
    "unit1":       unit1 == 1,
    "unit2":       unit2 == 1,
}
results = subgroup_analysis(y_true, y_prob, groups, threshold=0.3539)
```

### Known limitation

The eval JSONs store only `y_true` and `y_prob`. Subgroup analysis requires loading the
preprocessed patient `.pt` files to extract `Age`, `Gender`, and `Unit1`/`Unit2` features.
These files are in `data/processed/patients/` (gitignored — must regenerate from raw PhysioNet
PSV files via `python src/parallel_preprocess.py`).

### Expected subgroup risks

Based on the training data distribution (PhysioNet 2019):
- **Very short ICU stays (< 12h):** Lower performance expected — fewer timesteps, less temporal
  context for decay-based features in GRU-D.
- **Sparse lab draws:** Patients with very few observed lab values (high missingness) may have
  less reliable predictions. GRU-D's empirical mean imputation fills gaps but is not a substitute
  for real measurements.
- **Non-neonatal ICU:** The PhysioNet dataset contains general adult ICU patients, not exclusively
  neonatal. Performance on neonatal-specific populations is unvalidated.

---

## 6. Early Warning Model (Windowed GRU-D)

| Metric | Value |
|---|---|
| Task | Prospective 6h-ahead sepsis onset prediction |
| AUROC | 0.6002 |
| AUPRC | 0.0098 |
| Positive rate | ~0.5% (window level) |

**Status:** The windowed model AUROC (0.6002) is above chance but substantially below the
patient-level model (0.9190). This is expected — prospective prediction from a 12-hour window
is a harder task than patient-level classification from the full ICU stay.

**Not ready for clinical use.** The AUPRC of 0.0098 means the model has very low precision
at any useful sensitivity level. Further work needed: longer window context, richer feature
engineering, larger positive-rate oversampling.

---

## 7. Known Failure Modes

1. **Distribution shift:** Trained on PhysioNet 2019 (US ICUs). Performance will degrade on
   ICUs with different charting practices, lab panel composition, or patient mix.

2. **High-missingness patients:** GRU-D uses empirical means for imputation. Patients who are
   never measured on key features (e.g. no lactate drawn) receive mean-imputed values throughout,
   which can create artificially low decay signals.

3. **Late-onset sepsis:** The model predicts patient-level sepsis risk. For patients who develop
   sepsis only in the final hours of a long ICU stay, the model may not raise alerts until very
   late.

4. **Windowed model positive rate:** The 12h-window dataset has ~0.5% positive rate. The model
   is heavily biased toward the negative class. Focal loss partially compensates but calibrated
   probabilities remain very low.

5. **Federated simulation gap:** The federated models were trained on simulated clients from a
   single dataset. Real-world FL across hospitals with different EHR systems will exhibit
   greater heterogeneity and likely lower AUROC.

6. **Temperature Scaling under shift:** Calibration was fitted on the validation set from the
   same PhysioNet distribution. Under significant covariate shift, calibration guarantees break.

---

## 8. Prospective Validation Disclaimer

**This model has not been validated in a prospective clinical trial.**

All metrics reported in this document are based on retrospective evaluation on the held-out
PhysioNet 2019 test set. Before clinical deployment:

- [ ] Prospective validation study required (IRB approval needed)
- [ ] External validation on a separate institution's EHR data
- [ ] Clinician workflow integration study
- [ ] Regulatory clearance (FDA 510(k) or equivalent in applicable jurisdiction)
- [ ] Formal subgroup analysis on representative patient cohort
- [ ] Ongoing performance monitoring plan (model drift detection)

This project demonstrates the research and engineering approach. It is not a medical device.

---

## 9. How to Reproduce These Results

```bash
# Evaluate GRU-D on frozen test set
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt runs/20260614T064920Z__local_grud/checkpoints/model_best.pt \
  --model grud \
  --out_file eval_results_grud.json

# Compute clinical metrics from eval JSON
python - <<'EOF'
import json, numpy as np
from src.clinical_metrics import sensitivity_at_specificity, specificity_at_sensitivity, \
    alert_fatigue_rate, nna_lert

with open("eval_results_grud.json") as f:
    d = json.load(f)
y_true = np.array(d["y_true"])
y_prob  = np.array(d["y_prob"])
threshold = 0.3539

sens95, _ = sensitivity_at_specificity(y_true, y_prob, 0.95)
spec90, _ = specificity_at_sensitivity(y_true, y_prob, 0.90)
y_pred = (y_prob >= threshold).astype(int)
afr = alert_fatigue_rate(y_true, y_pred, patient_hours=float(len(y_true) * 48))
nna = nna_lert(y_true, y_pred)

print(f"Sens@0.95spec: {sens95:.3f}")
print(f"Spec@0.90sens: {spec90:.3f}")
print(f"Alert fatigue: {afr:.3f} FA/day")
print(f"NNAlert: {nna:.1f}")
EOF
```
