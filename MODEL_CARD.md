# Model Card — Neonatal Sepsis Detection

## Model Overview

| Field | Value |
|---|---|
| **Task** | Binary classification — sepsis onset prediction from clinical time-series |
| **Input** | 48-timestep × 40-feature ICU window per patient |
| **Output** | Probability of sepsis onset; binary label at calibrated threshold |
| **Architecture** | TimeSeriesTransformer (default) · GRU-D · Ensemble blend |
| **Training paradigm** | Federated Learning (Flower) + local baseline |
| **Version** | improvements/v2 |

---

## Intended Use

- **Primary use:** Early warning of sepsis in ICU patients using routinely collected clinical data
- **Intended users:** Clinical researchers, ML engineers building decision-support tools
- **Deployment context:** Research / decision-support aid — not a standalone diagnostic device

### Out-of-scope uses
- Direct clinical decision-making without physician oversight
- Deployment without institutional IRB approval and HIPAA compliance review
- Populations outside the training distribution (non-ICU, paediatric-only units not represented in training data)

---

## Dataset

| Field | Value |
|---|---|
| **Source** | PhysioNet / CinC Challenge 2019 |
| **Size** | ~40,000 ICU patient records |
| **Split** | 70% train · 15% val · 15% test (frozen, patient-level stratified) |
| **Label** | `SepsisLabel` — 1 at sepsis onset hour and all subsequent hours |
| **Label prevalence** | Imbalanced (~8–10% positive) |
| **Temporal resolution** | 1 record per ICU hour |

### Features (40 total)

| Group | Features |
|---|---|
| Vital signs (8) | HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2 |
| Laboratory (26) | BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets |
| Demographics (6) | Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS |

Missing values are common (labs are sparse). GRU-D handles missingness natively via decay; Transformer receives the raw values with a binary observation mask.

---

## Architecture

### TimeSeriesTransformer
- Linear projection → Transformer encoder (2 layers, 4 heads, d_model=128, dim_ff=256)
- CLS token pooling — attends to all timesteps, pad-masked for front-padded sequences
- Sinusoidal positional encoding (trainable fine-tuning)
- Output: sigmoid probability

### GRU-D
- Faithful implementation: per-feature exponential decay, empirical mean imputation, hidden decay
- Handles irregular missingness and varying observation intervals natively

### Ensemble
- Weighted blend of Transformer and GRU-D probabilities (alpha=0.5 default)
- Temperature Scaling calibration applied post-training (LBFGS on val set)

---

## Training

| Hyperparameter | Value |
|---|---|
| Loss | Focal loss (γ=2.0) |
| Optimizer | AdamW |
| LR schedule | Linear warmup (3 epochs) + cosine annealing |
| Gradient clipping | 1.0 |
| Batch size | 64 |
| Early stopping | Patience 10 (val AUROC) |
| Augmentation | Gaussian jitter on features (augment=True) |
| Calibration | Temperature Scaling |

### Federated Learning
| Setting | Value |
|---|---|
| Strategy | FedAvg · FedProx (μ=0.01) · FedBN |
| Rounds | 20 |
| Clients | 5 simulated hospitals |
| Local epochs | 1 per round |
| Non-IID simulation | Heterogeneous class distribution across clients (optional) |

---

## Evaluation

Metrics reported on the **frozen held-out test set** (6,049 patients, 440 sepsis-positive;
never seen during training or FL rounds):

### Discrimination

| Model | AUROC | AUPRC |
|---|---|---|
| GRU-D Windowed (Early Warning, 6h ahead) | 0.6002 | 0.0098 |
| GRU-D (FL, FedAvg, non-IID) | 0.8306 | 0.4471 |
| Transformer (FL, FedAvg, IID) | 0.9018 | 0.6359 |
| GRU-D (FL, FedBN, IID) | 0.9051 | 0.6404 |
| Transformer (local) | 0.9092 | 0.6276 |
| GRU-D (local) | 0.9189 | 0.6912 |
| GRU-D (FL, FedAvg, IID) | 0.9238 | 0.6955 |
| **Ensemble (Transformer + GRU-D, local)** | **0.9293** | **0.7020** |

### Clinical Operating Points (patient-level models, at calibrated threshold)

| Model | Threshold | Sensitivity | Specificity | PPV | NNAlert | FA/day |
|---|---|---|---|---|---|---|
| GRU-D (local) | 0.3539 | 0.805 | 0.885 | 0.354 | 2.8 | 0.05 |
| Transformer (local) | 0.3397 | 0.795 | 0.857 | 0.303 | 3.3 | 0.07 |
| GRU-D (FL, FedAvg, IID) | 0.3500 | 0.600 | 0.977 | 0.670 | 1.5 | 0.01 |
| Ensemble | 0.3500 | 0.827 | 0.884 | 0.358 | 2.8 | 0.05 |

**NNAlert** = number of alerts issued per true positive caught. **FA/day** = false alerts per patient-day.

### Metric definitions

| Metric | Description |
|---|---|
| AUROC | Overall discrimination |
| AUPRC | Precision-recall (preferred for imbalanced data) |
| Sensitivity @ 0.95 specificity | GRU-D: 0.693 at threshold 0.6775 |
| Specificity @ 0.90 sensitivity | GRU-D: 0.754 at threshold 0.1187 |
| Alert fatigue rate | False alerts per patient-day at calibrated threshold |
| NNAlert | Alerts needed to catch one true positive (= 1/PPV) |
| Bootstrap 95% CI | 1000 bootstrap resamples (shown in dashboard) |

Subgroup analysis is available by `Age`, `Gender`, and ICU unit via `src/clinical_metrics.py`.
See `CLINICAL_VALIDATION.md` for instructions.

---

## Limitations and Known Failure Modes

- **Data source:** Training data is from the PhysioNet 2019 Challenge (US ICUs). Performance may
  degrade on ICUs with different charting practices, lab panel composition, or patient demographics.
  The dataset is not exclusively neonatal — it includes general adult ICU patients.
- **Missing data:** GRU-D handles missingness via exponential decay, but patients with very few
  lab draws (sparse observation) receive mean-imputed values throughout, which can create
  artificially low decay signals and unreliable predictions.
- **Very short ICU stays (< 12h):** Fewer timesteps provide less temporal context for GRU-D's
  decay mechanism; performance expected to be lower in this subgroup.
- **Late-onset sepsis:** Patient-level model predicts overall risk across the full stay. Patients
  who develop sepsis only in the final hours of a long stay may not receive timely alerts.
- **Temporal leakage risk:** The windowed early-warning model uses a 6-hour forward horizon;
  deployment must strictly respect temporal ordering — never use future data as input.
- **Calibration under shift:** Temperature Scaling was fitted on the validation set from the same
  PhysioNet distribution. Calibration guarantees break under significant covariate shift.
- **Windowed model not deployment-ready:** AUROC=0.6002, AUPRC=0.0098 — above chance but
  insufficient precision for clinical alerting. Further development required.
- **Federated simulation gap:** Client splits are simulated from a single dataset; real-world
  heterogeneity between hospitals (different EHR systems, charting workflows) will differ
  substantially and likely reduce federated model performance.

---

## Ethical Considerations

- Model outputs are probabilistic estimates — clinical decisions must involve physician judgment.
- Subgroup disparities (age, gender, unit type) should be audited before any deployment.
- PHI handling must comply with HIPAA and relevant institutional data governance policies.
  See `DATA_GOVERNANCE.md` for a full audit of data flows and API PHI exposure.
- **The model was not validated in a prospective clinical trial.** All metrics are retrospective.
  Prospective validation, external validation on a separate institution, IRB approval, and
  regulatory clearance are required before clinical use. See `CLINICAL_VALIDATION.md`.
- The `patient_id` field in the API audit log (`predictions.jsonl`) is caller-supplied and may
  contain PHI if the caller passes a real patient identifier. De-identified tokens are recommended.

---

## Files

| File | Description |
|---|---|
| `src/model.py` | TimeSeriesTransformer |
| `src/model_grud.py` | GRU-D |
| `src/ensemble.py` | Ensemble blending |
| `src/calibration.py` | Temperature Scaling |
| `src/clinical_metrics.py` | Clinical evaluation metrics |
| `src/api.py` | FastAPI v2 serving |
| `configs/base.yaml` | All hyperparameters |
| `scripts/run_pipeline.py` | End-to-end pipeline orchestrator |

---

## Citation

If you use this work, please cite the PhysioNet 2019 Challenge dataset:

```
Reyna MA, et al. Early Prediction of Sepsis from Clinical Data: The PhysioNet/
Computing in Cardiology Challenge 2019. Critical Care Medicine, 2020.
```
