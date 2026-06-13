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

Metrics reported on the **frozen held-out test set** (never seen during training or FL rounds):

| Metric | Description |
|---|---|
| AUROC | Overall discrimination |
| AUPRC | Precision-recall (preferred for imbalanced data) |
| Sensitivity @ 0.90 specificity | Clinical operating point |
| Alert fatigue rate | False alerts per true alert |
| NNAlert | Number needed to alert (1 / PPV) |
| Bootstrap 95% CI | 1000 bootstrap resamples for all metrics |

Subgroup analysis is available by `Age`, `Gender`, and ICU unit.

---

## Limitations

- **Data source:** Training data is from the PhysioNet 2019 Challenge and may not represent all ICU populations, particularly neonatal-specific units.
- **Missing data:** High sparsity in laboratory values may degrade performance for patients with very few lab draws.
- **Temporal leakage risk:** The sliding-window early-warning setup uses a 6-hour forward horizon; deployment must strictly respect temporal ordering.
- **Calibration:** Temperature Scaling improves calibration but does not guarantee calibration under distribution shift.
- **Federated simulation:** Client splits are simulated from a single dataset; real-world heterogeneity between hospitals will differ.

---

## Ethical Considerations

- Model outputs are probabilistic estimates — clinical decisions must involve physician judgment.
- Subgroup disparities (age, gender, unit type) should be audited before any deployment.
- PHI handling must comply with HIPAA and relevant institutional data governance policies.
- The model was not validated in a prospective clinical trial.

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
