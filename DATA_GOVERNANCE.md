# Data Governance — Neonatal Sepsis Detection

## Purpose

This document describes how patient data is handled in this project, confirms de-identification
status, maps data flows, audits the API for PHI exposure, and states compliance scope.

---

## 1. Data Source and License

| Field | Value |
|---|---|
| **Dataset** | PhysioNet / Computing in Cardiology Challenge 2019 |
| **URL** | https://physionet.org/content/challenge-2019/1.0.0/ |
| **License** | PhysioNet Credentialed Health Data License 1.5.0 |
| **Access** | Requires PhysioNet credentialed account + signed DUA |
| **Size** | 40,336 ICU patient records (Set A: 20,336 + Set B: 20,000) |
| **Format** | One pipe-separated `.psv` file per patient |

### De-identification

PhysioNet has already applied de-identification to this dataset before public release:
- No patient names, dates of birth, admission dates, or provider identifiers
- Patient IDs are anonymous sequential identifiers (e.g. `p000001.psv`)
- No dates: only relative time (`ICULOS` = hours since ICU admission)

**This project does not re-identify or attempt to link any patient record.**

Reference: Reyna MA et al., "Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing
in Cardiology Challenge 2019." *Critical Care Medicine*, 2020.

---

## 2. Data Flow

```
PhysioNet PSV files (raw)
        │  src/parallel_preprocess.py
        ▼
data/processed/patients/   ← .pt tensors per patient (X, mask, deltas, y, y_seq, onset_hour)
data/processed/patients/scaler.json  ← feature means/stds (train-only, no patient IDs)
        │  scripts/create_splits.py
        ▼
data/splits/train_index.pt, val_index.pt, test_index.pt  ← COMMITTED to git (indices only)
        │  src/train_local.py  /  src/fl_server.py + fl_client.py
        ▼
runs/<timestamp>/checkpoints/model_best.pt  ← model weights only, no patient data
server_out/global_best.pt                   ← federated global model weights
        │  src/evaluate.py
        ▼
eval_results_*.json  ← y_true array + y_prob array + AUROC/AUPRC (no identifiers)
        │  src/api.py
        ▼
predictions.jsonl   ← audit log (see Section 4)
```

---

## 3. What Is and Is Not Committed to Git

| Path | Committed | Reason |
|---|---|---|
| `data/splits/train_index.pt` | Yes | Index of patient positions — no feature values |
| `data/splits/val_index.pt` | Yes | Same |
| `data/splits/test_index.pt` | Yes | Same |
| `data/processed/patients/` | **No** (gitignored) | Contains raw feature tensors |
| `data/processed/windows/` | **No** (gitignored) | Windowed feature tensors |
| `data/processed/windows_test/` | **No** (gitignored) | Windowed test tensors |
| `data/raw/` | **No** (gitignored) | Original PhysioNet PSV files |
| `eval_results_*.json` | **No** (gitignored) | Contains y_true/y_prob arrays |
| `mlruns/` | **No** (gitignored) | MLflow artifact store |
| `mlflow.db` | **No** (gitignored) | MLflow metadata |
| `predictions.jsonl` | **No** (gitignored) | Runtime audit log |
| `MODEL_CARD.md` | Yes | Public documentation |
| `CLINICAL_VALIDATION.md` | Yes | Public documentation |
| `DATA_GOVERNANCE.md` | Yes | Public documentation |

The `.gitignore` enforces this. The split index files contain only integer positions, no feature
values or identifiable information.

---

## 4. API Audit Log — PHI Analysis

`src/api.py` appends one JSON line per prediction to `predictions.jsonl`. The exact fields logged:

```json
{
  "patient_id":     "<string supplied by caller, or null>",
  "probability":    0.42,
  "risk_level":     "MODERATE",
  "alert":          false,
  "model_version":  "model_best",
  "n_timesteps":    48,
  "latency_ms":     12.3,
  "timestamp":      1718527200.0
}
```

### PHI risk assessment

| Field | PHI? | Notes |
|---|---|---|
| `patient_id` | **Conditional** | Whatever the caller passes. See below. |
| `probability` | No | Model output — not a clinical record |
| `risk_level` | No | Derived from probability |
| `alert` | No | Derived from probability |
| `model_version` | No | Checkpoint filename |
| `n_timesteps` | No | Input shape |
| `latency_ms` | No | Performance metric |
| `timestamp` | No | Unix epoch — no patient link |

**Raw feature values are never written to the audit log.** The model input (40-feature
time-series) is consumed entirely in memory and discarded after inference.

### Patient ID guidance

The `patient_id` field is an **optional string supplied by the API caller**. The API itself does
not generate or look up any identifier. In a clinical deployment:

- If the caller passes a real MRN or name, that value is written to the JSONL file.
- The audit log file must then be treated as containing PHI and protected accordingly.
- **Recommended practice:** Pass a de-identified encounter token (e.g. SHA-256 hash of MRN
  salted with a site-specific secret) rather than the raw MRN.

---

## 5. Encryption and Access Controls

### Current state (research / local deployment)

| Control | Status | Notes |
|---|---|---|
| Encryption at rest | Not implemented | Local dev machine only |
| Encryption in transit | Not implemented | API on localhost / Docker bridge network |
| Authentication | None | API has no auth layer |
| Role-based access | None | Single-user local project |
| Network exposure | Loopback / Docker | Not exposed to internet |

### Requirements for production clinical deployment

| Control | Requirement |
|---|---|
| Encryption at rest | AES-256 for all PHI-containing files (audit log, patient tensors) |
| Encryption in transit | TLS 1.2+ (HTTPS) — do not run uvicorn without a TLS-terminating proxy |
| Authentication | OAuth2 / API key with per-user scopes |
| Audit log integrity | Append-only log shipped to immutable storage (e.g. AWS CloudTrail) |
| Access logging | All `/predict` calls logged with caller identity |
| Data retention | Retain audit logs per institutional policy (typically 6 years under HIPAA) |

These controls are out of scope for this research prototype and must be implemented before
any clinical deployment.

---

## 6. Federated Learning — Data Governance Notes

The federated learning simulation (`src/fl_server.py` + `src/fl_client.py`) runs entirely on
one machine. In real-world FL deployment:

- Each hospital client trains locally — **raw patient data never leaves the hospital network**
- Only model weight updates (gradients or averaged parameters) are transmitted
- FedAvg aggregation at the server cannot reconstruct raw training data from weight updates alone
- Differential privacy (DP-SGD noise addition) would be the next step for formal privacy guarantees;
  this project does not implement DP but the architecture is compatible with adding it

---

## 7. Data Use Agreement Compliance

Use of the PhysioNet 2019 dataset requires agreement to the PhysioNet Credentialed Health Data
License. Key obligations:

- [ ] Do not attempt to identify individuals
- [ ] Do not share raw data files publicly (covered by `.gitignore`)
- [ ] Cite the dataset in any publication
- [ ] Report any suspected re-identification to PhysioNet

This project is compliant with all the above: raw data is gitignored, the dataset is cited in
`MODEL_CARD.md`, and no re-identification is attempted.

---

## 8. HIPAA Applicability

HIPAA applies to **covered entities** (healthcare providers, health plans, clearinghouses) and
their **business associates**. This project:

- Is a research prototype, not a deployed clinical system
- Does not operate as a covered entity or business associate
- Does not store, transmit, or receive Protected Health Information in production

**If this system were deployed in a clinical setting**, HIPAA Technical Safeguards would require:
- Access controls (unique user identification, automatic logoff)
- Audit controls (hardware, software, procedural mechanisms to record access)
- Integrity controls (preventing unauthorized PHI alteration)
- Transmission security (encryption of PHI in transit)

The API audit log, model serving infrastructure, and data storage would all fall under these
requirements. A formal HIPAA risk analysis (per 45 CFR § 164.308(a)(1)) would be required
before go-live.
