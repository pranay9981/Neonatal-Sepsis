<div align="center">

# Neonatal Sepsis Detection

### Early ICU Sepsis Warning via Federated Learning

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.23.0-00A9A5?style=flat-square)](https://flower.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-75%20passing-2ea44f?style=flat-square)](./tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)

A research system for early detection of sepsis in ICU patients using clinical time-series data.
Combines a **Transformer** and **GRU-D** ensemble with **Federated Learning** — enabling multi-hospital
training without sharing patient data.

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Dashboard](#-dashboard) · [API](#-api-server) · [Model Card](./MODEL_CARD.md)

</div>

---

## Overview

Sepsis is a life-threatening condition where early detection dramatically improves patient outcomes. This system trains on 40,000+ ICU patient records from the PhysioNet 2019 Challenge, learning to predict sepsis onset from routine clinical measurements — vitals, lab values, and demographics — up to **6 hours in advance**.

The federated design means individual hospitals never share raw patient data. Each institution trains locally; only model weights are aggregated — making this architecture viable for real-world deployment under HIPAA and similar data governance requirements.

| | |
|---|---|
| **Input** | 40-feature × 48-timestep ICU windows per patient (vitals, labs, demographics) |
| **Models** | TimeSeriesTransformer · GRU-D · Ensemble · Temperature Scaling calibration |
| **Training** | Focal loss · Warmup + cosine LR · Early stopping · MLflow tracking · Optuna HPO |
| **Federated** | Flower with FedAvg / FedProx / FedBN strategies · Non-IID simulation |
| **Evaluation** | Frozen 70/15/15 split · AUROC · AUPRC · Clinical metrics · 95% bootstrap CIs |
| **Serving** | FastAPI v2 · MC Dropout uncertainty · Prometheus metrics · Audit log |
| **Dashboard** | 5-page Streamlit app · Interactive ROC/PRC · Clinical metric explorer |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                               │
│                                                                     │
│  data/raw/*.psv  ──►  parallel_preprocess.py  ──►  .pt tensors     │
│                              │                    (X, mask, Δt,    │
│                              ▼                     y, y_seq)        │
│                    create_splits.py                                 │
│                    70% train / 15% val / 15% test (frozen)          │
└─────────────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│  LOCAL TRAINING │  │ FEDERATED (FL)  │  │   HYPERPARAMETER    │
│                 │  │                 │  │      SEARCH         │
│ train_local.py  │  │  split_clients  │  │  optuna_search.py   │
│  Transformer /  │  │       │         │  │  (TPE, 50 trials)   │
│    GRU-D        │  │  ┌────┴────┐    │  └─────────────────────┘
│                 │  │  ▼         ▼    │
│  focal loss     │  │ Client1  Client2│
│  warmup+cosine  │  │  │         │    │
│  grad clipping  │  │  └────┬────┘    │
│  temp. scaling  │  │       ▼         │
└────────┬────────┘  │   FL Server     │
         │           │  (FedAvg/FedBN) │
         ▼           └────────┬────────┘
    model_best.pt             ▼
         │            global_best.pt
         └──────────────────┐ │
                            ▼ ▼
                    ┌───────────────┐
                    │  EVALUATION   │
                    │  (frozen test)│
                    │  AUROC·AUPRC  │
                    │  Clinical KPIs│
                    └───────┬───────┘
                            ▼
              ┌─────────────────────────┐
              │  SERVING & DASHBOARD    │
              │                         │
              │  FastAPI  ──  /predict  │
              │  Streamlit ── 5 pages   │
              │  Prometheus ─ /metrics  │
              └─────────────────────────┘
```

### Models

**TimeSeriesTransformer** — Linear projection → 2-layer Transformer encoder (4 heads, d_model=128) → CLS token pooling with pad mask → sigmoid output. Sinusoidal positional encoding.

**GRU-D** — Faithful implementation with per-feature exponential decay, empirical mean imputation, and hidden-state decay. Handles irregular missingness natively — no imputation preprocessing needed.

**Ensemble** — Weighted probability blend of Transformer + GRU-D (α=0.5). Temperature Scaling calibration (LBFGS on val set) applied post-training.

---

## Quick Start

```bash
# 1. Clone and switch to the development branch
git clone https://github.com/pranay9981/Neonatal-Sepsis.git
cd Neonatal-Sepsis
git checkout improvements/v2

# 2. Set up environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 3. Place raw .psv files in data/raw/, then run everything
python scripts/run_pipeline.py --model transformer --epochs 50 --fl_rounds 20 --n_clients 5

# 4. Launch the dashboard
streamlit run app.py
```

> **New here?** See [GETTING_STARTED.md](./GETTING_STARTED.md) for a detailed step-by-step walkthrough from zero.

---

## Installation

**Requirements:** Python 3.11+, 8 GB RAM (16 GB recommended), 10 GB disk

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import torch, flwr, streamlit, fastapi, mlflow, optuna; print('All packages OK')"
python -m pytest tests/ -v   # 75 tests should pass
```

> **Windows note:** If `flwr` fails, run `pip install flwr==1.23.0 --no-deps` first, then re-run `pip install -r requirements.txt`.

---

## Dataset

Download the **PhysioNet / CinC Challenge 2019** training set from [physionet.org](https://physionet.org/content/challenge-2019/1.0.0/) and place the `.psv` files in `data/raw/`.

```
data/
└── raw/
    ├── p000001.psv
    ├── p000002.psv
    └── ...          ← one file per patient (~40,000 files)
```

Each file is pipe-separated with one row per ICU hour:

```
HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|
Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|
Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|
Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel
```

**40 features** (vitals · labs · demographics) + `SepsisLabel` (0/1). Missing values are `NaN` — handled natively by the pipeline via observation masking and GRU-D decay. Keep the header row in every file.

---

## Pipeline

The orchestrator handles all 7 steps automatically, skipping any step whose outputs already exist:

```bash
python scripts/run_pipeline.py
```

| Flag | Effect |
|---|---|
| `--model transformer\|grud` | Model architecture |
| `--epochs N` | Training epochs (default 10) |
| `--fl_rounds N` | Federated rounds (default 5) |
| `--n_clients N` | Number of simulated hospitals (default 3) |
| `--force_preprocess` | Re-run preprocessing even if outputs exist |
| `--force_train` | Re-run local training |
| `--force_eval` | Re-run evaluation only |
| `--skip_fl` | Local training only, no federated step |
| `--skip_local_train` | Federated only, skip local baseline |

### Step-by-step

<details>
<summary><strong>Step 1 — Preprocess raw data</strong></summary>

Converts each `.psv` into a per-patient `.pt` tensor: `X`, `mask`, `deltas`, `y`, `y_seq`, `onset_hour`.

```bash
python src/parallel_preprocess.py \
  --raw_folder data/raw \
  --out_folder data/processed/patients \
  --seq_len 48 \
  --nprocs 8
```

Output: `data/processed/patients/` — one `.pt` per patient + `index_with_labels.pt`.

</details>

<details>
<summary><strong>Step 2 — Create frozen splits</strong></summary>

Stratified 70/15/15 patient-level split. Run **once** — the test set must never be seen during training.

```bash
python scripts/create_splits.py \
  --index data/processed/patients/index_with_labels.pt \
  --out_dir data/splits
```

Output: `data/splits/train_index.pt`, `val_index.pt`, `test_index.pt`.

</details>

<details>
<summary><strong>Step 3 — Train local baseline</strong></summary>

```bash
python src/train_local.py \
  --index data/splits/train_index.pt \
  --model transformer \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-4 \
  --patience 10 \
  --run_name local_transformer \
  --use_temperature_scaling
```

Best checkpoint saved to `runs/<timestamp>__local_transformer/checkpoints/model_best.pt`.

**Optional flags:**

| Flag | Effect |
|---|---|
| `--use_mlflow` | Log params + metrics to MLflow (`mlflow ui` to view) |
| `--scaler_path <path>` | Normalise features using pre-fit scaler |
| `--augment` | Gaussian jitter augmentation |
| `--use_temperature_scaling` | Calibrate output probabilities after training |

</details>

<details>
<summary><strong>Step 4 — Federated learning simulation</strong></summary>

**Partition patients into client folders (test set excluded):**

```bash
python src/split_clients.py \
  --processed_folder data/processed/patients \
  --out_root data/processed/clients \
  --n_clients 5 \
  --splits_dir data/splits
```

Add `--heterogeneous` for non-IID (skewed class distribution per hospital).

**Run the simulation (automated — server + clients started automatically):**

```bash
python scripts/run_fl_sim.py \
  --client_indexes \
      data/processed/clients/client1/index.pt \
      data/processed/clients/client2/index.pt \
      data/processed/clients/client3/index.pt \
      data/processed/clients/client4/index.pt \
  --model transformer \
  --rounds 20 \
  --local_epochs 1
```

Output: `server_out/global_best.pt`.

**Use FedBN instead of FedAvg:**

```bash
python src/fl_server.py --strategy fedbn --model transformer --rounds 20
```

</details>

<details>
<summary><strong>Step 5 — Evaluate on frozen test set</strong></summary>

```bash
# Federated model
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt server_out/global_best.pt \
  --model transformer \
  --out_file eval_results_federated.json

# Local baseline
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt runs/<your-run>/checkpoints/model_best.pt \
  --model transformer \
  --out_file eval_results_local.json
```

Output JSON includes: `auroc`, `auprc`, `precision`, `recall`, `f1`, `threshold`, `y_true`, `y_prob`, and 95% bootstrap CIs.

</details>

<details>
<summary><strong>Step 6 — Generate comparison plots</strong></summary>

```bash
python src/plot_results.py \
  --results eval_results_federated.json eval_results_local.json \
  --out_file model_comparison_plot.png
```

Produces ROC and PRC curves with bootstrap CI bands for all models.

</details>

---

## Experiment Tools

### 5-Fold Cross-Validation

```bash
python scripts/cross_validate.py \
  --index data/splits/train_index.pt \
  --model transformer \
  --epochs 20
```

Reports mean ± std AUROC and AUPRC across folds with bootstrap CIs.

### Bayesian Hyperparameter Search

```bash
python scripts/optuna_search.py \
  --index data/splits/train_index.pt \
  --n_trials 50
```

Searches `lr`, `d_model`, `n_heads`, `dropout`, `batch_size` using TPE + median pruner. Best params saved to `optuna_best_params.json`.

### Early-Warning Sliding-Window Dataset

Creates one sample per ICU hour with a forward-looking label — *"will sepsis onset within the next 6 hours?"*

```bash
python scripts/create_windowed_dataset.py \
  --index data/processed/patients/index_with_labels.pt \
  --out_dir data/processed/windowed \
  --horizon 6 \
  --stride 1
```

### Clinical Metrics (Python API)

```python
import json
from src.clinical_metrics import compute_all

results = json.load(open("eval_results_federated.json"))
metrics = compute_all(results["y_true"], results["y_prob"], threshold=0.5)
# Returns: sensitivity@spec, alert_fatigue_rate, nn_alert, subgroup breakdown
```

---

## Dashboard

```bash
streamlit run app.py
# Open http://localhost:8501
```

| Page | Description |
|---|---|
| **Project Summary** | Architecture overview, dataset statistics, design decisions |
| **Predict** | Single-patient inference — manual input or file upload |
| **Model Metrics** | Interactive Plotly ROC/PRC with 95% bootstrap CI bands, confusion matrix |
| **Training Runs** | Browse `runs/` artifacts, compare AUROC/AUPRC across experiments |
| **Clinical Metrics** | Sensitivity @ specificity, alert fatigue rate, NNAlert — with explanations |

---

## API Server

```bash
export SEPSIS_MODEL_PATH=server_out/global_best.pt
export SEPSIS_SCALER_PATH=data/processed/patients/scaler.json

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

| Endpoint | Method | Description |
|---|---|---|
| `/v2/predict` | `POST` | Sepsis probability + optional MC Dropout confidence intervals |
| `/health` | `GET` | Model version, uptime |
| `/metrics` | `GET` | Prometheus metrics |

**Example request:**

```bash
curl -X POST http://localhost:8000/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[...]]}'
```

---

## Docker

```bash
# Build
docker build -t neonatal-sepsis .

# Run API server
docker run -p 8000:8000 \
  -v $(pwd)/server_out:/app/server_out:ro \
  neonatal-sepsis

# API + dashboard together
docker-compose up
```

The Dockerfile uses a multi-stage build (builder + slim runtime) and runs as a non-root user (uid=1000) with a `/health` HEALTHCHECK.

---

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Run a specific file
python -m pytest tests/test_model.py -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

**75 tests** covering preprocessing, dataset loading, model forward passes, training loop, FL client/server, evaluation metrics, split correctness, and windowed dataset creation.

---

## Configuration

All hyperparameters live in `configs/base.yaml`. Load programmatically with:

```python
from src.config_schema import load_config
cfg = load_config("configs/base.yaml")
print(cfg.training.lr)       # 0.0001
print(cfg.federated.rounds)  # 20
```

**Key sections:**

```yaml
model:
  name: transformer     # transformer | grud
  d_model: 128
  n_heads: 4
  num_layers: 2

training:
  epochs: 50
  batch_size: 64
  lr: 1.0e-4
  use_focal: true       # focal loss for class imbalance
  augment: true         # Gaussian jitter
  use_temperature_scaling: true

federated:
  rounds: 20
  n_clients: 5
  strategy: fedavg      # fedavg | fedbn
  mu: 0.01              # FedProx proximal term (0 = plain FedAvg)
  heterogeneous: false  # non-IID hospital simulation
```

---

## Repository Structure

```
├── app.py                          # Streamlit entry point
├── app_pages/                      # One file per dashboard page
├── configs/
│   └── base.yaml                   # All hyperparameters
├── data/
│   ├── raw/                        # Place .psv files here (gitignored)
│   ├── processed/                  # Generated tensors (gitignored)
│   └── splits/                     # Regenerated by create_splits.py
├── scripts/
│   ├── run_pipeline.py             # End-to-end orchestrator
│   ├── run_fl_sim.py               # Automated FL simulation
│   ├── create_splits.py            # Frozen 70/15/15 split
│   ├── create_windowed_dataset.py  # Sliding-window labels
│   ├── cross_validate.py           # 5-fold stratified CV
│   └── optuna_search.py            # Bayesian HPO
├── src/
│   ├── parallel_preprocess.py      # PSV → .pt tensors
│   ├── dataset.py                  # PatientDataset
│   ├── model.py                    # TimeSeriesTransformer
│   ├── model_grud.py               # GRU-D
│   ├── ensemble.py                 # Ensemble blending
│   ├── calibration.py              # Temperature Scaling
│   ├── train_local.py              # Local training loop
│   ├── split_clients.py            # FL data partitioning
│   ├── fl_server.py                # Flower server
│   ├── fl_client.py                # Flower client
│   ├── fl_fedbn.py                 # FedBN strategy
│   ├── evaluate.py                 # Test-set evaluation
│   ├── plot_results.py             # ROC / PRC plots
│   ├── clinical_metrics.py         # Clinical KPIs
│   ├── api.py                      # FastAPI v2
│   ├── config_schema.py            # Pydantic ProjectConfig
│   └── logging_config.py           # Logger factory
├── tests/                          # 75 unit + integration tests
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── MODEL_CARD.md
├── GETTING_STARTED.md
└── requirements.txt
```

---

## Citation

If you use this codebase, please cite the underlying dataset:

```bibtex
@article{reyna2020early,
  title     = {Early Prediction of Sepsis from Clinical Data:
               The PhysioNet/Computing in Cardiology Challenge 2019},
  author    = {Reyna, Matthew A and others},
  journal   = {Critical Care Medicine},
  year      = {2020}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

> **Clinical disclaimer:** This system is a research prototype intended to support — not replace — clinical judgment. It has not been validated in a prospective trial and must not be used for autonomous clinical decision-making.
