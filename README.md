# Neonatal Sepsis Detection

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#)

Research codebase for early sepsis detection in ICU patients using clinical time-series data. Combines a Transformer and a GRU-D model in an ensemble, trained both locally and via Federated Learning (Flower). Evaluation uses a frozen held-out test set with clinical metrics (sensitivity @ specificity, alert fatigue rate, NNAlert) in addition to standard AUROC/AUPRC.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Quick Start — One Command](#quick-start--one-command)
- [Step-by-Step Manual Run](#step-by-step-manual-run)
  - [1. Preprocess raw data](#1-preprocess-raw-data)
  - [2. Create frozen splits](#2-create-frozen-splits)
  - [3. Train local baseline](#3-train-local-baseline)
  - [4. Federated learning simulation](#4-federated-learning-simulation)
  - [5. Evaluate on test set](#5-evaluate-on-test-set)
  - [6. Generate plots](#6-generate-plots)
- [Optional Tools](#optional-tools)
- [Dashboard](#dashboard)
- [API Server](#api-server)
- [Docker](#docker)
- [Running Tests](#running-tests)
- [Configuration](#configuration)
- [Contact](#contact)

---

## Project Overview

| Component | Details |
|---|---|
| **Input** | Per-patient `.psv` files (PhysioNet 2019 format, 40 features × up to 48 hourly timesteps) |
| **Models** | TimeSeriesTransformer · GRU-D · Ensemble blend · Temperature Scaling calibration |
| **Training** | Focal loss, warmup + cosine LR, early stopping, optional MLflow logging |
| **Federated** | Flower (flwr) with FedAvg / FedProx / FedBN strategies |
| **Evaluation** | Frozen 70/15/15 patient split — test set never seen during training |
| **Serving** | FastAPI v2 with MC Dropout CIs, Prometheus `/metrics`, audit log |
| **Dashboard** | 5-page Streamlit app (summary, prediction, metrics, training runs, clinical metrics) |

---

## Repository Structure

```
.
├── app.py                        # Streamlit dashboard launcher
├── app_pages/
│   ├── 1_00_📘_Project_Summary.py
│   ├── 1_03_📈_Predict.py
│   ├── 1_04_🧪_Model_Metrics.py
│   ├── 1_05_📂_Training_Runs.py
│   └── 1_06_🏥_Clinical_Metrics.py
├── configs/
│   └── base.yaml                 # All hyperparameters
├── scripts/
│   ├── run_pipeline.py           # One-command end-to-end orchestrator
│   ├── run_fl_sim.py             # Automated FL simulation
│   ├── create_splits.py          # Create frozen 70/15/15 splits
│   ├── create_windowed_dataset.py# Sliding-window early-warning dataset
│   ├── cross_validate.py         # 5-fold stratified CV
│   └── optuna_search.py          # Bayesian hyperparameter optimisation
├── src/
│   ├── parallel_preprocess.py    # PSV → per-patient .pt tensors
│   ├── dataset.py                # PatientDataset (normalisation, augmentation)
│   ├── model.py                  # TimeSeriesTransformer (CLS token, pad mask)
│   ├── model_grud.py             # GRU-D (per-feature decay, empirical mean)
│   ├── ensemble.py               # Ensemble blending
│   ├── calibration.py            # Temperature Scaling
│   ├── train_local.py            # Local training loop
│   ├── split_clients.py          # Partition patients into FL client folders
│   ├── fl_server.py              # Flower server (FedAvg / FedProx)
│   ├── fl_client.py              # Flower client
│   ├── fl_fedbn.py               # FedBN strategy
│   ├── evaluate.py               # Metrics on frozen test set
│   ├── plot_results.py           # ROC / PRC comparison plots
│   ├── clinical_metrics.py       # Sensitivity@spec, NNAlert, alert fatigue
│   ├── api.py                    # FastAPI v2 inference server
│   ├── config.py                 # Path constants (env-var overrideable)
│   ├── config_schema.py          # Pydantic ProjectConfig
│   ├── logging_config.py         # Shared logger factory
│   └── utils.py                  # ensure_dir helper
├── tests/                        # 75 unit + integration tests
├── data/
│   ├── raw/                      # Raw .psv files (gitignored)
│   ├── processed/                # Generated .pt tensors (gitignored)
│   └── splits/                   # Frozen train/val/test indices (committed)
├── MODEL_CARD.md
├── Makefile
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

**Python 3.11+ required.**

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt
```

> **Windows note:** If `flwr` install fails, try `pip install flwr==1.23.0 --no-deps` then install remaining deps individually.

---

## Dataset

Place raw `.psv` files under `data/raw/`. Each file is one patient, named `p000001.psv` etc. (PhysioNet / CinC Challenge 2019 format).

**File format** — pipe-separated, one row per ICU hour:

```
HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|
Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|
Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|
Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel
```

- 40 feature columns (vitals, labs, demographics) + `SepsisLabel` (0/1)
- Missing values are `NaN` — the pipeline handles them via masking and GRU-D decay
- Keep the header row in every file

---

## Quick Start — One Command

The pipeline orchestrator handles all steps automatically, skipping steps whose outputs already exist:

```bash
python scripts/run_pipeline.py
```

Or with custom settings:

```bash
python scripts/run_pipeline.py \
  --raw_folder data/raw \
  --model transformer \
  --epochs 50 \
  --fl_rounds 20 \
  --n_clients 5
```

**Force flags** — re-run individual steps:

```bash
python scripts/run_pipeline.py --force_preprocess   # re-run preprocessing
python scripts/run_pipeline.py --force_train        # re-run local training
python scripts/run_pipeline.py --force_eval         # re-run evaluation only
python scripts/run_pipeline.py --skip_fl            # skip federated learning
```

Or use the Makefile:

```bash
make pipeline
```

---

## Step-by-Step Manual Run

### 1. Preprocess raw data

Converts each `.psv` file into a per-patient `.pt` tensor containing `X` (features), `mask` (observation mask), `deltas` (time-since-last-observation), `y` (patient label), `y_seq` (per-timestep labels), and `onset_hour`.

```bash
python src/parallel_preprocess.py \
  --raw_folder data/raw \
  --out_folder data/processed/patients \
  --seq_len 48 \
  --nprocs 8
```

Output: `data/processed/patients/` with one `.pt` per patient + `index_with_labels.pt`.

---

### 2. Create frozen splits

Creates a frozen 70% train / 15% val / 15% test patient-level stratified split. Run this **once** — the test set must never be touched during training or model selection.

```bash
python scripts/create_splits.py \
  --index data/processed/patients/index_with_labels.pt \
  --out_dir data/splits
```

Output: `data/splits/train_index.pt`, `val_index.pt`, `test_index.pt`.

---

### 3. Train local baseline

#### Option A — Single training run

```bash
# Transformer
python src/train_local.py \
  --index data/splits/train_index.pt \
  --model transformer \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-4 \
  --patience 10 \
  --run_name local_transformer

# GRU-D
python src/train_local.py \
  --index data/splits/train_index.pt \
  --model grud \
  --epochs 50 \
  --batch_size 64 \
  --run_name local_grud
```

The run directory and best checkpoint path are printed on completion:
```
Run folder : runs/20260613T120000Z__local_transformer/
Best ckpt  : runs/20260613T120000Z__local_transformer/checkpoints/model_best.pt
```

Optional flags:
- `--use_mlflow` — log params and metrics to MLflow (`mlflow ui` to view)
- `--scaler_path data/processed/patients/scaler.json` — normalise features
- `--augment` — enable Gaussian jitter augmentation
- `--use_temperature_scaling` — calibrate output probabilities after training

#### Option B — 5-fold cross-validation

```bash
python scripts/cross_validate.py \
  --index data/splits/train_index.pt \
  --model transformer \
  --epochs 20
```

Reports mean ± std AUROC and AUPRC across folds.

#### Option C — Bayesian hyperparameter search (Optuna)

```bash
python scripts/optuna_search.py \
  --index data/splits/train_index.pt \
  --n_trials 50
```

Searches: `lr`, `d_model`, `n_heads`, `dropout`, `batch_size`. Best params printed and saved.

---

### 4. Federated learning simulation

#### Option A — Automated (recommended)

Launches server and all clients as subprocesses automatically:

```bash
# First, partition patients into FL client folders (test patients excluded)
python src/split_clients.py \
  --processed_folder data/processed/patients \
  --out_root data/processed/clients \
  --n_clients 3 \
  --splits_dir data/splits

# Run FL simulation
python scripts/run_fl_sim.py \
  --client_indexes \
      data/processed/clients/client1/index.pt \
      data/processed/clients/client2/index.pt \
  --model transformer \
  --rounds 20 \
  --local_epochs 1 \
  --n_features 40 \
  --seq_len 48
```

Output: `server_out/global_best.pt` — best federated model.

#### Option B — Manual (separate terminals)

```bash
# Terminal 1 — Server
python src/fl_server.py \
  --model transformer \
  --n_features 40 \
  --seq_len 48 \
  --min_clients 2 \
  --rounds 20

# Terminal 2 — Client 1
python src/fl_client.py \
  --index data/processed/clients/client1/index.pt \
  --model transformer \
  --server_address 127.0.0.1:8080

# Terminal 3 — Client 2
python src/fl_client.py \
  --index data/processed/clients/client2/index.pt \
  --model transformer \
  --server_address 127.0.0.1:8080
```

#### FedBN strategy

```bash
python src/fl_server.py --strategy fedbn --model transformer --rounds 20
```

#### Non-IID (heterogeneous) simulation

```bash
python src/split_clients.py \
  --processed_folder data/processed/patients \
  --out_root data/processed/clients \
  --n_clients 5 \
  --splits_dir data/splits \
  --heterogeneous
```

---

### 5. Evaluate on test set

Always evaluate on the **frozen test split** (`data/splits/test_index.pt`):

```bash
# Federated model
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt server_out/global_best.pt \
  --model transformer \
  --n_features 40 \
  --seq_len 48 \
  --out_file eval_results_federated.json

# Local baseline (replace the path with your run's checkpoint)
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt runs/20260613T120000Z__local_transformer/checkpoints/model_best.pt \
  --model transformer \
  --n_features 40 \
  --seq_len 48 \
  --out_file eval_results_local.json
```

Outputs AUROC, AUPRC, precision, recall, F1, calibration, and 95% bootstrap CIs.

---

### 6. Generate plots

```bash
python src/plot_results.py \
  --results eval_results_federated.json eval_results_local.json \
  --out_file model_comparison_plot.png
```

Produces ROC and PRC comparison plots with bootstrap confidence intervals.

---

## Optional Tools

### Early-warning sliding-window dataset

Creates one window per ICU hour with a forward-looking label ("will sepsis onset within the next 6 hours?"):

```bash
python scripts/create_windowed_dataset.py \
  --index data/processed/patients/index_with_labels.pt \
  --out_dir data/processed/windowed \
  --horizon 6 \
  --stride 1
```

### Clinical metrics

```python
from src.clinical_metrics import compute_all
import json

results = json.load(open("eval_results_federated.json"))
metrics = compute_all(results["y_true"], results["y_prob"], threshold=0.5)
print(metrics)
```

---

## Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Pages:

| Page | Description |
|---|---|
| Project Summary | Architecture overview, dataset stats |
| Predict | Single-patient inference (manual input or file upload) |
| Model Metrics | ROC/PRC curves, confusion matrix, calibration |
| Training Runs | Browse `runs/` artifacts, compare metrics |
| Clinical Metrics | Sensitivity @ specificity, alert fatigue, NNAlert |

---

## API Server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

| Endpoint | Description |
|---|---|
| `POST /v2/predict` | Inference — returns probability + optional MC Dropout CIs |
| `GET /health` | Uptime and model version |
| `GET /metrics` | Prometheus metrics |

Set environment variables to configure paths:

```bash
SEPSIS_MODEL_PATH=server_out/global_best.pt
SEPSIS_SCALER_PATH=data/processed/patients/scaler.json
```

---

## Docker

```bash
# Build
docker build -t neonatal-sepsis .

# Run API
docker run -p 8000:8000 \
  -v $(pwd)/server_out:/app/server_out:ro \
  neonatal-sepsis

# Or with docker-compose (API + dashboard)
docker-compose up
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

75 tests covering preprocessing, dataset loading, model forward passes, training loop, evaluation metrics, split correctness, and windowed dataset creation.

```bash
# Run a specific test file
python -m pytest tests/test_model.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Configuration

All hyperparameters are in `configs/base.yaml`. Load in code with:

```python
from src.config_schema import load_config
cfg = load_config("configs/base.yaml")
print(cfg.training.lr)        # 0.0001
print(cfg.federated.rounds)   # 20
```

Key sections:

```yaml
data:
  seq_len: 48        # timesteps per patient window
  n_features: 40     # input feature count

model:
  name: transformer  # transformer | grud
  d_model: 128
  n_heads: 4

training:
  epochs: 50
  batch_size: 64
  lr: 1.0e-4
  use_focal: false   # focal loss for class imbalance
  augment: true      # Gaussian jitter augmentation

federated:
  rounds: 20
  n_clients: 5
  strategy: fedavg   # fedavg | fedbn
  mu: 0.01           # FedProx proximal term (0 = plain FedAvg)
```

---

## Contact

- Maintainer: [`pranay9981`](https://github.com/pranay9981)
- Collaborators: [`NinadAmane`](https://github.com/NinadAmane), [`Rakshak05`](https://github.com/Rakshak05)
