# Neonatal-Sepsis

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](#)  

**Neonatal-Sepsis** — research codebase for modelling and evaluating time-series models for neonatal sepsis detection. Includes preprocessing utilities, local baselines (Transformer, GRU-D for missingness), federated learning simulation (server + clients), and a secure aggregation PoC. Use this repository to run local experiments, simulate federated training, and compare evaluation metrics (AUROC, AUPRC, precision/recall, etc.).

---

## Table of contents

- [Key features](#key-features)  
- [Repository structure](#repository-structure)  
- [Requirements](#requirements)  
- [Dataset & expected format (sample included)](#dataset--expected-format-sample-included)  
- [Quickstart](#quickstart)  
  - [1. Create & activate venv](#1-create--activate-venv)  
  - [2. Preprocess raw data (parallel)](#2-preprocess-raw-data-parallel)  
  - [3. (Optional) Pack into LMDB](#3-optional-pack-into-lmdb)  
  - [4. Train local baseline (Transformer)](#4-train-local-baseline-transformer)  
  - [5. Train GRU-D (missing-data aware)](#5-train-gru-d-missing-data-aware)  
  - [Federated simulation](#federated-simulation)  
  - [Evaluate & plot results](#evaluate--plot-results)  
- [Evaluation & artifacts](#evaluation--artifacts)  
- [Development notes & tips](#development-notes--tips)  
- [Contributing](#contributing)
- [Contact](#contact)    


---

## Key features

- Preprocessing pipeline converting raw per-timestep `.psv` (pipe-separated) data to per-patient `.pt` objects.  
- Local training: Transformer baseline and GRU-D (handles missingness).  
- Simple hyperparameter search utilities.  
- Federated learning simulation (server + multiple clients).  
- Secure aggregation proof-of-concept (masking demonstration).  
- Evaluation & plotting tools producing JSON summaries and comparative plots.

---

## Repository structure

```
Neonatal-Sepsis/
├─ app.py
├─ dashboard.py
├─ README.md
├─ requirements.txt
├─ eval_results_federated.json
├─ eval_results_local.json
├─ model_comparison_plot.png
├─ model_comparison_plot_prc.png
├─ app_pages/                
│  ├─ 1_00_📘_Project_Summary
│  ├─ 1_03_📈_Predict
│  └─ 1_04_🧪_Model_Metrics
└─ src/
   ├─ parallel_preprocess.py
   ├─ lmdb_packer.py
   ├─ model.py
   ├─ model_grud.py
   ├─ train_local.py
   ├─ hyperparam_search.py
   ├─ split_clients.py
   ├─ fl_server.py
   ├─ fl_client.py
   ├─ secure_agg_poc.py
   ├─ evaluate.py
   └─ plot_results.py
```

> Note: If your repo uses a different static/templates path, either move your frontend files into `app_pages/` or update `app.py`/`dashboard.py` to point to the actual path.

---

## Requirements

- Python 3.8+ (recommended 3.8–3.11)  
- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset & expected format (sample included)

**Format**: pipe (`|`) separated values (`.psv`). Each row represents a timepoint for an encounter (typically hourly). Preprocessing converts raw `.psv` per-encounter files into per-patient Torch `.pt` objects used by training/federation/evaluation.

**Important columns**:
- Vital signs / labs:  
  `HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets`
- Demographics / metadata: `Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS`
- Label: `SepsisLabel` — binary (0/1) per timepoint

**Notes**:
- Missing values are `NaN`. Preprocessing builds masks and time-since-last-observation features (for GRU-D).
- Keep the header row intact for each raw `.psv` file.
- Place raw `.psv` files under `data/raw/` (or pass a different folder to the preprocessing script).
- Example filename: `data/raw/sample_patient.psv`

---

## Quickstart

### 1. Create & activate venv
(see Requirements above)

### 2. Preprocess raw `.psv` files into per-patient `.pt` (parallel)

```bash
python src/parallel_preprocess.py \
  --raw_folder data/raw \
  --out_folder data/processed/patients \
  --seq_len 48 \
  --nprocs 8
```

- `--seq_len`: number of timesteps (e.g., 48).  
- `--nprocs`: number of parallel workers.

### 3. (Optional) Pack into LMDB (recommended for large datasets)

```bash
python src/lmdb_packer.py \
  --in_folder data/processed/patients \
  --out_folder data/processed/lmdb_shards \
  --shard_size 4000
```

### 4. Train local baseline (Transformer)

```bash
python src/train_local.py \
  --index data/processed/patients/index_with_labels.pt \
  --epochs 10 \
  --batch_size 64 \
  --model transformer
```

Training prints the run directory and best checkpoint path (e.g., `runs/.../checkpoints/model_best.pt`).

### 5. Train GRU-D (missing-data aware)

```bash
python src/train_local.py \
  --index data/processed/patients/index_with_labels.pt \
  --epochs 10 \
  --batch_size 64 \
  --model grud
```

### 6. Hyperparameter quick grid

```bash
python src/hyperparam_search.py
```

---

## Federated simulation

1. Split processed patients into client folders:

```bash
python src/split_clients.py \
  --processed_folder data/processed/patients \
  --out_root data/processed/clients \
  --n_clients 3
```

2. Start the federated server (terminal 1):

```bash
python src/fl_server.py \
  --model transformer \
  --n_features 40 \
  --seq_len 48 \
  --min_clients 2 \
  --rounds 5
```

3. Start each client (one terminal per client):

```bash
python src/fl_client.py \
  --index data/processed/clients/client1/index.pt \
  --model transformer \
  --server_address 127.0.0.1:8080
```

Repeat for `client2` / `client3`.

### Secure aggregation PoC

```bash
python src/secure_agg_poc.py
```

Runs a local proof-of-concept demonstrating additive mask cancellation so the server only observes aggregated updates.

---

## Evaluate & plot results

1. Evaluate federated global checkpoint on a held-out client:

```bash
python src/evaluate.py \
  --index data/processed/clients/client3/index.pt \
  --ckpt server_out/global_best.pt \
  --model transformer \
  --n_features 40 \
  --seq_len 48 \
  --out_file eval_results_federated.json
```

2. Evaluate a local-only model:

```bash
python src/evaluate.py \
  --index data/processed/clients/client3/index.pt \
  --ckpt runs/<your_local_run>/checkpoints/model_best.pt \
  --model transformer \
  --n_features 40 \
  --seq_len 48 \
  --out_file eval_results_local.json
```

3. Generate comparison plot:

```bash
python src/plot_results.py \
  --results eval_results_federated.json eval_results_local.json \
  --out_file model_comparison_plot.png
```

---

## Evaluation & artifacts

Outputs produced by scripts:
- JSON evaluation summaries (AUROC, AUPRC, precision/recall, thresholds).  
- PNG comparison plots: `model_comparison_plot.png`, `model_comparison_plot_prc.png`.  
- Checkpoints and training logs in `runs/` or `server_out/`.

---

## Development notes & tips

- Use LMDB when I/O becomes the bottleneck.  
- Pin dependency versions in `requirements.txt` for reproducibility.  
- Save CLI args and random seeds for reproducible experiments.  
- Federated simulation uses local networking — ensure matching ports and separate terminals or tmux panes.  
- Inspect `src/parallel_preprocess.py` to see how `NaN` is handled and how masks/time deltas are created for GRU-D.

---

## Contributing

Contributions welcome. Suggested improvements:
- Add `data/README.md` describing dataset schema and a small mock `.psv` to help new users run the pipeline end-to-end.  
- Add unit tests for preprocessing, loaders, and training.  
- Add Dockerfile / docker-compose for reproducible local federation experiments.

When opening PRs:
1. Explain the change & rationale.  
2. Include runnable examples or tests.  
3. Keep changes small and focused.

---

## Contact

- Maintainer: [`pranay9981`](https://github.com/pranay9981)
- Collaborators: [`NinadAmane`](https://github.com/NinadAmane), [`Rakshak05`](https://github.com/Rakshak05)  

---

