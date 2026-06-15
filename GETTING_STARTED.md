# Getting Started — From Scratch

This guide walks you through running the **Neonatal Sepsis Detection** project end-to-end on a new machine, in order. No prior familiarity with the codebase required.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | `python --version` to check |
| Git | any | to clone the repo |
| RAM | 8 GB+ | 16 GB recommended for large datasets |
| CPU cores | 4+ | preprocessing uses multiprocessing |
| Disk space | ~10 GB | raw PSV files + processed tensors |

---

## Step 0 — Clone the Repository

```bash
git clone https://github.com/pranay9981/Neonatal-Sepsis.git
cd "Neonatal-Sepsis"

# Switch to the improvements branch (has all the new features)
git checkout improvements/v2
```

---

## Step 1 — Create a Virtual Environment

```bash
# Create the venv
python -m venv .venv

# Activate — Windows PowerShell
.venv\Scripts\Activate.ps1

# Activate — macOS / Linux
source .venv/bin/activate
```

Your prompt should now show `(.venv)` at the start.

---

## Step 2 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows note:** If `flwr` fails to install, run:
> ```bash
> pip install flwr==1.23.0 --no-deps
> pip install -r requirements.txt --ignore-installed flwr
> ```

Verify the key packages installed correctly:

```bash
python -c "import torch, flwr, streamlit, fastapi, mlflow, optuna; print('All packages OK')"
```

---

## Step 3 — Verify the Test Suite

Before touching any data, confirm the codebase is healthy:

```bash
python -m pytest tests/ -v
```

Expected: **75 tests passing, 0 failures.** If any fail, do not proceed — something is wrong with the environment.

---

## Step 4 — Get the Data

The project uses the **PhysioNet Computing in Cardiology Challenge 2019** dataset.

### Download

1. Go to: https://physionet.org/content/challenge-2019/1.0.0/
2. Create a free PhysioNet account if you do not have one.
3. Download **training set A** (the primary dataset, ~40,000 patients).
4. Extract the archive — you will get a folder of `.psv` files, one per patient.

### Place the files

```
Neonatal-Sepsis/
└── data/
    └── raw/
        ├── p000001.psv
        ├── p000002.psv
        └── ...  (all .psv files go here)
```

```bash
# On Windows — move or copy your extracted folder contents into data/raw/
# Example (adjust the source path):
xcopy "C:\Downloads\training_setA\*" "data\raw\" /E /Y
```

Verify a few files are present:

```bash
# Should print at least a few .psv filenames
ls data/raw/ | head -5        # macOS/Linux
Get-ChildItem data\raw\ | Select-Object -First 5  # Windows PowerShell
```

Each file looks like this (pipe-separated, one row per ICU hour):

```
HR|O2Sat|Temp|SBP|...|SepsisLabel
85|98|37.1|120|...|0
88|97|37.2|118|...|0
...
```

---

## Step 5 — Run the Full Pipeline (One Command)

The easiest way to run everything is the pipeline orchestrator. It runs all 7 steps in order and **skips steps whose outputs already exist**, so it is safe to re-run.

```bash
python scripts/run_pipeline.py
```

This runs with sensible defaults (Transformer, 10 epochs, 5 FL rounds, 3 clients). For a fuller run closer to the paper settings:

```bash
python scripts/run_pipeline.py \
  --model transformer \
  --epochs 50 \
  --fl_rounds 20 \
  --n_clients 5
```

Watch the terminal — it will print each step as it starts:

```
============================================================
[PIPELINE] Step 1/7 — Preprocessing raw PSV data
...
[PIPELINE] Step 2/7 — Creating frozen 70/15/15 patient splits
...
[PIPELINE] Step 7/7 — Generating ROC/PRC comparison plots
============================================================
[PIPELINE] All steps complete!
```

**Skip to Step 7** if you want to understand what each step does individually.

---

## Step 6 — Launch the Dashboard

Once the pipeline finishes:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`. You will see 5 pages:

| Page | What it shows |
|---|---|
| **Project Summary** | Architecture overview, dataset stats, feature list |
| **Predict** | Run inference on a single patient (manual input or file upload) |
| **Model Metrics** | Interactive ROC/PRC curves with 95% bootstrap CI, confusion matrix |
| **Training Runs** | Browse all runs in `runs/`, compare AUROC/AUPRC across experiments |
| **Clinical Metrics** | Sensitivity @ specificity, alert fatigue rate, NNAlert — clinical framing |

---

## Step 7 — What the Pipeline Steps Do (Manual Run)

If you prefer to run each step individually (for debugging, custom flags, or partial reruns):

### 7.1 — Preprocess raw data

Converts every `.psv` file into a per-patient `.pt` tensor with:
`X` (features) · `mask` (observation mask) · `deltas` (time-since-last-obs) · `y` (sepsis label) · `y_seq` (per-timestep labels) · `onset_hour`

```bash
python src/parallel_preprocess.py \
  --raw_folder data/raw \
  --out_folder data/processed/patients \
  --seq_len 48 \
  --nprocs 8
```

Output: `data/processed/patients/` — one `.pt` per patient + `index_with_labels.pt`.

---

### 7.2 — Create frozen splits

Splits patients into 70% train / 15% val / 15% test (stratified by label). This is done **once** and never changes — the test set is frozen and must never be seen during training.

```bash
python scripts/create_splits.py \
  --index data/processed/patients/index_with_labels.pt \
  --out_dir data/splits
```

Output: `data/splits/train_index.pt`, `val_index.pt`, `test_index.pt`.

> The `data/splits/` files are already committed in the repo for the full 40,323-patient dataset. You only need to re-run this if you use a different dataset.

---

### 7.3 — Train a local baseline

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
  --run_name local_grud
```

Useful extra flags:

| Flag | Effect |
|---|---|
| `--use_mlflow` | Log to MLflow — view with `mlflow ui` |
| `--scaler_path data/processed/patients/scaler.json` | Enable feature normalisation |
| `--augment` | Gaussian jitter augmentation during training |
| `--use_temperature_scaling` | Calibrate output probabilities after training |

The best checkpoint is saved at:
```
runs/<timestamp>__<run_name>/checkpoints/model_best.pt
```

---

### 7.4 — Federated learning simulation

#### Split patients into client folders first

```bash
python src/split_clients.py \
  --processed_folder data/processed/patients \
  --out_root data/processed/clients \
  --n_clients 3 \
  --splits_dir data/splits
```

Add `--heterogeneous` to simulate non-IID data across hospitals (different class distributions per client).

#### Run the FL simulation — FedAvg (automated — recommended)

```bash
python scripts/run_fl_sim.py --client_indexes data/processed/clients/client1/index.pt data/processed/clients/client2/index.pt data/processed/clients/client3/index.pt data/processed/clients/client4/index.pt data/processed/clients/client5/index.pt --model grud --rounds 10 --strategy fedavg --save_dir server_out --checkpoints_dir checkpoints
```

This starts the Flower server and all clients as subprocesses automatically. Output: `server_out/global_best.pt`.

> **Windows PowerShell:** Pass all `--client_indexes` paths on a **single line**. A line break without a trailing backtick (`` ` ``) ends the command early and only the paths before the break are parsed — you end up with fewer clients than intended.

#### Run the FL simulation — FedBN

FedBN keeps BatchNorm statistics local to each client and only averages non-BN weights. Use a different `--save_dir` to avoid overwriting FedAvg results:

```bash
python scripts/run_fl_sim.py --client_indexes data/processed/clients/client1/index.pt data/processed/clients/client2/index.pt data/processed/clients/client3/index.pt data/processed/clients/client4/index.pt data/processed/clients/client5/index.pt --model grud --rounds 10 --strategy fedbn --save_dir server_out_fedbn --checkpoints_dir checkpoints_fedbn
```

Output: `server_out_fedbn/global_best.pt`.

#### Run the FL simulation — Non-IID (heterogeneous hospital data)

In real deployments, hospitals have different patient populations — some see far more sepsis cases than others. The `--heterogeneous` flag simulates this by skewing the class distribution across clients. Always use a separate `--out_root` and `--save_dir` to keep results independent from the IID run.

```bash
# Step 1 — Create non-IID splits (positives concentrated in early clients)
python src/split_clients.py \
  --processed_folder data/processed/patients \
  --out_root data/processed/clients_noniid \
  --n_clients 5 \
  --splits_dir data/splits \
  --heterogeneous
```

You will see skewed per-client positive rates printed (e.g. client 1 ~high %, client 5 ~low %).

```bash
# Step 2 — Run FL simulation
python scripts/run_fl_sim.py --client_indexes data/processed/clients_noniid/client1/index.pt data/processed/clients_noniid/client2/index.pt data/processed/clients_noniid/client3/index.pt data/processed/clients_noniid/client4/index.pt data/processed/clients_noniid/client5/index.pt --model grud --rounds 10 --strategy fedavg --save_dir server_out_noniid --checkpoints_dir checkpoints_noniid
```

Output: `server_out_noniid/global_best.pt`.

```bash
# Step 3 — Evaluate
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt server_out_noniid/global_best.pt \
  --model grud \
  --out_file eval_results_noniid.json
```

Expected result: **AUROC ~0.83** (vs ~0.92 for IID FedAvg). The gap is the non-IID penalty — skewed distributions cause gradient divergence during aggregation. The loss curve will also be more volatile (oscillating rather than monotonically decreasing).

#### Run manually (separate terminals)

```bash
# Terminal 1 — Server
python src/fl_server.py --model grud --rounds 10 --min_clients 5 --strategy fedavg

# Terminal 2 — Client 1
python src/fl_client.py \
  --index data/processed/clients/client1/index.pt \
  --model grud \
  --server_address 127.0.0.1:8080

# (Repeat for clients 2–5 in separate terminals)
```

---

### 7.5 — Evaluate on the frozen test set

Run `src/evaluate.py` once per model. The `--model` flag must match the architecture of the checkpoint you are evaluating.

```bash
# Federated GRU-D — FedAvg
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt server_out/global_best.pt \
  --model grud \
  --out_file eval_results_federated.json

# Federated GRU-D — FedBN (if you ran the FedBN simulation)
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt server_out_fedbn/global_best.pt \
  --model grud \
  --out_file eval_results_fedbn.json

# Federated GRU-D — non-IID (if you ran the non-IID simulation)
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt server_out_noniid/global_best.pt \
  --model grud \
  --out_file eval_results_noniid.json

# GRU-D local baseline
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt runs/<your-grud-run>/checkpoints/model_best.pt \
  --model grud \
  --out_file eval_results_grud.json

# Transformer local baseline
python src/evaluate.py \
  --index data/splits/test_index.pt \
  --ckpt runs/<your-transformer-run>/checkpoints/model_best.pt \
  --model transformer \
  --out_file eval_results_transformer.json
```

Each output JSON contains: `auroc`, `auprc`, `precision`, `recall`, `f1`, `threshold`, `y_true`, `y_prob`, and 95% bootstrap CIs.

Once the individual eval JSONs exist, the **Model Metrics** dashboard page auto-discovers them and displays interactive ROC/PRC curves for all four models side by side.

---

### 7.5b — Evaluate the Ensemble

The Ensemble blends Transformer + GRU-D probabilities (weighted average). Both local checkpoints must exist before running this.

```bash
python scripts/eval_ensemble.py \
  --index data/splits/test_index.pt \
  --transformer_ckpt runs/<your-transformer-run>/checkpoints/model_best.pt \
  --grud_ckpt runs/<your-grud-run>/checkpoints/model_best.pt \
  --out_file eval_results_ensemble.json \
  --alpha 0.5
```

| Flag | Effect |
|---|---|
| `--alpha` | Transformer weight in the blend (0.5 = equal; lower = more GRU-D weight) |
| `--device` | `cpu` or `cuda` |

Output: `eval_results_ensemble.json` — same format as single-model evals. The dashboard picks it up automatically.

---

### 7.6 — Generate comparison plots

```bash
python src/plot_results.py \
  --results eval_results_federated.json eval_results_local.json \
  --out_file model_comparison_plot.png
```

Produces side-by-side ROC and PRC curves with bootstrap confidence bands.

---

## Optional Experiments

### 5-fold cross-validation

```bash
# GRU-D (best single model — AUROC 0.9192 ± 0.0044 on 5 folds)
python scripts/cross_validate.py \
  --index data/splits/train_index.pt \
  --model grud \
  --epochs 20 \
  --out_file cv_results.json

# Transformer
python scripts/cross_validate.py \
  --index data/splits/train_index.pt \
  --model transformer \
  --epochs 20 \
  --out_file cv_results_transformer.json
```

Reports mean ± std AUROC and AUPRC across 5 folds with 95% bootstrap CIs. Results are saved to the specified `--out_file`.

---

### Bayesian hyperparameter search (Optuna)

```bash
# Quick search — 20 trials, 5 epochs per trial (~1-2 hrs)
python scripts/optuna_search.py \
  --model grud \
  --index data/splits/train_index.pt \
  --n_trials 20 \
  --epochs 5 \
  --out_file optuna_results_grud.json

# Full search — 50 trials, more epochs (~4-6 hrs)
python scripts/optuna_search.py \
  --model grud \
  --index data/splits/train_index.pt \
  --n_trials 50 \
  --epochs 20 \
  --out_file optuna_results_grud.json
```

Searches `lr`, `hidden_size`, `dropout`, `batch_size` using TPE sampler + median pruner. Best params are printed and saved to `--out_file`.

Our best params (20 trials, 5 epochs): `lr=0.000233`, `batch_size=32`, `hidden_size=256`, `dropout=0.110` — best val AUROC 0.9225.

---

### Early-warning sliding-window dataset

Creates one sample per ICU hour with a forward-looking label — "will sepsis onset within the next 6 hours?":

```bash
python scripts/create_windowed_dataset.py \
  --index data/processed/patients/index_with_labels.pt \
  --out_dir data/processed/windowed \
  --horizon 6 \
  --stride 1
```

---

### Compute clinical metrics in code

```python
import json
from src.clinical_metrics import compute_all

results = json.load(open("eval_results_federated.json"))
metrics = compute_all(results["y_true"], results["y_prob"], threshold=0.5)
# Returns: sensitivity@spec, specificity@sens, alert_fatigue_rate, nn_alert
print(metrics)
```

---

## API Server

The FastAPI server loads one model checkpoint and serves it over HTTP. You must tell it **which checkpoint** and **which architecture** via environment variables before starting.

### Environment variables

| Variable | Values | Default | Notes |
|---|---|---|---|
| `SEPSIS_MODEL_PATH` | path to `.pt` file | `server_out/global_best.pt` | The checkpoint to load |
| `SEPSIS_MODEL_TYPE` | `grud` \| `transformer` | `grud` | Must match the checkpoint's architecture |
| `SEPSIS_MC_SAMPLES` | integer | `0` | MC Dropout passes for confidence intervals (0 = disabled) |
| `SEPSIS_SCALER_PATH` | path to `scaler.json` | — | Feature normalisation — recommended |

### Start the server

```bash
# Serve the federated GRU-D model
export SEPSIS_MODEL_PATH=server_out/global_best.pt
export SEPSIS_MODEL_TYPE=grud
export SEPSIS_SCALER_PATH=data/processed/patients/scaler.json
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Serve the FedBN GRU-D model instead
export SEPSIS_MODEL_PATH=server_out_fedbn/global_best.pt
export SEPSIS_MODEL_TYPE=grud
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Serve the local Transformer
export SEPSIS_MODEL_PATH=runs/<your-transformer-run>/checkpoints/model_best.pt
export SEPSIS_MODEL_TYPE=transformer
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

> **Windows PowerShell:** Use `$env:SEPSIS_MODEL_PATH = "server_out/global_best.pt"` instead of `export`.

### Test the server

```bash
# Health check
curl http://localhost:8000/health

# Predict (GRU-D — data, mask, and deltas all required for full accuracy)
curl -X POST http://localhost:8000/v2/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[85, 98, 37.1, 120, 80, 65, 18, 0, 0, 24, 0, 7.4, 40, 98, 0, 15, 70, 9, 102, 0.9, 0, 110, 2.1, 0.8, 3.5, 4.2, 1.2, 0, 42, 14, 28, 8, 250, 200, 65, 1, 0, 0, -6, 3]],
    "mask": [[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]],
    "deltas": [[0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 6, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]]
  }'
```

### Enable MC Dropout confidence intervals

```bash
export SEPSIS_MC_SAMPLES=50
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
# The /v2/predict response will now include ci_lower and ci_upper
```

Key endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/v2/predict` | POST | Returns sepsis probability + optional MC Dropout CIs |
| `/health` | GET | Model version and server uptime |
| `/metrics` | GET | Prometheus metrics |

---

## Docker

```bash
# Build the image (~3-5 min on first build; subsequent builds use cached layers)
docker build -t neonatal-sepsis .

# Run the API server only
docker run --rm -p 8000:8000 \
  -v ${PWD}/server_out:/app/server_out \
  -e SEPSIS_MODEL_PATH=/app/server_out/global_best.pt \
  -e SEPSIS_MODEL_TYPE=grud \
  neonatal-sepsis uvicorn src.api:app --host 0.0.0.0 --port 8000

# Test it (in a second terminal)
curl http://localhost:8000/health

# Run API + dashboard together (reads docker-compose.yml)
docker-compose up
```

> **Windows PowerShell:** Use `${PWD}` for the volume mount path (not `$(pwd)`).

The image uses a multi-stage build and runs as a non-root user (uid=1000). A `.dockerignore` excludes `data/`, `runs/`, `.venv/`, and other large local directories — keeping the build context under 1 MB.

---

## Re-running Specific Steps

Force flags let you re-run a single step without redoing everything:

```bash
python scripts/run_pipeline.py --force_preprocess   # redo step 1
python scripts/run_pipeline.py --force_splits       # redo step 2
python scripts/run_pipeline.py --force_train        # redo step 3
python scripts/run_pipeline.py --force_fl           # redo step 5
python scripts/run_pipeline.py --force_eval         # redo step 6
python scripts/run_pipeline.py --skip_fl            # run everything except FL
python scripts/run_pipeline.py --skip_local_train   # run everything except local training
```

---

## File Outputs Reference

After a full pipeline run, you will have:

```
data/
  processed/patients/       .pt tensors + scaler.json + index_with_labels.pt
  splits/                   train_index.pt  val_index.pt  test_index.pt
  processed/clients/        client1/ … clientN/ index.pt files

runs/
  <timestamp>__<name>/
    checkpoints/model_best.pt
    logs/

server_out/                 FedAvg IID results
  global_best.pt            best federated model (FedAvg, IID)
  checkpoints/              per-round checkpoints

server_out_fedbn/           FedBN IID results
  global_best.pt            best federated model (FedBN)
  checkpoints_fedbn/        per-round FedBN checkpoints

server_out_noniid/          FedAvg non-IID results
  global_best.pt            best federated model (FedAvg, non-IID)
  checkpoints_noniid/       per-round non-IID checkpoints

eval_results_federated.json      FedAvg IID GRU-D test-set results  (AUROC 0.9238)
eval_results_fedbn.json          FedBN IID GRU-D test-set results   (AUROC 0.9051)
eval_results_noniid.json         FedAvg non-IID GRU-D results       (AUROC 0.8306)
eval_results_grud.json           GRU-D local test-set results       (AUROC 0.9189)
eval_results_transformer.json    Transformer local test-set results  (AUROC 0.9092)
eval_results_ensemble.json       Ensemble (Transformer+GRU-D)       (AUROC 0.9293)
model_comparison_plot.png
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run commands from the project root, not from inside `src/` |
| `flwr` install fails on Windows | `pip install flwr==1.23.0 --no-deps` then install rest separately |
| Torch watcher `RuntimeError` spam | Already fixed — `.streamlit/config.toml` sets `fileWatcherType = "none"` |
| Dashboard shows blank cards | Ensure you are on `improvements/v2` — old master branch has this bug |
| `index_with_labels.pt not found` | Run preprocessing (step 7.1) first |
| FL server exits immediately | Check `--min_clients` matches the number of client processes you start |
| FL sim starts with fewer clients than expected (PowerShell) | All `--client_indexes` paths must be on a **single line**. A line break without a trailing backtick ends the command — paths on later lines are silently dropped |
| FedBN run overwrites FedAvg results | Pass `--save_dir server_out_fedbn --checkpoints_dir checkpoints_fedbn` to keep them separate |
| `eval_ensemble.py` — model not found | Both `--transformer_ckpt` and `--grud_ckpt` paths must point to the `model_best.pt` inside the run's `checkpoints/` folder |
| OOM during preprocessing | Reduce `--nprocs` (e.g. `--nprocs 2`) |
| Tests fail | Verify venv is activated and `pip install -r requirements.txt` completed without errors |
| Non-IID AUROC is ~0.09 lower than IID | Expected — skewed client distributions cause gradient divergence. Try more rounds or stronger FedProx `--mu`. |
| Docker build context is large (several GB) | Ensure `.dockerignore` is present in the project root — it excludes `data/`, `runs/`, `.venv/` etc. |
| `docker run` — permission denied on uvicorn | Packages must be installed into the non-root user's home. Rebuild after pulling the latest Dockerfile. |
