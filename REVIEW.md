---
phase: deep-audit
reviewed: 2026-06-17
depth: deep
files_reviewed: 46
files_reviewed_list:
  - app.py
  - app_pages/1_00_📘_Project_Summary.py
  - app_pages/1_03_📈_Predict.py
  - app_pages/1_04_🧪_Model_Metrics.py
  - app_pages/1_05_📂_Training_Runs.py
  - app_pages/1_06_🏥_Clinical_Metrics.py
  - src/api.py
  - src/calibration.py
  - src/clinical_metrics.py
  - src/config.py
  - src/config_schema.py
  - src/dataset.py
  - src/ensemble.py
  - src/evaluate.py
  - src/fl_client.py
  - src/fl_fedbn.py
  - src/fl_server.py
  - src/logging_config.py
  - src/model.py
  - src/model_grud.py
  - src/parallel_preprocess.py
  - src/plot_results.py
  - src/split_clients.py
  - src/train_local.py
  - src/utils.py
  - scripts/create_splits.py
  - scripts/create_windowed_dataset.py
  - scripts/cross_validate.py
  - scripts/eval_ensemble.py
  - scripts/optuna_search.py
  - scripts/run_fl_sim.py
  - scripts/run_pipeline.py
  - tests/test_dataset.py
  - tests/test_evaluate.py
  - tests/test_model.py
  - tests/test_preprocess.py
  - tests/test_splits.py
  - tests/test_train.py
  - tests/test_windowed_dataset.py
  - .github/workflows/ci.yml
  - Dockerfile
  - docker-compose.yml
  - requirements.txt
  - .gitignore
  - .dockerignore
  - configs/base.yaml
commit: dc18d34
branch: improvements/v3
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
status: clean
---

# Deep Code Review — Neonatal Sepsis Detection Project

**Reviewed:** 2026-06-17
**Commit:** dc18d34 (improvements/v3)
**Depth:** deep
**Files Reviewed:** 46
**Status:** clean — no findings

## Summary

All 40 findings from the prior deep audit (Session 18) have been verified fixed at commit
`dc18d34`. The codebase is clean at this depth. The architecture is sound: federated learning
with FedBN/FedProx, GRU-D and Transformer models, temperature-scaled calibration, PHI-safe
audit logging, and a frozen 70/15/15 split with train-only scaler.

## Verified Fixes (previously 14 Critical, 18 Warning, 8 Info)

### Critical — all resolved

| ID | File | Issue | Verification |
|----|------|-------|-------------|
| CR-01 | `src/fl_fedbn.py:145` | FedBN NameError (`chosen_key` undefined) | Log line uses `server_round, val` only |
| CR-02 | `src/api.py:244-247` | MC Dropout leaves model in `train()` on exception | `try/finally: _model.eval()` present |
| CR-03 | `src/train_local.py`, `src/fl_client.py` | GRU-D `x_mean` not scaled to z-score space | `(x_mean - scaler_mean) / scaler_std` applied before `set_empirical_mean` |
| CR-04 | `scripts/run_pipeline.py:223,240` | `--scaler_path` never passed to evaluate steps | Both eval subprocess calls include `--scaler_path` |
| CR-05 | `src/calibration.py:37` | LBFGS `lr=1.0` overshoots temperature scaling | Default `lr=0.01` |
| CR-06 | `src/fl_fedbn.py:99-101` | FedBN weighted average sums raw arrays | `np.zeros_like` + `+= w[i] * (n / total)` |
| CR-07 | `src/fl_client.py` | FedProx ordering contract undocumented | Comment + assert on `w0` capture ordering |
| CR-08 | `src/api.py`, `.dockerignore` | Raw `n_timesteps` (PHI) in audit log; 16-char SHA-256 | `_bucket_timesteps()` used; full 64-char digest; `*.jsonl` in `.dockerignore` |
| CR-09 | `scripts/run_pipeline.py:135` | No guard before scaler recompute without `train_index.pt` | `sys.exit(1)` if `train_index` missing |
| CR-10 | `src/evaluate.py` | No warning when non-test index passed to evaluate | Warning logged if `"test"` not in index filename |
| CR-11 | `src/api.py:52,121-128,253` | Temperature never loaded in production inference | `_temperature` loaded from `threshold.json`; `logit / _temperature` before sigmoid |
| CR-12 | `src/fl_fedbn.py:115` | `aggregate_fit` returns empty `{}` metrics | Returns `{"round": server_round}` |
| CR-13 | `src/api.py:60,164-170` | No input size limit — DoS via huge requests | `MAX_ROWS = 1000`; rejected in `check_shape` validator |
| CR-14 | `app_pages/1_03_📈_Predict.py` | Scaler applied to zero-padding rows | Scaler applied only to `data_np[pad_rows:]` |

### Warning — all resolved

| ID | File | Issue | Verification |
|----|------|-------|-------------|
| WR-01 | `src/fl_fedbn.py:39-40` | FedBN BN_KEYWORDS missed LayerNorm | `"layer_norm"`, `"layernorm"` added |
| WR-02 | `src/train_local.py` | `ts` variable shadowed (timestamp vs TemperatureScaler) | Renamed to `temp_scaler` |
| WR-03 | `src/calibration.py` | `load()` accepts non-positive temperature silently | Range check; resets to 1.0 with warning |
| WR-04 | `src/model_grud.py` | Recurrent dropout inside GRU timestep loop | `h = self.dropout(h)` removed from loop |
| WR-05 | `src/evaluate.py` | Third-pass smart-map may resample wrong weight as pos_emb | Name-based guard: only keys containing `"pos"/"embed"/"position"` |
| WR-06 | `src/dataset.py` | Scaler/augment ordering intent undocumented | Comment added |
| WR-07 | `src/api.py` | `torch.load(weights_only=False)` without hash check | SHA-256 check via `SEPSIS_MODEL_SHA256` env var |
| WR-08 | `src/fl_server.py` | Partial parameter mapping saved as valid checkpoint | `RuntimeError` raised on count mismatch |
| WR-09 | `scripts/run_fl_sim.py` | Server left hanging if client launch fails | `try/except` terminates server on exception |
| WR-10 | `src/parallel_preprocess.py` | `'y_seq_full' in dir()` — fragile anti-pattern | `y_seq_full = None` init; `is not None` check |
| WR-11 | `app_pages/1_04_🧪_Model_Metrics.py` | `rng` seeded inside loop — same bootstrap samples for all models | Comment confirms `rng` correctly outside loop |
| WR-12 | `src/clinical_metrics.py` | F1 `max(1,...)` denominator differs from sklearn | Clarifying comment added |
| WR-13 | `src/config_schema.py` | Pydantic v1 fallback shim skipped `_check_ratios` validator | Shim removed; pydantic v2 required |
| WR-14 | `src/model.py:105` | `src_key_padding_mask` not moved to `x.device` before cat | `.to(x.device)` added |
| WR-15 | `requirements.txt` | `torch>=2.0.0` no upper bound | `torch>=2.0.0,<3.0.0` |
| WR-16 | `Dockerfile` | Python 3.11 in Dockerfile vs 3.13 in CI | Both stages updated to `python:3.13-slim` |
| WR-17 | `src/fl_client.py` | `ds.y_indexed is None` → silent all-zero labels | `logger.warning(...)` emitted |
| WR-18 | `scripts/run_pipeline.py` | `range(n_clients - 1)` excluded last client from FL | `range(n_clients)` |

### Info — all resolved

| ID | File | Issue | Verification |
|----|------|-------|-------------|
| IN-01 | `src/utils.py` | `logging.basicConfig` at module scope | Removed |
| IN-02 | `app.py` | Page modules re-executed on every navigation | Cached in `st.session_state["page_modules"]` |
| IN-03 | `src/plot_results.py` | PRC baseline drawn even when no data plotted | `any_plotted` flag guards baseline draw |
| IN-04 | `src/fl_client.py` | `start_client` retry behaviour undocumented | Docstring added |
| IN-05 | `configs/base.yaml` | `augment: true` intent unclear | Comment clarifies CLI-only application |
| IN-06 | `.gitignore` | `predictions.jsonl` not gitignored | `*.jsonl` added |
| IN-07 | `src/ensemble.py` | `import logging` inside function body | Moved to module level |
| IN-08 | `tests/test_train.py` | `test_early_stopping_fires` underdocumented | No fix required; logic confirmed correct |

---

_Reviewed: 2026-06-17_
_Commit: dc18d34 (improvements/v3)_
_Depth: deep_
