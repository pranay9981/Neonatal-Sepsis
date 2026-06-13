# Neonatal Sepsis Detection — Makefile
# Works with GNU make (Git Bash on Windows, or Linux/macOS).
# Run `make help` to see all targets.

.PHONY: help preprocess split train-local fl-sim evaluate plot dashboard api tests hyper pipeline clean clean-all

# ── Configurable defaults (override on the command line) ──────────────
RAW_FOLDER    ?= data/raw
MODEL         ?= transformer
EPOCHS        ?= 10
BATCH_SIZE    ?= 64
FL_ROUNDS     ?= 5
N_CLIENTS     ?= 3
N_FEATURES    ?= 40
SEQ_LEN       ?= 48
PATIENCE      ?= 5

PROCESSED_DIR  = data/processed/patients
INDEX          = $(PROCESSED_DIR)/index_with_labels.pt
CLIENTS_DIR    = data/processed/clients
CLIENT1_INDEX  = $(CLIENTS_DIR)/client1/index.pt
CLIENT2_INDEX  = $(CLIENTS_DIR)/client2/index.pt
CLIENT3_INDEX  = $(CLIENTS_DIR)/client3/index.pt
GLOBAL_BEST    = server_out/global_best.pt

# ── Help ──────────────────────────────────────────────────────────────
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Individual pipeline steps ─────────────────────────────────────────
preprocess: ## Step 1 — Preprocess raw PSV files into patient tensors
	python src/parallel_preprocess.py \
		--raw_folder $(RAW_FOLDER) \
		--out_folder $(PROCESSED_DIR) \
		--seq_len $(SEQ_LEN)

split: ## Step 2 — Split processed patients into federated client folders
	python src/split_clients.py \
		--processed_folder $(PROCESSED_DIR) \
		--out_root $(CLIENTS_DIR) \
		--n_clients $(N_CLIENTS)

train-local: ## Step 3 — Train a local baseline model
	python src/train_local.py \
		--index $(INDEX) \
		--model $(MODEL) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--patience $(PATIENCE) \
		--run_name local_baseline

fl-sim: ## Step 4 — Run federated learning simulation (server + clients auto-launched)
	python scripts/run_fl_sim.py \
		--client_indexes $(CLIENT1_INDEX) $(CLIENT2_INDEX) \
		--model $(MODEL) \
		--rounds $(FL_ROUNDS) \
		--n_features $(N_FEATURES) \
		--seq_len $(SEQ_LEN)

evaluate: ## Step 5 — Evaluate federated + local models on held-out test set
	python src/evaluate.py \
		--index $(CLIENT3_INDEX) \
		--ckpt $(GLOBAL_BEST) \
		--model $(MODEL) \
		--n_features $(N_FEATURES) \
		--seq_len $(SEQ_LEN) \
		--out_file eval_results_federated.json
	@LATEST=$$(ls -t runs/*/checkpoints/model_best.pt 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		python src/evaluate.py \
			--index $(CLIENT3_INDEX) \
			--ckpt "$$LATEST" \
			--model $(MODEL) \
			--n_features $(N_FEATURES) \
			--seq_len $(SEQ_LEN) \
			--out_file eval_results_local.json; \
	else \
		echo "[MAKE] No local checkpoint found — skipping local evaluation"; \
	fi

plot: ## Step 6 — Generate ROC and PRC comparison plots
	python src/plot_results.py \
		--results eval_results_federated.json eval_results_local.json \
		--out_file model_comparison_plot.png

# ── Run everything at once ────────────────────────────────────────────
pipeline: ## Run the full end-to-end pipeline (smart skip if outputs exist)
	python scripts/run_pipeline.py --raw_folder $(RAW_FOLDER)

# ── Tools ─────────────────────────────────────────────────────────────
dashboard: ## Launch the Streamlit dashboard
	streamlit run app.py

api: ## Start the FastAPI inference server on port 8000
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

tests: ## Run all unit and integration tests
	python -m pytest tests/ -v

hyper: ## Run hyperparameter search
	python src/hyperparam_search.py

# ── Cleanup ───────────────────────────────────────────────────────────
clean: ## Remove training runs and generated results (keeps processed data)
	rm -rf runs/ hyper_results/
	rm -f eval_results_*.json model_comparison_plot*.png

clean-all: ## Remove ALL generated outputs including processed data and checkpoints
	rm -rf runs/ hyper_results/ data/processed/ server_out/ checkpoints/
	rm -f eval_results_*.json model_comparison_plot*.png
