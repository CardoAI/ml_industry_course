# ──────────────────────────────────────────────────────────────────────────────
# Day 4 — MLOps Workshop Makefile
#
# Run from the REPO ROOT, not from inside day4/:
#   make -f day4/Makefile <target>
#
# Workshop sequence:
#   1a. make -f day4/Makefile install
#   1b. make -f day4/Makefile kernel-install
#   2a. make -f day4/Makefile mlflow-server   ← run in a separate terminal
#   2b. make -f day4/Makefile train
#   3.  make -f day4/Makefile drift-check
#   4.  make -f day4/Makefile docker-build
#       make -f day4/Makefile docker-run
# ──────────────────────────────────────────────────────────────────────────────

UV_RUN         := uv run --project day4 --frozen
DATA_PATH      := day1/generated/adult_income_issues.csv
MLFLOW_URI     := http://127.0.0.1:5000
EXPERIMENT     := adult-income-lgbm
MODEL_NAME     := adult-income-classifier
OUTPUT_DIR     := day4/outputs
DOCKER_TAG     := adult-income-predictor:latest
COMPOSE_FILE   := day4/docker-compose.yml

# Read the last run ID saved by `make train`
RUN_ID_FILE    := $(OUTPUT_DIR)/last_run_id.txt
MODEL_RUN_ID   ?= $(shell cat $(RUN_ID_FILE) 2>/dev/null)

.DEFAULT_GOAL  := help

.PHONY: help install kernel-install nannyml-install mlflow-server train drift-check \
        docker-build docker-run docker-compose-up clean

# ── Help ───────────────────────────────────────────────────────────────────────
help: ## Show available targets
	@echo ""
	@echo "  Day 4 MLOps Workshop"
	@echo "  ─────────────────────────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Environment setup ──────────────────────────────────────────────────────────
install: ## Install Day 4 Python dependencies
	cd day4 && uv sync

kernel-install: ## Register the Jupyter kernel for VS Code / Jupyter
	$(UV_RUN) python -m ipykernel install --user \
	    --name mlops-industry-course \
	    --display-name "MLOps Industry Course"

nannyml-install: ## Install NannyML in an isolated compat env (needed if pandas conflict)
	cd day4/nannyml-compat && uv sync

# ── MLflow server (run in a separate terminal) ─────────────────────────────────
mlflow-server: ## Start the local MLflow tracking server (separate terminal)
	@echo "Open http://127.0.0.1:5000 in your browser once the server is ready."
	$(UV_RUN) mlflow server \
	    --host 127.0.0.1 \
	    --port 5000 \
	    --backend-store-uri sqlite:///mlflow.db

# ── Training ───────────────────────────────────────────────────────────────────
train: ## Train LightGBM and log the run to MLflow; saves run ID to outputs/
	@mkdir -p $(OUTPUT_DIR)
	$(UV_RUN) python day4/src/train.py \
	    --n-estimators 300 \
	    --learning-rate 0.1 \
	    --max-depth 5 \
	    --mlflow-uri $(MLFLOW_URI) \
	    --experiment-name $(EXPERIMENT) \
	    --run-name "makefile-run" \
	    --data-path $(DATA_PATH) \
	    --register true \
	    --model-name $(MODEL_NAME) \
	    2>&1 | tee $(OUTPUT_DIR)/train.log
	@echo ""
	@echo "Run ID: $$(cat $(RUN_ID_FILE))"
	@echo "Open http://127.0.0.1:5000 to inspect the run."

# ── Drift detection ────────────────────────────────────────────────────────────
drift-check: ## Run NannyML drift and performance check; log report to MLflow
	@if [ ! -f $(RUN_ID_FILE) ]; then \
	    echo "❌  No run ID found. Run 'make -f day4/Makefile train' first."; \
	    exit 1; \
	fi
	$(UV_RUN) python day4/src/nannyml_check.py \
	    --data-path $(DATA_PATH) \
	    --model-uri "models:/$(MODEL_NAME)/Production" \
	    --mlflow-uri $(MLFLOW_URI) \
	    --output-dir $(OUTPUT_DIR) \
	    --fail-on-drift \
	    2>&1 | tee $(OUTPUT_DIR)/drift.log
	@echo "NannyML reports saved to $(OUTPUT_DIR)/"

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build: ## Build the prediction service Docker image
	@if [ -z "$(MODEL_RUN_ID)" ]; then \
	    echo "❌  No MODEL_RUN_ID. Run 'make -f day4/Makefile train' first."; \
	    exit 1; \
	fi
	@echo "Extracting model for run $(MODEL_RUN_ID) …"
	@$(UV_RUN) python -c "\
	import mlflow, shutil; \
	mlflow.set_tracking_uri('$(MLFLOW_URI)'); \
	src = mlflow.artifacts.download_artifacts(run_id='$(MODEL_RUN_ID)', artifact_path='model'); \
	dst = 'day4/outputs/model'; \
	shutil.rmtree(dst, ignore_errors=True); \
	shutil.copytree(src, dst); \
	print(f'Model extracted to {dst}')"
	@echo "Building image …"
	docker build \
	    -t $(DOCKER_TAG) \
	    -f day4/Dockerfile \
	    .
	@echo ""
	@echo "Image built: $(DOCKER_TAG)"
	@echo "Run: make -f day4/Makefile docker-run"

docker-run: ## Run the prediction service container on port 8080
	docker run --rm -p 8080:8080 $(DOCKER_TAG)

docker-compose-up: ## Start MLflow UI + model service with docker compose
	docker compose -f $(COMPOSE_FILE) up

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean: ## Remove generated outputs (logs, reports, run ID file)
	rm -rf $(OUTPUT_DIR)
	@echo "Cleaned $(OUTPUT_DIR)/"
