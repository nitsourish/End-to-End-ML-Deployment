# =============================================================================
#  Makefile — common developer shortcuts
# =============================================================================

.PHONY: help install train serve test lint docker-build mlflow-start clean

PYTHON               ?= python3
DATA                 ?= data/creditcard_fraud.csv
ARTS                 ?= artifacts
PORT                 ?= 8000
MLPORT               ?= 5000
FEATURE_STORE_BUCKET ?= $(shell echo $$FEATURE_STORE_BUCKET)

help:
	@echo ""
	@echo "  make install          Install Python dependencies"
	@echo "  make upload-data      Upload raw CSV to S3 feature store bucket"
	@echo "  make train            Run feature pipeline + train model (local MLflow)"
	@echo "  make train-s3         Train + save features to S3 feature store"
	@echo "  make train-from-store Train by loading features from S3 (fast retrain)"
	@echo "  make serve            Start FastAPI server on :$(PORT)"
	@echo "  make test             Run all tests with coverage"
	@echo "  make lint             Lint with ruff"
	@echo "  make mlflow-start     Start local MLflow tracking server on :$(MLPORT)"
	@echo "  make docker-build     Build Docker image locally"
	@echo "  make drift            Run Evidently drift report (reference vs current)"
	@echo "  make clean            Remove generated artefacts + reports"
	@echo ""

# ---- Setup ------------------------------------------------------------------
install:
	pip install -r requirements.txt

# ---- Data upload to S3 feature store ----------------------------------------
upload-data:
	$(PYTHON) scripts/upload_data.py \
		--bucket $(FEATURE_STORE_BUCKET) \
		--data-path $(DATA)

# ---- Training ---------------------------------------------------------------
train:
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
	$(PYTHON) -m src.training.train \
		--data-path $(DATA) \
		--artifacts-dir $(ARTS)

train-s3:
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
	$(PYTHON) -m src.training.train \
		--data-path $(DATA) \
		--artifacts-dir $(ARTS) \
		--s3-bucket $(FEATURE_STORE_BUCKET)

train-from-store:
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
	$(PYTHON) -m src.training.train \
		--artifacts-dir $(ARTS) \
		--s3-bucket $(FEATURE_STORE_BUCKET) \
		--from-feature-store

# ---- Serving ----------------------------------------------------------------
serve:
	$(PYTHON) -m uvicorn src.serving.app:app \
		--host 0.0.0.0 \
		--port $(PORT) \
		--reload

# ---- Tests ------------------------------------------------------------------
test:
	MLFLOW_TRACKING_URI=file:///tmp/mlruns-test \
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ --select E,F,W --ignore E501

# ---- MLflow -----------------------------------------------------------------
mlflow-start:
	@echo "Starting MLflow server on http://localhost:$(MLPORT) ..."
	@echo "DB: mlflow.db  |  Artifacts: ./mlflow-artifacts/"
	mlflow server \
		--host 0.0.0.0 \
		--port $(MLPORT) \
		--backend-store-uri postgresql://$(MLFLOW_DB_USER):$(MLFLOW_DB_PASS)@$(MLFLOW_DB_HOST)/mlflow \
		--default-artifact-root s3://$(MLFLOW_S3_BUCKET)/mlflow-artifacts \
		|| mlflow server \
		   --host 0.0.0.0 \
		   --port $(MLPORT) \
		   --backend-store-uri sqlite:///mlflow.db \
		   --default-artifact-root ./mlflow-artifacts

# ---- Monitoring -------------------------------------------------------------
drift:
	$(PYTHON) -m src.monitoring.drift_detection \
		--reference $(DATA) \
		--current   $(DATA) \
		--report-dir reports/ \
		--reference-nrows 10000 \
		--current-nrows 5000

# ---- Docker -----------------------------------------------------------------
docker-build:
	mkdir -p artifacts
	docker build \
		-f infrastructure/docker/Dockerfile \
		-t fraud-detection:local \
		.

docker-run:
	docker run --rm -p $(PORT):8000 \
		-e MLFLOW_TRACKING_URI=http://host.docker.internal:$(MLPORT) \
		-e MODEL_NAME=fraud-detection-lr \
		-e MODEL_STAGE=Staging \
		fraud-detection:local

# ---- Clean ------------------------------------------------------------------
clean:
	rm -rf artifacts/ reports/ mlruns/ mlflow-artifacts/ mlflow.db
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
