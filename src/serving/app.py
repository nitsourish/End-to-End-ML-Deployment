"""
FastAPI serving layer for real-time fraud inference.

Endpoints
---------
GET  /health           — liveness probe (used by AWS App Runner)
GET  /ready            — readiness probe (model loaded)
POST /predict          — single transaction inference
POST /predict/batch    — batch inference (list of transactions)
GET  /model/info       — current loaded model metadata

Environment variables
---------------------
MLFLOW_TRACKING_URI    — MLflow server URI
MODEL_NAME             — registered model name (default: fraud-detection-lr)
MODEL_STAGE            — registry stage to load (default: Staging)
ARTIFACTS_DIR          — local path where scaler / features are cached
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.feature_pipeline.feature_engineering import transform_input

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME           = os.getenv("MODEL_NAME", "fraud-detection-lr")
MODEL_STAGE          = os.getenv("MODEL_STAGE", "Staging")
ARTIFACTS_DIR        = os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts"))
FRAUD_THRESHOLD      = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
FEATURE_STORE_BUCKET = os.getenv("FEATURE_STORE_BUCKET")  # None → skip S3 for artifacts

# ------------------------------------------------------------------
# Global model state (loaded once at startup)
# ------------------------------------------------------------------
_state: dict = {
    "model": None,
    "scaler": None,
    "selected_features": None,
    "all_features": None,
    "model_version": None,
    "loaded_at": None,
}


def _load_model_and_artifacts() -> None:
    """Load model from MLflow registry + feature artifacts from disk."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT / 'mlruns'}")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from %s …", model_uri)

    try:
        _state["model"] = mlflow.sklearn.load_model(model_uri)

        # Fetch version metadata
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        _state["model_version"] = versions[0].version if versions else "unknown"
        logger.info("Loaded model version %s", _state["model_version"])

    except Exception as e:
        logger.warning("MLflow registry load failed (%s); falling back to local artefacts", e)
        # Fallback: load from local artefacts path (useful in CI / unit tests)
        local_model_path = Path(ARTIFACTS_DIR) / "model"
        if local_model_path.exists():
            _state["model"] = mlflow.sklearn.load_model(str(local_model_path))
            _state["model_version"] = "local"
        else:
            logger.error("No model found. Start the server after training.")
            return

    # Load scaler + feature list
    # Priority: S3 feature store (with local cache) → local disk
    scaler, meta = _load_feature_artifacts()

    if scaler is not None:
        _state["scaler"] = scaler
        _state["all_features"] = meta["all_features"]
        _state["selected_features"] = meta["selected_features"]
        logger.info(
            "Feature artifacts loaded: %d all / %d selected features",
            len(meta["all_features"]),
            len(meta["selected_features"]),
        )
    else:
        logger.error("Feature artifacts not found — predictions will fail.")

    _state["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_feature_artifacts():
    """
    Load scaler + feature_meta.

    Strategy
    --------
    1. If FEATURE_STORE_BUCKET is set → pull from S3 (caches locally in ARTIFACTS_DIR)
    2. Otherwise → load from local ARTIFACTS_DIR
    3. Returns (None, None) if nothing is found.
    """
    if FEATURE_STORE_BUCKET:
        try:
            from src.feature_pipeline.feature_store import S3FeatureStore
            fs = S3FeatureStore(FEATURE_STORE_BUCKET)
            logger.info("Pulling feature artifacts from S3 bucket: %s", FEATURE_STORE_BUCKET)
            scaler, meta = fs.load_artifacts(local_cache_dir=ARTIFACTS_DIR)
            return scaler, meta
        except Exception as e:
            logger.warning("S3 artifact load failed (%s); falling back to local cache", e)

    # Local fallback
    scaler_path   = Path(ARTIFACTS_DIR) / "scaler.joblib"
    features_path = Path(ARTIFACTS_DIR) / "selected_features.json"

    if scaler_path.exists() and features_path.exists():
        scaler = joblib.load(scaler_path)
        with open(features_path) as f:
            meta = json.load(f)
        logger.info("Loaded feature artifacts from local cache %s", ARTIFACTS_DIR)
        return scaler, meta

    return None, None


# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------

class Transaction(BaseModel):
    """Raw transaction — matches original CSV columns."""
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = Field(..., ge=0, description="Transaction amount")

    @field_validator("Amount")
    @classmethod
    def amount_non_negative(cls, v):
        if v < 0:
            raise ValueError("Amount must be ≥ 0")
        return v


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load model on startup, nothing on shutdown."""
    _load_model_and_artifacts()
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud inference using Logistic Regression + MLflow",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    """AWS App Runner liveness probe."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness — fails if model not loaded."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": _state["model_version"]}


@app.get("/model/info")
def model_info():
    """Return metadata about the currently loaded model."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_version": _state["model_version"],
        "loaded_at": _state["loaded_at"],
        "fraud_threshold": FRAUD_THRESHOLD,
        "selected_features": _state["selected_features"],
        "n_selected_features": len(_state["selected_features"] or []),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Predict fraud probability for a single transaction.

    Returns fraud_probability ∈ [0, 1] and a boolean is_fraud flag
    based on the configured threshold.
    """
    _check_model_ready()
    t0 = time.perf_counter()

    raw = transaction.model_dump()
    X = transform_input(
        raw,
        _state["scaler"],
        _state["selected_features"],
        _state["all_features"],
    )
    prob = float(_state["model"].predict_proba(X)[0, 1])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return PredictionResponse(
        fraud_probability=round(prob, 6),
        is_fraud=prob >= FRAUD_THRESHOLD,
        threshold=FRAUD_THRESHOLD,
        model_version=_state["model_version"],
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(transactions: list[Transaction]):
    """Batch inference — accepts up to 1000 transactions."""
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size must be ≤ 1000")
    _check_model_ready()
    t0 = time.perf_counter()

    results = []
    for txn in transactions:
        raw = txn.model_dump()
        X = transform_input(
            raw,
            _state["scaler"],
            _state["selected_features"],
            _state["all_features"],
        )
        prob = float(_state["model"].predict_proba(X)[0, 1])
        results.append(
            PredictionResponse(
                fraud_probability=round(prob, 6),
                is_fraud=prob >= FRAUD_THRESHOLD,
                threshold=FRAUD_THRESHOLD,
                model_version=_state["model_version"],
                latency_ms=None,
            )
        )

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("Batch of %d processed in %.1f ms", len(transactions), total_ms)
    return BatchPredictionResponse(predictions=results, count=len(results))


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _check_model_ready():
    if _state["model"] is None or _state["scaler"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model or feature pipeline not loaded. Has training been run?",
        )


# ------------------------------------------------------------------
# Run directly (dev)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=True)
