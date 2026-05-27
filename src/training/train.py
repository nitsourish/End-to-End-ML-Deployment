"""
Model training with MLflow experiment tracking and model registration.

Usage:
    # Train from local CSV (default):
    python -m src.training.train --data-path data/creditcard_fraud.csv

    # Train + push features to S3 feature store:
    python -m src.training.train --s3-bucket my-fraud-bucket

    # Train by reading pre-built features directly from S3 (fastest CI retrain):
    python -m src.training.train --s3-bucket my-fraud-bucket --from-feature-store

Environment variables (set in .env or CI secrets):
    MLFLOW_TRACKING_URI     — e.g. http://localhost:5000 or a remote server
    FEATURE_STORE_BUCKET    — S3 bucket for feature store (overrides --s3-bucket)
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from dotenv import load_dotenv

# Add project root to path so `src` is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.feature_pipeline.feature_engineering import run_feature_pipeline

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Defaults (overridable via CLI or env)
# ------------------------------------------------------------------
DEFAULT_DATA_PATH = str(ROOT / "data" / "creditcard_fraud.csv")
DEFAULT_EXPERIMENT = "fraud-detection-lr"
DEFAULT_ARTIFACTS_DIR = str(ROOT / "artifacts")
REGISTERED_MODEL_NAME = "fraud-detection-lr"

# Hyperparameter grid for cross-validated C
C_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
    }


def _train_single(X_train, y_train, C: float, random_state: int = 42):
    model = LogisticRegression(
        C=C,
        l1_ratio=0.0,          # pure L2 for the final model (generalisation)
        solver="saga",         # supports large datasets efficiently
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------------
# Main training routine
# ------------------------------------------------------------------

def train(
    data_path: str = DEFAULT_DATA_PATH,
    experiment_name: str = DEFAULT_EXPERIMENT,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
    register_model: bool = True,
    s3_bucket: str | None = None,
    from_feature_store: bool = False,
) -> str:
    """
    Run the full training pipeline and return the MLflow run_id of the
    best model.

    Parameters
    ----------
    s3_bucket : str | None
        When set, the feature pipeline writes engineered features + artifacts
        to S3. Falls back to env var FEATURE_STORE_BUCKET.
    from_feature_store : bool
        When True, skip local feature engineering entirely and load pre-built
        features directly from S3 (requires s3_bucket to be set).
    """
    bucket = s3_bucket or os.getenv("FEATURE_STORE_BUCKET")

    # ---- Feature pipeline ----------------------------------------
    if from_feature_store and bucket:
        logger.info("Loading pre-built features from S3 feature store …")
        pipeline = _load_pipeline_from_store(bucket, artifacts_dir)
    else:
        logger.info("Running feature pipeline locally …")
        pipeline = run_feature_pipeline(
            data_path,
            artifacts_dir=artifacts_dir,
            s3_bucket=bucket,
        )
    X_train = pipeline["X_train"]
    X_val = pipeline["X_val"]
    y_train = pipeline["y_train"]
    y_val = pipeline["y_val"]
    selected_features = pipeline["selected_features"]
    all_features = pipeline["all_features"]

    # ---- MLflow setup --------------------------------------------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT / 'mlruns'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    # ---- Grid-search over C with one run per candidate -----------
    best_run_id: str | None = None
    best_auc: float = -1.0
    best_C: float = C_VALUES[0]

    for C in C_VALUES:
        with mlflow.start_run(run_name=f"LR_C={C}") as run:
            run_id = run.info.run_id

            model = _train_single(X_train, y_train, C)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            metrics = _compute_metrics(y_val, y_pred, y_prob)

            # Log params + metrics
            mlflow.log_params({
                "C": C,
                "penalty": "l2",
                "solver": "saga",
                "class_weight": "balanced",
                "selected_features": len(selected_features),
            })
            mlflow.log_metrics(metrics)

            # Log confusion matrix as a text artifact
            cm = confusion_matrix(y_val, y_pred)
            cm_text = (
                f"Confusion matrix (val):\n{cm}\n\n"
                f"{classification_report(y_val, y_pred, target_names=['legit','fraud'])}"
            )
            mlflow.log_text(cm_text, "confusion_matrix.txt")

            # Log feature meta
            mlflow.log_dict(
                {"all_features": all_features, "selected_features": selected_features},
                "feature_config.json",
            )

            logger.info("C=%-5s  ROC-AUC=%.4f  Avg-Prec=%.4f  F1=%.4f",
                        C, metrics["roc_auc"], metrics["avg_precision"], metrics["f1"])

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run_id = run_id
                best_C = C

    # ---- Re-train best C and log full artifacts ------------------
    logger.info("Best C=%s  ROC-AUC=%.4f — registering …", best_C, best_auc)

    with mlflow.start_run(run_name=f"BEST_LR_C={best_C}") as run:
        best_run_id = run.info.run_id

        model = _train_single(X_train, y_train, best_C)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = _compute_metrics(y_val, y_pred, y_prob)

        mlflow.log_params({
            "C": best_C,
            "penalty": "l2",
            "solver": "saga",
            "class_weight": "balanced",
            "selected_features": len(selected_features),
            "val_size": 0.20,
            "data_path": data_path,
        })
        mlflow.log_metrics(metrics)

        # Scaler + feature config as artefacts
        mlflow.log_artifact(pipeline["scaler_path"], artifact_path="feature_pipeline")
        mlflow.log_artifact(pipeline["features_path"], artifact_path="feature_pipeline")

        # Signature for model serving
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])

        # Log sklearn model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME if register_model else None,
            input_example=X_train.iloc[:3],
        )

        logger.info("Registered model '%s'  run_id=%s", REGISTERED_MODEL_NAME, best_run_id)

    # ---- Transition latest version to Staging --------------------
    if register_model:
        _promote_to_staging(REGISTERED_MODEL_NAME)

    return best_run_id


def _load_pipeline_from_store(bucket: str, artifacts_dir: str) -> dict:
    """
    Load pre-materialised features + artifacts from S3 feature store.
    Returns a dict shaped identically to run_feature_pipeline().
    """
    from src.feature_pipeline.feature_store import S3FeatureStore
    import json

    fs = S3FeatureStore(bucket)
    version = fs.latest_version()
    logger.info("Feature store version: %s", version)

    X_train, X_val, y_train, y_val = fs.load_offline_features(version)
    scaler, feature_meta = fs.load_artifacts(
        version=version,
        local_cache_dir=artifacts_dir,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "scaler": scaler,
        "selected_features": feature_meta["selected_features"],
        "all_features": feature_meta["all_features"],
        "scaler_path": os.path.join(artifacts_dir, "scaler.joblib"),
        "features_path": os.path.join(artifacts_dir, "selected_features.json"),
        "feature_store_version": version,
    }


def _promote_to_staging(model_name: str) -> None:
    """Set the latest version of a registered model to Staging."""
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return
    latest = max(versions, key=lambda v: int(v.version))
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage="Staging",
            archive_existing_versions=True,
        )
        logger.info("Model '%s' v%s promoted to Staging", model_name, latest.version)
    except Exception as e:
        logger.warning("Could not transition model stage: %s", e)


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud-detection LR model")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--no-register", action="store_true",
                        help="Skip model registration (useful for quick tests)")
    parser.add_argument("--s3-bucket", default=None,
                        help="S3 bucket for feature store (overrides FEATURE_STORE_BUCKET env var)")
    parser.add_argument("--from-feature-store", action="store_true",
                        help="Load pre-built features from S3 instead of re-running the pipeline")
    args = parser.parse_args()

    run_id = train(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        artifacts_dir=args.artifacts_dir,
        register_model=not args.no_register,
        s3_bucket=args.s3_bucket,
        from_feature_store=args.from_feature_store,
    )
    logger.info("Done. Best run_id: %s", run_id)
