"""
Feature pipeline for credit card fraud detection.

Steps:
  1. Load raw data
  2. Transform: log(Amount+1), hour-of-day from Time, drop raw Time/Amount
  3. Split train/validation (stratified to preserve class ratio)
  4. Fit StandardScaler on train, apply to both splits
  5. Feature selection via L1-penalised LR → keep non-zero coefficients

The fitted scaler and selected feature names are saved as artifacts so the
serving layer can reproduce exactly the same transformations at inference time.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
RAW_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
TARGET = "Class"
RANDOM_STATE = 42
VAL_SIZE = 0.20          # 80/20 split
L1_C_SELECTION = 0.1    # tighter L1 to remove truly irrelevant features


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and do a basic sanity-check."""
    df = pd.read_csv(path)
    required = set(RAW_FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features, drop originals.

    Transformations
    ---------------
    - ``log_amount``  = log(Amount + 1)   — right-skewed monetary values
    - ``hour``        = (Time % 86400) / 3600  — within-day periodicity
    """
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour"] = (df["Time"] % 86400) / 3600
    df.drop(columns=["Amount", "Time"], inplace=True)
    return df


def build_feature_list(df: pd.DataFrame) -> list[str]:
    """Return ordered list of feature columns (everything except Target)."""
    return [c for c in df.columns if c != TARGET]


def split_data(
    df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train / validation split."""
    feature_cols = build_feature_list(df)
    X = df[feature_cols]
    y = df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        "Split → train=%d  val=%d  fraud_rate_train=%.4f  fraud_rate_val=%.4f",
        len(X_train), len(X_val),
        y_train.mean(), y_val.mean(),
    )
    return X_train, X_val, y_train, y_val


def fit_scaler(X_train: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame]:
    """Fit StandardScaler on training set, return scaler + scaled DataFrame."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    return scaler, X_scaled


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a pre-fitted scaler; preserves column names and index."""
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )


def select_features_l1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = L1_C_SELECTION,
    random_state: int = RANDOM_STATE,
) -> list[str]:
    """
    Fit a sparse L1 Logistic Regression and return features with non-zero
    coefficients.  Uses class_weight='balanced' to handle the severe
    imbalance (≈0.17 % fraud).
    """
    lr_selector = LogisticRegression(
        C=C,
        l1_ratio=1.0,          # pure L1 (elasticnet solver supports l1_ratio)
        solver="saga",
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state,
    )
    lr_selector.fit(X_train, y_train)

    coef = np.abs(lr_selector.coef_[0])
    selected = [col for col, c in zip(X_train.columns, coef) if c > 0]
    dropped = [col for col, c in zip(X_train.columns, coef) if c == 0]

    logger.info(
        "L1 feature selection: kept %d / %d  (dropped: %s)",
        len(selected), X_train.shape[1], dropped or "none",
    )
    return selected


def run_feature_pipeline(
    data_path: str,
    artifacts_dir: str = "artifacts",
    s3_bucket: str | None = None,
    feature_store_version: str | None = None,
) -> dict:
    """
    Full feature pipeline.

    Parameters
    ----------
    data_path : str
        Local path to the raw CSV (or any path resolvable by pandas).
    artifacts_dir : str
        Local directory to write scaler.joblib + selected_features.json.
    s3_bucket : str | None
        If set, saves engineered features + artifacts to S3 via S3FeatureStore.
        Also uploads the raw CSV if not already present.
    feature_store_version : str | None
        Explicit version key to use when writing to the feature store.
        Auto-generated (timestamp) when None.

    Returns a dict with:
      - X_train, X_val, y_train, y_val  (scaled, selected features)
      - scaler, selected_features, all_features
      - scaler_path, features_path       (local artifact paths)
      - feature_store_version            (version key; None if no S3 store)
    """
    import json
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load & engineer
    df = load_data(data_path)
    df = engineer_features(df)

    # 2. Split (stratified)
    X_train_raw, X_val_raw, y_train, y_val = split_data(df)

    # 3. Scale (fit on train only → no leakage)
    scaler, X_train_scaled = fit_scaler(X_train_raw)
    X_val_scaled = apply_scaler(scaler, X_val_raw)

    # 4. Feature selection via L1
    selected_features = select_features_l1(X_train_scaled, y_train)

    X_train_final = X_train_scaled[selected_features]
    X_val_final = X_val_scaled[selected_features]

    # 5. Persist artifacts locally
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
    features_path = os.path.join(artifacts_dir, "selected_features.json")
    joblib.dump(scaler, scaler_path)

    feature_meta = {
        "all_features": list(X_train_raw.columns),
        "selected_features": selected_features,
    }
    with open(features_path, "w") as f:
        json.dump(feature_meta, f, indent=2)

    logger.info("Scaler saved locally → %s", scaler_path)
    logger.info("Feature meta saved locally → %s", features_path)

    # 6. (Optional) Push to S3 feature store
    fs_version = None
    if s3_bucket:
        from src.feature_pipeline.feature_store import S3FeatureStore
        fs = S3FeatureStore(s3_bucket)

        # Upload raw data (idempotent — skips if already present)
        if os.path.exists(data_path):
            fs.upload_raw_data(data_path)

        # Save offline features (parquet)
        fs_version = fs.save_offline_features(
            X_train_final, X_val_final,
            y_train, y_val,
            source_path=data_path,
            version=feature_store_version,
        )

        # Save artifacts (scaler + feature meta)
        fs.save_artifacts(
            scaler, feature_meta,
            version=fs_version,
            local_artifacts_dir=artifacts_dir,
        )
        logger.info("Feature store version written → s3://%s  version=%s",
                    s3_bucket, fs_version)

    return {
        "X_train": X_train_final,
        "X_val": X_val_final,
        "y_train": y_train,
        "y_val": y_val,
        "scaler": scaler,
        "selected_features": selected_features,
        "all_features": list(X_train_raw.columns),
        "scaler_path": scaler_path,
        "features_path": features_path,
        "feature_store_version": fs_version,
    }


# ------------------------------------------------------------------
# Inference-time helper
# ------------------------------------------------------------------

def transform_input(
    raw: dict,
    scaler: StandardScaler,
    selected_features: list[str],
    all_features: list[str],
) -> np.ndarray:
    """
    Transform a single raw inference payload dict into a 2-D array
    ready for model.predict_proba().

    ``raw`` must contain the original keys: V1..V28, Amount, Time.
    """
    df = pd.DataFrame([raw])

    # Engineer same derived features
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour"] = (df["Time"] % 86400) / 3600
    df.drop(columns=["Amount", "Time"], inplace=True)

    # Align to expected column order, fill any missing with 0
    for col in all_features:
        if col not in df.columns:
            df[col] = 0.0
    df = df[all_features]

    # Scale then select
    scaled = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=all_features)
    return scaled_df[selected_features].values
