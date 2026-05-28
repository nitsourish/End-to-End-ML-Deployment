"""
S3-backed Feature Store for the fraud-detection pipeline.

Layout inside the S3 bucket
----------------------------
{bucket}/
  raw/
    creditcard_fraud.csv            ← original raw data (immutable)
  features/
    offline/
      {version}/
        train.parquet               ← engineered + scaled train split
        val.parquet                 ← validation split
        metadata.json               ← lineage: source hash, shape, timestamp
    artifacts/
      {version}/
        scaler.joblib               ← fitted StandardScaler
        selected_features.json      ← all_features + selected_features
  mlflow-artifacts/                 ← used by MLflow (managed separately)

Design principles
-----------------
- S3 is the *source of truth*; local disk is a *cache*.
- Every write is versioned — old versions are never overwritten.
- ``latest`` symlink files (metadata.json with a ``latest_version`` key)
  make it easy to pull the most recent features without knowing the version.
- For online/real-time inference we don't need a separate online store:
  the scaler + selected_features artifacts are small (~KB) and are loaded
  into memory at API startup. The model itself is served from MLflow.

Environment variables
---------------------
  FEATURE_STORE_BUCKET   — S3 bucket name (required for S3 operations)
  AWS_ACCESS_KEY_ID      — standard boto3 credentials
  AWS_SECRET_ACCESS_KEY
  AWS_DEFAULT_REGION
"""

import hashlib
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
import joblib
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ---- S3 key prefixes -------------------------------------------------------
PREFIX_RAW       = "raw"
PREFIX_OFFLINE   = "features/offline"
PREFIX_ARTIFACTS = "features/artifacts"
LATEST_POINTER   = "features/latest.json"   # points to most recent version key


# ---------------------------------------------------------------------------
# S3FeatureStore
# ---------------------------------------------------------------------------

class S3FeatureStore:
    """
    Read/write interface for raw data, offline features, and pipeline
    artifacts stored in an S3 bucket.

    Usage (write path — training)
    -----------------------------
    fs = S3FeatureStore("my-bucket")
    fs.upload_raw_data("data/creditcard_fraud.csv")          # once
    version = fs.save_offline_features(X_train, X_val, y_train, y_val)
    fs.save_artifacts(scaler, feature_meta, version)

    Usage (read path — training from store)
    ----------------------------------------
    version = fs.latest_version()
    X_train, X_val, y_train, y_val = fs.load_offline_features(version)
    scaler, meta = fs.load_artifacts(version)

    Usage (serving startup)
    -----------------------
    version = fs.latest_version()
    scaler, meta = fs.load_artifacts(version, local_cache_dir="/app/artifacts")
    """

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,   # useful for LocalStack / MinIO
    ):
        self.bucket = bucket
        # os.getenv(key, default) only uses the default when the key is ABSENT.
        # GitHub Actions sets missing secrets to "", so we need an extra `or`
        # to fall back from an empty string to the hardcoded default.
        effective_region = region or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        # Similarly, an empty-string endpoint_url must become None so boto3
        # uses its own default endpoint rather than constructing "s3..amazonaws.com".
        effective_endpoint = endpoint_url or os.getenv("AWS_S3_ENDPOINT_URL") or None
        self._s3 = boto3.client(
            "s3",
            region_name=effective_region,
            endpoint_url=effective_endpoint,
        )
        logger.info("S3FeatureStore initialised — bucket: s3://%s", bucket)

    # ------------------------------------------------------------------
    # Raw data
    # ------------------------------------------------------------------

    def upload_raw_data(
        self,
        local_path: str,
        s3_key: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Upload raw CSV to s3://{bucket}/raw/{filename}.

        Returns the S3 URI of the uploaded file.
        By default skips the upload if the key already exists (idempotent).
        """
        key = s3_key or f"{PREFIX_RAW}/{Path(local_path).name}"

        if not overwrite and self._key_exists(key):
            uri = f"s3://{self.bucket}/{key}"
            logger.info("Raw data already present at %s — skipping upload", uri)
            return uri

        logger.info("Uploading %s → s3://%s/%s …", local_path, self.bucket, key)
        self._s3.upload_file(local_path, self.bucket, key)
        uri = f"s3://{self.bucket}/{key}"
        logger.info("Raw data uploaded → %s", uri)
        return uri

    def download_raw_data(self, local_path: str, s3_key: Optional[str] = None) -> str:
        """Download raw CSV from S3 to local_path."""
        key = s3_key or f"{PREFIX_RAW}/creditcard_fraud.csv"
        logger.info("Downloading s3://%s/%s → %s …", self.bucket, key, local_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self.bucket, key, local_path)
        return local_path

    # ------------------------------------------------------------------
    # Offline feature store (Parquet)
    # ------------------------------------------------------------------

    def save_offline_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        source_path: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """
        Save engineered + scaled train/val splits to S3 as Parquet.

        Returns the version string (timestamp-based key).
        """
        version = version or _make_version()
        prefix = f"{PREFIX_OFFLINE}/{version}"

        train_df = X_train.copy()
        train_df["__target__"] = y_train.values

        val_df = X_val.copy()
        val_df["__target__"] = y_val.values

        # Upload parquet files
        self._upload_parquet(train_df, f"{prefix}/train.parquet")
        self._upload_parquet(val_df,   f"{prefix}/val.parquet")

        # Build + upload metadata
        meta = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "feature_columns": list(X_train.columns),
            "n_features": X_train.shape[1],
            "fraud_rate_train": float(y_train.mean()),
            "fraud_rate_val": float(y_val.mean()),
        }
        if source_path:
            meta["source_hash"] = _file_md5(source_path)
            meta["source_path"] = str(source_path)

        self._upload_json(meta, f"{prefix}/metadata.json")

        # Update the latest pointer
        self._upload_json({"latest_version": version, "updated_at": meta["created_at"]},
                          LATEST_POINTER)

        logger.info(
            "Offline features saved → s3://%s/%s  (train=%d  val=%d)",
            self.bucket, prefix, len(train_df), len(val_df),
        )
        return version

    def load_offline_features(
        self, version: Optional[str] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load train/val splits from the offline store.

        If version is None, loads the latest version.
        Returns (X_train, X_val, y_train, y_val).
        """
        version = version or self.latest_version()
        prefix = f"{PREFIX_OFFLINE}/{version}"

        logger.info("Loading offline features from s3://%s/%s …", self.bucket, prefix)

        train_df = self._download_parquet(f"{prefix}/train.parquet")
        val_df   = self._download_parquet(f"{prefix}/val.parquet")

        y_train = train_df.pop("__target__").rename("Class")
        y_val   = val_df.pop("__target__").rename("Class")

        logger.info(
            "Loaded features: train=%d × %d  val=%d × %d",
            *train_df.shape, *val_df.shape,
        )
        return train_df, val_df, y_train, y_val

    def latest_version(self) -> str:
        """Return the key of the most recently saved feature version."""
        try:
            data = self._download_json(LATEST_POINTER)
            return data["latest_version"]
        except (ClientError, KeyError) as e:
            raise RuntimeError(
                f"No feature versions found in s3://{self.bucket}. "
                "Run the feature pipeline first."
            ) from e

    def list_versions(self) -> list[dict]:
        """List all available offline feature versions with metadata."""
        paginator = self._s3.get_paginator("list_objects_v2")
        versions = []
        # Trailing slash required so S3 lists one level *inside* the prefix
        offline_prefix = f"{PREFIX_OFFLINE}/"
        for page in paginator.paginate(Bucket=self.bucket, Prefix=offline_prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                version_key = cp["Prefix"].rstrip("/").split("/")[-1]
                try:
                    meta = self._download_json(
                        f"{PREFIX_OFFLINE}/{version_key}/metadata.json"
                    )
                    versions.append(meta)
                except ClientError:
                    versions.append({"version": version_key})
        return sorted(versions, key=lambda v: v.get("created_at", ""), reverse=True)

    # ------------------------------------------------------------------
    # Pipeline artifacts (scaler + feature metadata)
    # ------------------------------------------------------------------

    def save_artifacts(
        self,
        scaler,
        feature_meta: dict,
        version: Optional[str] = None,
        local_artifacts_dir: Optional[str] = None,
    ) -> str:
        """
        Upload scaler.joblib and selected_features.json to S3.

        If local_artifacts_dir is provided, also writes local copies (for
        Docker image bake-in or CI caching).

        Returns the version key used.
        """
        version = version or self.latest_version()
        prefix = f"{PREFIX_ARTIFACTS}/{version}"

        # Serialise scaler to bytes in memory (avoid temp file race conditions)
        buf = io.BytesIO()
        joblib.dump(scaler, buf)
        buf.seek(0)
        self._s3.upload_fileobj(buf, self.bucket, f"{prefix}/scaler.joblib")
        logger.info("Scaler uploaded → s3://%s/%s/scaler.joblib", self.bucket, prefix)

        # Upload feature metadata
        self._upload_json(feature_meta, f"{prefix}/selected_features.json")
        logger.info("Feature meta uploaded → s3://%s/%s/selected_features.json",
                    self.bucket, prefix)

        # Also write a "latest artifacts" pointer
        self._upload_json(
            {"latest_version": version,
             "scaler_key": f"{prefix}/scaler.joblib",
             "features_key": f"{prefix}/selected_features.json"},
            "features/latest_artifacts.json",
        )

        # Optionally persist locally too (for Docker bake-in)
        if local_artifacts_dir:
            _save_artifacts_locally(scaler, feature_meta, local_artifacts_dir)

        return version

    def load_artifacts(
        self,
        version: Optional[str] = None,
        local_cache_dir: Optional[str] = None,
    ) -> tuple:
        """
        Download scaler and feature_meta from S3.

        If local_cache_dir is provided, files are written to disk as a cache
        so subsequent calls skip the S3 download.

        Returns (scaler, feature_meta_dict).
        """
        # Try cache first
        if local_cache_dir:
            cached = _load_artifacts_from_cache(local_cache_dir)
            if cached:
                logger.info("Loaded artifacts from local cache %s", local_cache_dir)
                return cached

        # Resolve version
        if version is None:
            try:
                ptr = self._download_json("features/latest_artifacts.json")
                scaler_key   = ptr["scaler_key"]
                features_key = ptr["features_key"]
            except ClientError:
                version = self.latest_version()
                prefix = f"{PREFIX_ARTIFACTS}/{version}"
                scaler_key   = f"{prefix}/scaler.joblib"
                features_key = f"{prefix}/selected_features.json"
        else:
            prefix = f"{PREFIX_ARTIFACTS}/{version}"
            scaler_key   = f"{prefix}/scaler.joblib"
            features_key = f"{prefix}/selected_features.json"

        logger.info("Downloading artifacts from s3://%s/%s …", self.bucket, scaler_key)

        # Scaler
        buf = io.BytesIO()
        self._s3.download_fileobj(self.bucket, scaler_key, buf)
        buf.seek(0)
        scaler = joblib.load(buf)

        # Feature metadata
        feature_meta = self._download_json(features_key)

        # Persist to local cache
        if local_cache_dir:
            _save_artifacts_locally(scaler, feature_meta, local_cache_dir)
            logger.info("Artifacts cached to %s", local_cache_dir)

        return scaler, feature_meta

    # ------------------------------------------------------------------
    # Convenience: check bucket connectivity
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the bucket is accessible."""
        try:
            self._s3.head_bucket(Bucket=self.bucket)
            return True
        except ClientError:
            return False

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _key_exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def _upload_parquet(self, df: pd.DataFrame, key: str) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=True, compression="snappy")
        buf.seek(0)
        self._s3.upload_fileobj(buf, self.bucket, key)

    def _download_parquet(self, key: str) -> pd.DataFrame:
        buf = io.BytesIO()
        self._s3.download_fileobj(self.bucket, key, buf)
        buf.seek(0)
        return pd.read_parquet(buf)

    def _upload_json(self, data: dict, key: str) -> None:
        body = json.dumps(data, indent=2, default=str).encode()
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=body,
                            ContentType="application/json")

    def _download_json(self, key: str) -> dict:
        response = self._s3.get_object(Bucket=self.bucket, Key=key)
        return json.loads(response["Body"].read())


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_version() -> str:
    """Generate a timestamp-based version string, e.g. v_20240115_143022."""
    return datetime.now(timezone.utc).strftime("v_%Y%m%d_%H%M%S")


def _file_md5(path: str, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 hex digest of a file for lineage tracking."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_artifacts_locally(scaler, feature_meta: dict, directory: str) -> None:
    """Write scaler.joblib and selected_features.json to a local directory."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, os.path.join(directory, "scaler.joblib"))
    with open(os.path.join(directory, "selected_features.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)


def _load_artifacts_from_cache(directory: str):
    """Try to load artifacts from local cache; return None if incomplete."""
    scaler_path   = Path(directory) / "scaler.joblib"
    features_path = Path(directory) / "selected_features.json"
    if scaler_path.exists() and features_path.exists():
        scaler = joblib.load(scaler_path)
        with open(features_path) as f:
            meta = json.load(f)
        return scaler, meta
    return None


# ---------------------------------------------------------------------------
# Bucket bootstrap helper (called from scripts/upload_data.py)
# ---------------------------------------------------------------------------

def ensure_bucket(bucket: str, region: str = "us-east-1") -> None:
    """
    Create the S3 bucket if it doesn't already exist.
    Sets versioning + server-side encryption as sensible defaults.
    """
    s3 = boto3.client("s3", region_name=region)
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        logger.info("Created bucket s3://%s", bucket)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            logger.info("Bucket s3://%s already exists", bucket)
        else:
            raise

    # Enable versioning
    s3.put_bucket_versioning(
        Bucket=bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )

    # Enable SSE-S3 encryption
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                }
            }]
        },
    )
    logger.info("Bucket s3://%s configured (versioning + AES256 encryption)", bucket)
