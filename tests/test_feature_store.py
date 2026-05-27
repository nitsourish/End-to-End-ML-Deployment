"""
Tests for S3FeatureStore — use moto to mock AWS S3 so no real credentials
or bucket are needed.
"""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- Try to import moto; skip whole module if not installed ----------------
boto3 = pytest.importorskip("boto3", reason="boto3 not installed")
try:
    from moto import mock_aws
    MOTO_AVAILABLE = True
except ImportError:
    MOTO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MOTO_AVAILABLE,
    reason="moto not installed — install with: pip install moto[s3]",
)

import boto3 as _boto3
from src.feature_pipeline.feature_store import (
    S3FeatureStore,
    _make_version,
    _file_md5,
    _save_artifacts_locally,
    _load_artifacts_from_cache,
    ensure_bucket,
    LATEST_POINTER,
    PREFIX_RAW,
    PREFIX_OFFLINE,
    PREFIX_ARTIFACTS,
)

BUCKET = "test-fraud-feature-store"
REGION = "us-east-1"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_splits(n: int = 200, n_features: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    feature_cols = [f"V{i}" for i in range(1, n_features - 1)] + ["log_amount", "hour"]
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=feature_cols)
    y = pd.Series((rng.random(n) > 0.9).astype(int), name="Class")
    n_train = int(n * 0.8)
    return (
        X.iloc[:n_train].copy(),
        X.iloc[n_train:].copy(),
        y.iloc[:n_train].copy(),
        y.iloc[n_train:].copy(),
    )


def _make_scaler():
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    sc = StandardScaler()
    sc.fit(rng.normal(size=(100, 5)))
    return sc


def _make_feature_meta(n_features: int = 10):
    all_f = [f"V{i}" for i in range(1, n_features - 1)] + ["log_amount", "hour"]
    return {"all_features": all_f, "selected_features": all_f[:8]}


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def s3_bucket():
    """Spin up a moto-mocked S3 bucket and yield the bucket name."""
    with mock_aws():
        client = _boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        yield BUCKET


@pytest.fixture
def fs(s3_bucket):
    """S3FeatureStore pointing at the mocked bucket."""
    return S3FeatureStore(bucket=s3_bucket, region=REGION)


# ------------------------------------------------------------------
# ensure_bucket
# ------------------------------------------------------------------

class TestEnsureBucket:
    def test_creates_bucket(self):
        with mock_aws():
            ensure_bucket("brand-new-bucket", region=REGION)
            s3 = _boto3.client("s3", region_name=REGION)
            buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
            assert "brand-new-bucket" in buckets

    def test_idempotent_on_existing_bucket(self):
        with mock_aws():
            ensure_bucket("dup-bucket", region=REGION)
            ensure_bucket("dup-bucket", region=REGION)   # should not raise


# ------------------------------------------------------------------
# ping
# ------------------------------------------------------------------

class TestPing:
    def test_ping_returns_true_for_existing_bucket(self, fs):
        assert fs.ping() is True

    def test_ping_returns_false_for_missing_bucket(self):
        with mock_aws():
            fs2 = S3FeatureStore("does-not-exist", region=REGION)
            assert fs2.ping() is False


# ------------------------------------------------------------------
# upload / download raw data
# ------------------------------------------------------------------

class TestRawData:
    def test_upload_and_exists(self, fs, tmp_path):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)
        uri = fs.upload_raw_data(str(csv))
        assert uri.startswith(f"s3://{BUCKET}/{PREFIX_RAW}/")

    def test_upload_idempotent(self, fs, tmp_path):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
        uri1 = fs.upload_raw_data(str(csv))
        uri2 = fs.upload_raw_data(str(csv))   # should skip, return same URI
        assert uri1 == uri2

    def test_upload_overwrite(self, fs, tmp_path):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
        fs.upload_raw_data(str(csv))
        # overwrite=True must not raise
        fs.upload_raw_data(str(csv), overwrite=True)

    def test_download_roundtrip(self, fs, tmp_path):
        src = tmp_path / "src.csv"
        dst = tmp_path / "dst.csv"
        pd.DataFrame({"x": [10, 20, 30]}).to_csv(src, index=False)
        # upload resolves key as raw/src.csv — pass the same key to download
        s3_key = f"{PREFIX_RAW}/src.csv"
        fs.upload_raw_data(str(src), s3_key=s3_key)
        fs.download_raw_data(str(dst), s3_key=s3_key)
        assert dst.exists()
        df = pd.read_csv(dst)
        assert list(df["x"]) == [10, 20, 30]


# ------------------------------------------------------------------
# offline feature store (save / load parquet)
# ------------------------------------------------------------------

class TestOfflineFeatureStore:
    def test_save_returns_version_string(self, fs):
        X_tr, X_val, y_tr, y_val = _make_splits()
        version = fs.save_offline_features(X_tr, X_val, y_tr, y_val)
        assert isinstance(version, str)
        assert version.startswith("v_")

    def test_latest_version_matches_saved(self, fs):
        X_tr, X_val, y_tr, y_val = _make_splits()
        version = fs.save_offline_features(X_tr, X_val, y_tr, y_val)
        assert fs.latest_version() == version

    def test_latest_version_tracks_most_recent(self, fs):
        X_tr, X_val, y_tr, y_val = _make_splits()
        fs.save_offline_features(X_tr, X_val, y_tr, y_val, version="v_20240101_000000")
        fs.save_offline_features(X_tr, X_val, y_tr, y_val, version="v_20240102_000000")
        assert fs.latest_version() == "v_20240102_000000"

    def test_load_returns_correct_shapes(self, fs):
        X_tr, X_val, y_tr, y_val = _make_splits(n=200)
        version = fs.save_offline_features(X_tr, X_val, y_tr, y_val)
        X_tr2, X_val2, y_tr2, y_val2 = fs.load_offline_features(version)
        assert X_tr2.shape == X_tr.shape
        assert X_val2.shape == X_val.shape
        assert len(y_tr2) == len(y_tr)
        assert len(y_val2) == len(y_val)

    def test_load_no_target_leakage(self, fs):
        """__target__ column must not appear in X after loading."""
        X_tr, X_val, y_tr, y_val = _make_splits()
        version = fs.save_offline_features(X_tr, X_val, y_tr, y_val)
        X_tr2, X_val2, _, _ = fs.load_offline_features(version)
        assert "__target__" not in X_tr2.columns
        assert "__target__" not in X_val2.columns

    def test_load_fraud_rate_preserved(self, fs):
        X_tr, X_val, y_tr, y_val = _make_splits(n=500)
        version = fs.save_offline_features(X_tr, X_val, y_tr, y_val)
        _, _, y_tr2, y_val2 = fs.load_offline_features(version)
        assert abs(y_tr2.mean() - y_tr.mean()) < 0.01
        assert abs(y_val2.mean() - y_val.mean()) < 0.01

    def test_list_versions_returns_all(self, fs):
        X_tr, X_val, y_tr, y_val = _make_splits()
        fs.save_offline_features(X_tr, X_val, y_tr, y_val, version="v_20240101_000000")
        fs.save_offline_features(X_tr, X_val, y_tr, y_val, version="v_20240102_000000")
        versions = fs.list_versions()
        keys = [v["version"] for v in versions]
        assert "v_20240101_000000" in keys
        assert "v_20240102_000000" in keys

    def test_no_version_raises_if_empty(self):
        with mock_aws():
            _boto3.client("s3", region_name=REGION).create_bucket(Bucket="empty-bucket")
            fs2 = S3FeatureStore("empty-bucket", region=REGION)
            with pytest.raises(RuntimeError, match="No feature versions found"):
                fs2.latest_version()


# ------------------------------------------------------------------
# artifacts (scaler + feature meta)
# ------------------------------------------------------------------

class TestArtifacts:
    def test_save_artifacts_succeeds(self, fs):
        scaler = _make_scaler()
        meta = _make_feature_meta()
        version = fs.save_artifacts(scaler, meta, version="v_test")
        assert version == "v_test"

    def test_load_artifacts_returns_scaler_and_meta(self, fs):
        scaler = _make_scaler()
        meta = _make_feature_meta()
        version = fs.save_artifacts(scaler, meta, version="v_test")
        scaler2, meta2 = fs.load_artifacts(version=version)
        assert meta2["selected_features"] == meta["selected_features"]
        assert meta2["all_features"] == meta["all_features"]
        # Scaler should be functional
        import numpy as np
        X = np.random.default_rng(0).normal(size=(10, 5))
        np.testing.assert_allclose(
            scaler.transform(X),
            scaler2.transform(X),
        )

    def test_load_artifacts_via_latest_pointer(self, fs):
        """load_artifacts(version=None) should use the latest_artifacts pointer."""
        scaler = _make_scaler()
        meta = _make_feature_meta()
        fs.save_artifacts(scaler, meta, version="v_abc")
        scaler2, meta2 = fs.load_artifacts()   # no version — uses pointer
        assert meta2["selected_features"] == meta["selected_features"]

    def test_save_artifacts_writes_local_cache(self, fs, tmp_path):
        scaler = _make_scaler()
        meta = _make_feature_meta()
        fs.save_artifacts(scaler, meta, version="v_test",
                          local_artifacts_dir=str(tmp_path))
        assert (tmp_path / "scaler.joblib").exists()
        assert (tmp_path / "selected_features.json").exists()

    def test_load_artifacts_uses_local_cache(self, fs, tmp_path):
        """If cache exists, load_artifacts should not hit S3."""
        scaler = _make_scaler()
        meta = _make_feature_meta()
        # Manually seed the cache
        _save_artifacts_locally(scaler, meta, str(tmp_path))

        # Point to a non-existent bucket — should still work via cache
        fs_bad = S3FeatureStore("does-not-exist", region=REGION)
        with mock_aws():   # no real bucket — cache must win
            scaler2, meta2 = fs_bad.load_artifacts(local_cache_dir=str(tmp_path))
        assert meta2["selected_features"] == meta["selected_features"]


# ------------------------------------------------------------------
# Local cache helpers
# ------------------------------------------------------------------

class TestLocalCacheHelpers:
    def test_save_and_load_roundtrip(self, tmp_path):
        scaler = _make_scaler()
        meta = _make_feature_meta()
        _save_artifacts_locally(scaler, meta, str(tmp_path))
        result = _load_artifacts_from_cache(str(tmp_path))
        assert result is not None
        _, loaded_meta = result
        assert loaded_meta["selected_features"] == meta["selected_features"]

    def test_load_returns_none_if_missing(self, tmp_path):
        assert _load_artifacts_from_cache(str(tmp_path)) is None

    def test_load_returns_none_if_partial(self, tmp_path):
        # Only scaler exists, no features json
        scaler = _make_scaler()
        import joblib
        joblib.dump(scaler, tmp_path / "scaler.joblib")
        assert _load_artifacts_from_cache(str(tmp_path)) is None


# ------------------------------------------------------------------
# Misc helpers
# ------------------------------------------------------------------

class TestHelpers:
    def test_make_version_format(self):
        v = _make_version()
        assert v.startswith("v_")
        assert len(v) == len("v_20240101_120000")

    def test_file_md5_is_stable(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h1 = _file_md5(str(f))
        h2 = _file_md5(str(f))
        assert h1 == h2
        assert len(h1) == 32

    def test_file_md5_differs_for_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert _file_md5(str(f1)) != _file_md5(str(f2))
