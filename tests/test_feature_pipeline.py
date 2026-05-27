"""Unit tests for the feature engineering pipeline."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_pipeline.feature_engineering import (
    engineer_features,
    split_data,
    fit_scaler,
    apply_scaler,
    select_features_l1,
    transform_input,
    run_feature_pipeline,
    TARGET,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _make_df(n: int = 500, fraud_frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Synthetic dataframe that mirrors the real CSV schema."""
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_frac))
    n_legit = n - n_fraud

    rows = []
    for label, count in [(0, n_legit), (1, n_fraud)]:
        block = {
            "Time": rng.uniform(0, 172800, count),
            "Amount": rng.exponential(50, count),
        }
        for i in range(1, 29):
            block[f"V{i}"] = rng.normal(0, 1, count)
        block["Class"] = label
        rows.append(pd.DataFrame(block))

    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed)
    return df.reset_index(drop=True)


@pytest.fixture
def sample_df():
    return _make_df(500)


@pytest.fixture
def tiny_df():
    return _make_df(200)


# ------------------------------------------------------------------
# engineer_features
# ------------------------------------------------------------------

class TestEngineerFeatures:
    def test_log_amount_created(self, sample_df):
        result = engineer_features(sample_df)
        assert "log_amount" in result.columns

    def test_hour_created(self, sample_df):
        result = engineer_features(sample_df)
        assert "hour" in result.columns

    def test_amount_and_time_dropped(self, sample_df):
        result = engineer_features(sample_df)
        assert "Amount" not in result.columns
        assert "Time" not in result.columns

    def test_log_amount_non_negative(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["log_amount"] >= 0).all()

    def test_hour_in_range(self, sample_df):
        result = engineer_features(sample_df)
        assert result["hour"].between(0, 24).all()

    def test_no_rows_dropped(self, sample_df):
        result = engineer_features(sample_df)
        assert len(result) == len(sample_df)


# ------------------------------------------------------------------
# split_data
# ------------------------------------------------------------------

class TestSplitData:
    def test_val_size(self, sample_df):
        df = engineer_features(sample_df)
        X_tr, X_val, y_tr, y_val = split_data(df, val_size=0.2)
        total = len(X_tr) + len(X_val)
        assert total == len(df)
        assert abs(len(X_val) / total - 0.2) < 0.05

    def test_stratification(self, sample_df):
        df = engineer_features(sample_df)
        _, _, y_tr, y_val = split_data(df)
        # Fraud rate should be similar in both splits
        assert abs(y_tr.mean() - y_val.mean()) < 0.05

    def test_no_overlap(self, sample_df):
        df = engineer_features(sample_df)
        X_tr, X_val, _, _ = split_data(df)
        assert len(set(X_tr.index) & set(X_val.index)) == 0


# ------------------------------------------------------------------
# fit_scaler / apply_scaler
# ------------------------------------------------------------------

class TestScaler:
    def test_fit_returns_scaler_and_df(self, sample_df):
        df = engineer_features(sample_df)
        X_tr, _, _, _ = split_data(df)
        scaler, X_scaled = fit_scaler(X_tr)
        assert X_scaled.shape == X_tr.shape
        assert list(X_scaled.columns) == list(X_tr.columns)

    def test_scaled_mean_near_zero(self, sample_df):
        df = engineer_features(sample_df)
        X_tr, _, _, _ = split_data(df)
        _, X_scaled = fit_scaler(X_tr)
        # Training set column means should be close to 0
        assert (X_scaled.mean().abs() < 0.1).all()

    def test_apply_scaler_same_shape(self, sample_df):
        df = engineer_features(sample_df)
        X_tr, X_val, _, _ = split_data(df)
        scaler, _ = fit_scaler(X_tr)
        X_val_scaled = apply_scaler(scaler, X_val)
        assert X_val_scaled.shape == X_val.shape


# ------------------------------------------------------------------
# select_features_l1
# ------------------------------------------------------------------

class TestSelectFeaturesL1:
    def test_returns_subset(self, sample_df):
        df = engineer_features(sample_df)
        X_tr, _, y_tr, _ = split_data(df)
        scaler, X_scaled = fit_scaler(X_tr)
        selected = select_features_l1(X_scaled, y_tr, C=0.1)
        assert len(selected) <= X_scaled.shape[1]
        assert len(selected) >= 1

    def test_selected_are_valid_cols(self, sample_df):
        df = engineer_features(sample_df)
        X_tr, _, y_tr, _ = split_data(df)
        _, X_scaled = fit_scaler(X_tr)
        selected = select_features_l1(X_scaled, y_tr, C=0.1)
        assert all(c in X_scaled.columns for c in selected)


# ------------------------------------------------------------------
# transform_input
# ------------------------------------------------------------------

class TestTransformInput:
    def test_output_shape(self, sample_df):
        df = engineer_features(sample_df)
        X_tr, _, y_tr, _ = split_data(df)
        scaler, X_scaled = fit_scaler(X_tr)
        selected = select_features_l1(X_scaled, y_tr, C=0.5)
        all_features = list(X_tr.columns)

        raw = {
            "Time": 3600.0,
            "Amount": 100.0,
            **{f"V{i}": 0.0 for i in range(1, 29)},
        }
        result = transform_input(raw, scaler, selected, all_features)
        assert result.shape == (1, len(selected))


# ------------------------------------------------------------------
# run_feature_pipeline (integration)
# ------------------------------------------------------------------

class TestRunFeaturePipeline:
    def test_pipeline_runs_and_returns_keys(self, sample_df, tmp_path):
        # Write sample data to temp CSV
        csv_path = tmp_path / "sample.csv"
        sample_df.to_csv(csv_path, index=False)

        result = run_feature_pipeline(str(csv_path), artifacts_dir=str(tmp_path))

        for key in ["X_train", "X_val", "y_train", "y_val",
                    "scaler", "selected_features", "all_features"]:
            assert key in result, f"Missing key: {key}"

    def test_scaler_artifact_saved(self, sample_df, tmp_path):
        csv_path = tmp_path / "sample.csv"
        sample_df.to_csv(csv_path, index=False)
        result = run_feature_pipeline(str(csv_path), artifacts_dir=str(tmp_path))
        assert Path(result["scaler_path"]).exists()

    def test_features_artifact_saved(self, sample_df, tmp_path):
        csv_path = tmp_path / "sample.csv"
        sample_df.to_csv(csv_path, index=False)
        result = run_feature_pipeline(str(csv_path), artifacts_dir=str(tmp_path))
        assert Path(result["features_path"]).exists()
        with open(result["features_path"]) as f:
            meta = json.load(f)
        assert "selected_features" in meta
        assert "all_features" in meta

    def test_no_data_leakage(self, sample_df, tmp_path):
        """Validation index should not appear in training index."""
        csv_path = tmp_path / "sample.csv"
        sample_df.to_csv(csv_path, index=False)
        result = run_feature_pipeline(str(csv_path), artifacts_dir=str(tmp_path))
        train_idx = set(result["X_train"].index)
        val_idx = set(result["X_val"].index)
        assert len(train_idx & val_idx) == 0
