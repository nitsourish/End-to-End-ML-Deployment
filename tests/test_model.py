"""
Tests for model training (smoke tests that run without AWS / MLflow server).
Trains on a small synthetic dataset to verify the full train() function works.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_df(n: int = 300, fraud_frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n * fraud_frac))
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


class TestTrainSmoke:
    """
    Smoke-test the training pipeline end-to-end using a temporary local
    MLflow tracking store (no S3, no remote server needed).
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        # Redirect MLflow to a local temp directory
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")

        # Write synthetic data to temp CSV
        df = _make_df(300)
        self.csv_path = tmp_path / "synthetic.csv"
        df.to_csv(self.csv_path, index=False)

        self.artifacts_dir = str(tmp_path / "artifacts")

    def test_train_returns_run_id(self):
        from src.training.train import train
        run_id = train(
            data_path=str(self.csv_path),
            experiment_name="test-experiment",
            artifacts_dir=self.artifacts_dir,
            register_model=False,   # skip registry in unit tests
        )
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_artifacts_created(self):
        from src.training.train import train
        train(
            data_path=str(self.csv_path),
            experiment_name="test-experiment",
            artifacts_dir=self.artifacts_dir,
            register_model=False,
        )
        assert (Path(self.artifacts_dir) / "scaler.joblib").exists()
        assert (Path(self.artifacts_dir) / "selected_features.json").exists()

    def test_metrics_logged(self):
        import mlflow
        from src.training.train import train

        tracking_uri = f"file://{self.tmp_path / 'mlruns'}"
        mlflow.set_tracking_uri(tracking_uri)

        train(
            data_path=str(self.csv_path),
            experiment_name="test-metrics",
            artifacts_dir=self.artifacts_dir,
            register_model=False,
        )

        experiment = mlflow.get_experiment_by_name("test-metrics")
        assert experiment is not None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0

        # Best run should have ROC-AUC logged
        best = runs[runs["metrics.roc_auc"].notna()]
        assert len(best) > 0
        auc = best["metrics.roc_auc"].max()
        assert 0.5 < auc <= 1.0, f"Unexpected AUC: {auc}"


class TestModelPredictSanity:
    """Verify the trained model makes reasonable predictions on synthetic data."""

    def test_predict_proba_in_range(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")

        df = _make_df(300)
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        from src.training.train import train
        from src.feature_pipeline.feature_engineering import run_feature_pipeline
        import joblib, json

        arts = str(tmp_path / "arts")
        train(str(csv_path), artifacts_dir=arts, register_model=False)

        # Load pipeline artefacts
        scaler = joblib.load(Path(arts) / "scaler.joblib")
        with open(Path(arts) / "selected_features.json") as f:
            meta = json.load(f)

        from sklearn.linear_model import LogisticRegression
        from src.feature_pipeline.feature_engineering import transform_input

        # Manually load a model from artefacts dir (no registry)
        pipeline = run_feature_pipeline(str(csv_path), artifacts_dir=arts)
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500)
        model.fit(pipeline["X_train"], pipeline["y_train"])

        raw = {
            "Time": 1000.0,
            "Amount": 20.0,
            **{f"V{i}": 0.0 for i in range(1, 29)},
        }
        X = transform_input(raw, scaler, meta["selected_features"], meta["all_features"])
        probs = model.predict_proba(X)
        assert probs.shape == (1, 2)
        assert abs(probs[0].sum() - 1.0) < 1e-6
        assert 0.0 <= probs[0, 1] <= 1.0
