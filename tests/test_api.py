"""
Integration tests for the FastAPI serving layer.

These tests patch the global model state so no real MLflow / trained model
is required — allowing them to run in CI without AWS credentials.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------
# Build a minimal trained model + pipeline artefacts for the tests
# ------------------------------------------------------------------

def _build_fake_state():
    """Return a realistic _state dict with a tiny trained model."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n_features = 10  # keep it small for speed
    feature_names = [f"V{i}" for i in range(1, n_features - 1)] + ["log_amount", "hour"]

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, n_features))
    y = (rng.random(200) > 0.9).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", max_iter=200)
    model.fit(X_scaled, y)

    return {
        "model": model,
        "scaler": scaler,
        "selected_features": feature_names,
        "all_features": feature_names,
        "model_version": "test-v1",
        "loaded_at": "2024-01-01T00:00:00Z",
    }


_FAKE_STATE = _build_fake_state()


def _mock_transform_input(raw, scaler, selected_features, all_features):
    """Minimal transform: just returns random 2-D array of correct shape."""
    n = len(selected_features)
    rng = np.random.default_rng(abs(hash(str(raw))) % (2**31))
    return rng.normal(size=(1, n))


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def client():
    """
    TestClient with the global _state patched so no model loading happens.
    The lifespan startup is short-circuited by pre-seeding _state.
    """
    from src.serving import app as app_module

    # Pre-seed state so the lifespan startup finds model already loaded
    app_module._state.update(_FAKE_STATE)

    with (
        patch("src.serving.app._load_model_and_artifacts", return_value=None),
        patch("src.serving.app.transform_input", side_effect=_mock_transform_input),
    ):
        with TestClient(app_module.app) as c:
            yield c
        # Restore blank state after test
        for k in _FAKE_STATE:
            app_module._state[k] = None


# ------------------------------------------------------------------
# /health
# ------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_body(self, client):
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


# ------------------------------------------------------------------
# /ready
# ------------------------------------------------------------------

class TestReady:
    def test_ready_when_model_loaded(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_not_ready_when_model_missing(self):
        from src.serving import app as app_module
        # Temporarily clear model from state
        original = dict(app_module._state)
        for k in app_module._state:
            app_module._state[k] = None
        try:
            with patch("src.serving.app._load_model_and_artifacts", return_value=None):
                with TestClient(app_module.app) as c:
                    resp = c.get("/ready")
                    assert resp.status_code == 503
        finally:
            app_module._state.update(original)


# ------------------------------------------------------------------
# /model/info
# ------------------------------------------------------------------

class TestModelInfo:
    def test_returns_model_name(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "selected_features" in data


# ------------------------------------------------------------------
# /predict
# ------------------------------------------------------------------

def _sample_transaction(**overrides) -> dict:
    txn = {
        "Time": 3600.0,
        "Amount": 50.0,
        **{f"V{i}": float(i) * 0.1 for i in range(1, 29)},
    }
    txn.update(overrides)
    return txn


class TestPredict:
    def test_valid_transaction_200(self, client):
        resp = client.post("/predict", json=_sample_transaction())
        assert resp.status_code == 200

    def test_response_has_fraud_probability(self, client):
        resp = client.post("/predict", json=_sample_transaction())
        data = resp.json()
        assert "fraud_probability" in data
        prob = data["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_response_has_is_fraud(self, client):
        resp = client.post("/predict", json=_sample_transaction())
        assert "is_fraud" in resp.json()

    def test_response_has_threshold(self, client):
        resp = client.post("/predict", json=_sample_transaction())
        data = resp.json()
        assert "threshold" in data

    def test_negative_amount_rejected(self, client):
        resp = client.post("/predict", json=_sample_transaction(Amount=-10))
        assert resp.status_code == 422

    def test_missing_amount_rejected(self, client):
        txn = _sample_transaction()
        del txn["Amount"]
        resp = client.post("/predict", json=txn)
        assert resp.status_code == 422

    def test_missing_time_rejected(self, client):
        txn = _sample_transaction()
        del txn["Time"]
        resp = client.post("/predict", json=txn)
        assert resp.status_code == 422

    def test_is_fraud_consistent_with_probability(self, client):
        """is_fraud must be True iff fraud_probability >= threshold."""
        with patch("src.serving.app.FRAUD_THRESHOLD", 0.5):
            resp = client.post("/predict", json=_sample_transaction())
            data = resp.json()
            expected = data["fraud_probability"] >= data["threshold"]
            assert data["is_fraud"] == expected


# ------------------------------------------------------------------
# /predict/batch
# ------------------------------------------------------------------

class TestPredictBatch:
    def test_batch_of_three(self, client):
        batch = [_sample_transaction() for _ in range(3)]
        resp = client.post("/predict/batch", json=batch)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_too_large(self, client):
        batch = [_sample_transaction() for _ in range(1001)]
        resp = client.post("/predict/batch", json=batch)
        assert resp.status_code == 400

    def test_empty_batch(self, client):
        resp = client.post("/predict/batch", json=[])
        assert resp.status_code == 200
        assert resp.json()["count"] == 0
