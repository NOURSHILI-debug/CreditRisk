from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.main import create_app

MODEL_HASH = "75fd9738cb1a709b47f64c89126e8c5344d3a4d6368ef16c6870c960360a8497"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@pytest.fixture(scope="module")
def application_ids(project_root: Path) -> np.ndarray:
    data_path = project_root / "data" / "raw" / "application_train.csv"
    if not data_path.exists():
        pytest.skip("Raw application_train.csv is missing.")
    return pd.read_csv(data_path, usecols=["SK_ID_CURR"])[
        "SK_ID_CURR"
    ].to_numpy()


@pytest.fixture(scope="module")
def api_client(
    project_root: Path, model_dir: Path
) -> TestClient:
    data_path = project_root / "data" / "raw" / "application_train.csv"
    if not data_path.exists():
        pytest.skip("Raw application_train.csv is missing.")
    settings = Settings(
        model_dir=model_dir,
        application_data_path=data_path,
        cors_origins=("http://localhost:8501",),
        log_level="CRITICAL",
    )
    with TestClient(create_app(settings)) as client:
        yield client


def _features(client: TestClient, application_id: int) -> dict[str, Any]:
    state = client.app.state.container
    row = state.repository.get_by_id(application_id)
    assert row is not None
    return _jsonable(
        {
            key: value
            for key, value in row.to_dict().items()
            if key != "SK_ID_CURR"
        }
    )


def test_health_models_and_artifacts(api_client: TestClient) -> None:
    health = api_client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["data_available"] is True
    assert payload["model_version"] == "baseline-v2-local"
    assert payload["model_hash"] == MODEL_HASH

    models = api_client.get("/models")
    assert models.status_code == 200
    info = models.json()
    assert info["model_name"] == "XGBoost"
    assert info["model_version"] == "baseline-v2-local"
    assert info["model_hash"] == MODEL_HASH
    assert info["training_date"]
    assert info["thresholds"]["decision_threshold"] == 0.5
    assert info["thresholds"]["provisional"] is True
    assert "low_upper" in info["thresholds"]["risk_bands"]

    evaluation = api_client.get("/model/evaluation")
    assert evaluation.status_code == 200
    results = evaluation.json()
    assert results["threshold"] == 0.5
    assert "roc_auc" in results["metrics"]["test"]

    importance = api_client.get("/model/global-importance")
    assert importance.status_code == 200
    importance_payload = importance.json()
    assert importance_payload["method"] == "tree_shap"
    assert importance_payload["features"][0]["feature"] == "EXT_SOURCE_3"
    assert len(importance_payload["features"]) == 120
    assert "model_dir" not in importance_payload
    assert "source_data" not in importance_payload
    assert "source_predictions" not in importance_payload


def test_prediction_by_id_matches_artifact(api_client: TestClient) -> None:
    response = api_client.post("/predict", json={"application_id": 172753})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["application_id"] == 172753
    assert payload["probability"] == pytest.approx(
        0.9589290618896484, abs=1e-12
    )
    assert payload["risk_level"] == "HIGH"
    assert payload["model_name"] == "XGBoost"
    assert payload["model_version"] == "baseline-v2-local"
    assert payload["explanation"] is None


def test_prediction_by_features_matches_id_path(api_client: TestClient) -> None:
    features = _features(api_client, 120321)
    response = api_client.post("/predict", json={"features": features})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["application_id"] is None
    assert payload["probability"] == pytest.approx(
        0.4915030896663666, abs=1e-12
    )

    by_id = api_client.post("/predict", json={"application_id": 120321})
    assert by_id.status_code == 200
    assert by_id.json()["probability"] == payload["probability"]


def test_by_features_matches_by_id_for_50_random_applicants(
    api_client: TestClient, application_ids: np.ndarray
) -> None:
    sampled = np.random.default_rng(42).choice(
        application_ids, size=50, replace=False
    )
    for raw_id in sampled:
        application_id = int(raw_id)
        by_id = api_client.post(
            "/predict", json={"application_id": application_id}
        )
        by_features = api_client.post(
            "/predict",
            json={"features": _features(api_client, application_id)},
        )
        assert by_id.status_code == 200, application_id
        assert by_features.status_code == 200, application_id
        assert by_features.json()["probability"] == pytest.approx(
            by_id.json()["probability"], rel=0.0, abs=1e-9
        ), application_id


def test_by_features_matches_by_id_with_explanation_for_50_random_applicants(
    api_client: TestClient, application_ids: np.ndarray
) -> None:
    sampled = np.random.default_rng(42).choice(
        application_ids, size=50, replace=False
    )
    for raw_id in sampled:
        application_id = int(raw_id)
        by_id = api_client.post(
            "/predict",
            params={"include_explanation": True},
            json={"application_id": application_id},
        )
        by_features = api_client.post(
            "/predict",
            params={"include_explanation": True},
            json={"features": _features(api_client, application_id)},
        )
        assert by_id.status_code == 200, application_id
        assert by_features.status_code == 200, application_id
        assert by_features.json()["probability"] == pytest.approx(
            by_id.json()["probability"], rel=0.0, abs=1e-9
        ), application_id


def test_explicit_nulls_match_by_id_path(api_client: TestClient) -> None:
    application_id = 120321
    by_id = api_client.post(
        "/predict", json={"application_id": application_id}
    )
    features = _features(api_client, application_id)
    for column in (
        "EMERGENCYSTATE_MODE",
        "WALLSMATERIAL_MODE",
        "HOUSETYPE_MODE",
        "FONDKAPREMONT_MODE",
    ):
        features[column] = None

    by_features = api_client.post("/predict", json={"features": features})

    assert by_id.status_code == 200
    assert by_features.status_code == 200
    assert by_features.json()["probability"] == pytest.approx(
        by_id.json()["probability"], rel=0.0, abs=1e-9
    )


def test_explanation_preserves_probability_and_additivity(
    api_client: TestClient,
) -> None:
    plain = api_client.get("/applications/339432/prediction").json()
    explained_response = api_client.get(
        "/applications/339432/prediction",
        params={"include_explanation": True},
    )
    assert explained_response.status_code == 200
    explained = explained_response.json()
    assert explained["status"] == "success"
    assert explained["probability"] == plain["probability"]
    assert explained["probability"] == pytest.approx(
        0.00021285100956447423, abs=1e-12
    )
    explanation = explained["explanation"]
    assert explanation["method"] == "tree_shap"
    assert explanation["output_space"] == "log_odds"
    assert explanation["application_id"] == 339432
    assert explanation["probability"] == explained["probability"]
    assert math.isclose(
        explanation["base_value"] + explanation["contribution_sum"],
        explanation["prediction_margin"],
        rel_tol=0.0,
        abs_tol=1e-5,
    )
    assert explanation["top_features"]


def test_api_input_errors(api_client: TestClient) -> None:
    missing = api_client.get("/applications/999999999/prediction")
    assert missing.status_code == 404
    assert missing.json()["status"] == "error"
    assert missing.json()["probability"] is None

    both = api_client.post(
        "/predict",
        json={"application_id": 172753, "features": {"AMT_CREDIT": 100}},
    )
    assert both.status_code == 422
    assert both.json()["status"] == "error"

    neither = api_client.post("/predict", json={})
    assert neither.status_code == 422
    assert neither.json()["status"] == "error"
