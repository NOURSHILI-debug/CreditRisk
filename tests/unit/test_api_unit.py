from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from api.config import Settings
from api.features import FeatureValidator
from api.main import create_app
from api.repository import (
    ApplicationRepository,
    DuplicateApplicationIdError,
)
from dashboard.data_access import normalize_prediction_response
from processing.preprocessing import build_preprocessor


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


@pytest.fixture()
def tiny_artifacts(tmp_path: Path) -> tuple[Settings, pd.DataFrame]:
    data = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
            "TARGET": [0, 1, 0, 1, 0, 1],
            "AMT_CREDIT": [1000.0, 2200.0, 1500.0, 3200.0, 1800.0, 4200.0],
            "EXT_SOURCE_2": [0.12, 0.82, 0.35, 0.71, 0.28, 0.91],
            "CODE_GENDER": ["M", "F", "M", "F", "M", "F"],
        }
    )
    data_path = tmp_path / "applications.csv"
    data.to_csv(data_path, index=False)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    feature_columns = ["AMT_CREDIT", "EXT_SOURCE_2", "CODE_GENDER"]
    numeric_columns = ["AMT_CREDIT", "EXT_SOURCE_2"]
    categorical_columns = ["CODE_GENDER"]

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    train = data.iloc[:4]
    preprocessor.fit(train[feature_columns])
    transformed = preprocessor.transform(train[feature_columns])
    model = XGBClassifier(
        n_estimators=8,
        max_depth=2,
        learning_rate=0.2,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1,
    )
    model.fit(transformed, train["TARGET"].to_numpy())
    model.get_booster().save_model(str(model_dir / "xgboost_baseline.json"))
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")
    (model_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns), encoding="utf-8"
    )
    post_transform_features = [
        str(name) for name in preprocessor.get_feature_names_out()
    ]
    (model_dir / "preprocessor_metadata.json").write_text(
        json.dumps(
            {
                "input_features": feature_columns,
                "numeric_features": numeric_columns,
                "categorical_features": categorical_columns,
                "post_transform_features": post_transform_features,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "XGBoost",
                "model_version": "tiny-test",
                "training_date": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "XGBoost",
                "model_version": "tiny-test",
                "training_date": "2026-01-01T00:00:00+00:00",
                "library_versions": {"xgboost": "test", "python": "test"},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "evaluation_results.json").write_text(
        json.dumps(
            {
                "model_name": "XGBoost",
                "model_version": "tiny-test",
                "threshold": 0.5,
                "metrics": {"test": {"roc_auc": 0.9}},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "global_importance.json").write_text(
        json.dumps(
            {
                "method": "tree_shap",
                "features": [
                    {"rank": 1, "feature": "EXT_SOURCE_2", "mean_abs_shap": 0.5}
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        model_dir=model_dir,
        application_data_path=data_path,
        cors_origins=("http://localhost:8501",),
        log_level="CRITICAL",
    )
    return settings, data


@pytest.fixture()
def tiny_client(tiny_artifacts: tuple[Settings, pd.DataFrame]):
    settings, _ = tiny_artifacts
    with TestClient(create_app(settings)) as client:
        yield client


def test_health_and_model_metadata(tiny_client: TestClient) -> None:
    health = tiny_client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["data_available"] is True
    assert payload["model_version"] == "tiny-test"

    models = tiny_client.get("/models").json()
    assert models["model_name"] == "XGBoost"
    assert models["model_version"] == "tiny-test"
    assert models["thresholds"]["decision_threshold"] == 0.5
    assert models["thresholds"]["provisional"] is True

    evaluation = tiny_client.get("/model/evaluation")
    assert evaluation.status_code == 200
    assert evaluation.json()["provenance"]["threshold"] == 0.5

    importance = tiny_client.get("/model/global-importance")
    assert importance.status_code == 200
    assert importance.json()["features"][0]["feature"] == "EXT_SOURCE_2"


def test_prediction_by_application_id_and_explanation(tiny_client: TestClient) -> None:
    response = tiny_client.post("/predict", json={"application_id": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert payload["application_id"] == 2
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert payload["explanation"] is None
    assert "model_hash" not in payload

    explained = tiny_client.post(
        "/predict",
        params={"include_explanation": True},
        json={"application_id": 2},
    )
    assert explained.status_code == 200
    explained_payload = explained.json()
    assert explained_payload["probability"] == payload["probability"]
    explanation = explained_payload["explanation"]
    assert explanation["method"] == "tree_shap"
    assert explanation["application_id"] == 2
    assert math.isclose(
        math.exp(explanation["prediction_margin"])
        / (1.0 + math.exp(explanation["prediction_margin"])),
        explained_payload["probability"],
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        explanation["base_value"] + explanation["contribution_sum"],
        explanation["prediction_margin"],
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_prediction_by_features_and_unknown_category(
    tiny_client: TestClient, tiny_artifacts: tuple[Settings, pd.DataFrame]
) -> None:
    _, data = tiny_artifacts
    features = _jsonable(
        {
            key: value
            for key, value in data.iloc[4].to_dict().items()
            if key not in {"SK_ID_CURR", "TARGET"}
        }
    )
    features["CODE_GENDER"] = "UNSEEN"
    response = tiny_client.post("/predict", json={"features": features})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert payload["application_id"] is None


def test_prediction_errors_are_explicit(tiny_client: TestClient) -> None:
    missing = tiny_client.get("/applications/999/prediction")
    assert missing.status_code == 404
    assert missing.json()["status"] == "no_prediction"
    assert missing.json()["probability"] is None

    both = tiny_client.post(
        "/predict", json={"application_id": 1, "features": {"AMT_CREDIT": 1}}
    )
    assert both.status_code == 422
    assert both.json()["status"] == "invalid"

    neither = tiny_client.post("/predict", json={})
    assert neither.status_code == 422


def test_feature_validation_rejects_bad_payloads(
    tiny_client: TestClient, tiny_artifacts: tuple[Settings, pd.DataFrame]
) -> None:
    _, data = tiny_artifacts
    valid = _jsonable(
        {
            key: value
            for key, value in data.iloc[0].to_dict().items()
            if key not in {"SK_ID_CURR", "TARGET"}
        }
    )

    unknown = dict(valid)
    unknown["UNKNOWN_COLUMN"] = 1
    assert tiny_client.post("/predict", json={"features": unknown}).status_code == 422

    missing = dict(valid)
    missing.pop("CODE_GENDER")
    assert tiny_client.post("/predict", json={"features": missing}).status_code == 422

    wrong_type = dict(valid)
    wrong_type["AMT_CREDIT"] = "not-a-number"
    assert tiny_client.post("/predict", json={"features": wrong_type}).status_code == 422

    out_of_range = dict(valid)
    out_of_range["AMT_CREDIT"] = 999999999
    assert (
        tiny_client.post("/predict", json={"features": out_of_range}).status_code
        == 422
    )


def test_feature_validator_normalizes_all_missing_sentinels() -> None:
    validator = FeatureValidator(
        feature_columns=["AMT_CREDIT", "CODE_GENDER"],
        numeric_features=["AMT_CREDIT"],
        categorical_features=["CODE_GENDER"],
        numeric_bounds={},
    )

    for missing in (None, np.nan, pd.NA):
        frame = validator.validate(
            {"AMT_CREDIT": missing, "CODE_GENDER": missing}
        )
        assert pd.isna(frame["AMT_CREDIT"].iloc[0])
        assert pd.isna(frame["CODE_GENDER"].iloc[0])


def test_explicit_nulls_are_accepted_by_api(
    tiny_client: TestClient, tiny_artifacts: tuple[Settings, pd.DataFrame]
) -> None:
    _, data = tiny_artifacts
    features = _jsonable(
        {
            key: value
            for key, value in data.iloc[0].to_dict().items()
            if key not in {"SK_ID_CURR", "TARGET"}
        }
    )
    features["CODE_GENDER"] = None
    features["EXT_SOURCE_2"] = None

    response = tiny_client.post("/predict", json={"features": features})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert 0.0 <= payload["probability"] <= 1.0


def test_missing_data_returns_degraded_health_and_unavailable_prediction(
    tiny_artifacts: tuple[Settings, pd.DataFrame], tmp_path: Path
) -> None:
    settings, _ = tiny_artifacts
    missing_data_settings = Settings(
        model_dir=settings.model_dir,
        application_data_path=tmp_path / "missing.csv",
        cors_origins=settings.cors_origins,
        log_level="CRITICAL",
    )
    with TestClient(create_app(missing_data_settings)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert health.json()["data_available"] is False

        prediction = client.post("/predict", json={"application_id": 1})
        assert prediction.status_code == 503
        assert prediction.json()["status"] == "unavailable"
        assert prediction.json()["probability"] is None


def test_duplicate_application_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.csv"
    pd.DataFrame({"SK_ID_CURR": [1, 1], "AMT_CREDIT": [10, 20]}).to_csv(
        path, index=False
    )
    with pytest.raises(DuplicateApplicationIdError):
        ApplicationRepository(path)


def test_cors_allows_only_configured_origins(tiny_client: TestClient) -> None:
    allowed = tiny_client.get(
        "/health", headers={"Origin": "http://localhost:8501"}
    )
    assert allowed.headers.get("access-control-allow-origin") == (
        "http://localhost:8501"
    )
    denied = tiny_client.get(
        "/health", headers={"Origin": "https://example.com"}
    )
    assert denied.headers.get("access-control-allow-origin") is None


def test_dashboard_normalizer_accepts_api_available_status(
    tiny_client: TestClient,
) -> None:
    payload = tiny_client.post("/predict", json={"application_id": 2}).json()
    normalized = normalize_prediction_response(payload)
    assert normalized.status == "available"
    assert normalized.probability == payload["probability"]
