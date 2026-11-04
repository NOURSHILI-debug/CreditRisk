from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from api.config import Settings
from api.fields import APPLICATION_FIELDS
from api.main import create_app
from api.repository import ApplicationRepository, DuplicateApplicationIdError
from api.status import http_status_for, probability_from, status_for_probability
from dashboard.data_access import normalize_prediction_response
from inference.predictor import PredictionError
from processing.preprocessing import build_preprocessor

ERROR_KEYS = {
    "status",
    "message",
    "application_id",
    "probability",
    "risk_level",
    "model_name",
    "model_version",
    "prediction_timestamp",
    "explanation",
}

DASHBOARD_FIELDS = {
    "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
    "TARGET": [0, 1, 0, 1, 0, 1],
    "AMT_CREDIT": [1000.0, 2200.0, 1500.0, 3200.0, 1800.0, 4200.0],
    "EXT_SOURCE_2": [0.12, 0.82, 0.35, 0.71, 0.28, 0.91],
    "CODE_GENDER": ["M", "F", "M", "F", "M", "F"],
    "NAME_CONTRACT_TYPE": ["Cash", "Cash", "Revolving", "Cash", "Cash", "Revolving"],
    "NAME_INCOME_TYPE": ["Working", "Pensioner", "Working", "Working", "Student", "Pensioner"],
    "NAME_EDUCATION_TYPE": ["Higher education", "Secondary", "Higher education", "Higher education", "Secondary", "Secondary"],
    "NAME_FAMILY_STATUS": ["Single", "Married", "Single", "Married", "Single", "Married"],
    "NAME_HOUSING_TYPE": ["House", "Flat", "House", "Flat", "House", "Flat"],
    "AMT_INCOME_TOTAL": [30000.0, 42000.0, 28000.0, 51000.0, 19000.0, 61000.0],
    "AMT_ANNUITY": [500.0, 900.0, 700.0, 1200.0, 600.0, 1500.0],
    "AMT_GOODS_PRICE": [950.0, 2100.0, 1400.0, 3000.0, 1700.0, 4000.0],
    "DAYS_BIRTH": [-12000, -15000, -13000, -18000, -11000, -20000],
    "DAYS_EMPLOYED": [-1000, -3000, -2000, -5000, -500, -6000],
    "CNT_CHILDREN": [0, 1, 0, 2, 0, 1],
    "CNT_FAM_MEMBERS": [2.0, 3.0, 2.0, 4.0, 1.0, 3.0],
}


def _write_artifacts(tmp_path: Path) -> tuple[Settings, pd.DataFrame]:
    data = pd.DataFrame(DASHBOARD_FIELDS)
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
    (model_dir / "preprocessor_metadata.json").write_text(
        json.dumps(
            {
                "input_features": feature_columns,
                "numeric_features": numeric_columns,
                "categorical_features": categorical_columns,
                "post_transform_features": [
                    str(name) for name in preprocessor.get_feature_names_out()
                ],
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
                "library_versions": {"xgboost": "test"},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "evaluation_results.json").write_text(
        json.dumps(
            {
                "artifact_version": "1.0",
                "model_name": "XGBoost",
                "model_version": "tiny-test",
                "generated_at": "2026-01-02T00:00:00+00:00",
                "training_date": "2026-01-01T00:00:00+00:00",
                "threshold": 0.5,
                "split_protocol": {"method": "test"},
                "metrics": {
                    "validation": {
                        "n_samples": 2,
                        "class_counts": {"0": 1, "1": 1},
                        "roc_auc": 0.8,
                        "pr_auc": 0.7,
                        "precision": 0.6,
                        "recall": 0.5,
                        "f1": 0.55,
                        "confusion_matrix": {
                            "labels": [0, 1],
                            "tn": 1,
                            "fp": 0,
                            "fn": 0,
                            "tp": 1,
                            "matrix": [[1, 0], [0, 1]],
                        },
                    },
                    "test": {
                        "n_samples": 2,
                        "class_counts": {"0": 1, "1": 1},
                        "roc_auc": 0.9,
                        "pr_auc": 0.75,
                        "precision": 0.65,
                        "recall": 0.55,
                        "f1": 0.6,
                        "confusion_matrix": {
                            "labels": [0, 1],
                            "tn": 1,
                            "fp": 0,
                            "fn": 0,
                            "tp": 1,
                            "matrix": [[1, 0], [0, 1]],
                        },
                    },
                },
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
def contract_artifacts(tmp_path: Path) -> tuple[Settings, pd.DataFrame]:
    return _write_artifacts(tmp_path)


@pytest.fixture()
def client(contract_artifacts: tuple[Settings, pd.DataFrame]):
    settings, _ = contract_artifacts
    with TestClient(create_app(settings)) as test_client:
        yield test_client


class _StubPredictor:
    model_name = "XGBoost"
    model_version = "stub"
    model_hash = "stub-hash"

    def __init__(self, result: Any) -> None:
        self.result = result

    def predict(self, application: Any, include_explanation: bool = False) -> Any:
        del application, include_explanation
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_status_mapping_is_canonical() -> None:
    assert status_for_probability(0.0) == "available"
    assert status_for_probability(1.0) == "available"
    assert status_for_probability(0.5) == "available"
    assert status_for_probability(None) == "no_prediction"
    for invalid in (
        True,
        False,
        "0.5",
        float("nan"),
        float("inf"),
        -0.01,
        1.01,
        object(),
    ):
        assert status_for_probability(invalid) == "invalid"

    assert http_status_for("available") == 200
    assert http_status_for("no_prediction") == 200
    assert http_status_for("invalid") == 422
    assert http_status_for("model_unavailable") == 503
    assert http_status_for("unavailable") == 503
    assert probability_from(0.25) == 0.25
    assert probability_from(None) is None
    with pytest.raises(ValueError):
        probability_from(float("nan"))


def test_root_and_docs(client: TestClient) -> None:
    root = client.get("/")
    assert root.status_code == 200
    assert root.json() == {
        "service": "Credit Risk Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
    }
    assert client.get("/docs").status_code == 200


def test_applications_list_defaults_allowlist_and_pagination(
    client: TestClient,
) -> None:
    response = client.get("/applications")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 6
    assert payload["limit"] == 100
    assert payload["offset"] == 0
    assert payload["fields"] == list(APPLICATION_FIELDS)
    assert [item["SK_ID_CURR"] for item in payload["items"]] == [1, 2, 3, 4, 5, 6]
    assert all(set(item) == set(APPLICATION_FIELDS) for item in payload["items"])

    page = client.get(
        "/applications",
        params={"fields": "SK_ID_CURR,AMT_CREDIT", "limit": 2, "offset": 2},
    )
    assert page.status_code == 200
    page_payload = page.json()
    assert page_payload["fields"] == ["SK_ID_CURR", "AMT_CREDIT"]
    assert page_payload["total"] == 6
    assert [item["SK_ID_CURR"] for item in page_payload["items"]] == [3, 4]
    assert all(
        set(item) == {"SK_ID_CURR", "AMT_CREDIT"}
        for item in page_payload["items"]
    )


def test_applications_list_rejects_unknown_fields_and_bad_bounds(
    client: TestClient,
) -> None:
    unknown = client.get("/applications", params={"fields": "SK_ID_CURR,NOPE"})
    assert unknown.status_code == 422
    assert unknown.json()["status"] == "invalid"
    assert "NOPE" in unknown.json()["message"]

    for params in ({"limit": 0}, {"limit": 501}, {"offset": -1}):
        response = client.get("/applications", params=params)
        assert response.status_code == 422
        assert response.json()["status"] == "invalid"


def test_applications_support_exact_id_and_category_filters(
    client: TestClient,
) -> None:
    by_id = client.get("/applications", params={"application_id": 3})
    assert by_id.status_code == 200
    assert by_id.json()["total"] == 1
    assert by_id.json()["items"][0]["SK_ID_CURR"] == 3
    assert "TARGET" not in by_id.json()["items"][0]

    by_category = client.get(
        "/applications",
        params={
            "filter_field": "NAME_INCOME_TYPE",
            "filter_value": "Working",
            "fields": "SK_ID_CURR,NAME_INCOME_TYPE",
        },
    )
    assert by_category.status_code == 200
    assert all(
        item["NAME_INCOME_TYPE"] == "Working"
        for item in by_category.json()["items"]
    )

    invalid_filter = client.get(
        "/applications",
        params={"filter_field": "TARGET", "filter_value": "1"},
    )
    assert invalid_filter.status_code == 422
    assert invalid_filter.json()["status"] == "invalid"


def test_application_detail_and_unknown_id(client: TestClient) -> None:
    response = client.get("/applications/3")
    assert response.status_code == 200
    payload = response.json()
    assert payload["application_id"] == 3
    assert payload["application"]["SK_ID_CURR"] == 3
    assert "TARGET" not in payload["application"]

    trimmed = client.get(
        "/applications/3", params={"fields": "SK_ID_CURR,AMT_CREDIT"}
    )
    assert trimmed.status_code == 200
    assert set(trimmed.json()["application"]) == {"SK_ID_CURR", "AMT_CREDIT"}
    target = client.get("/applications/3", params={"fields": "TARGET"})
    assert target.status_code == 422

    bad_fields = client.get(
        "/applications/3", params={"fields": "SK_ID_CURR,NOPE"}
    )
    assert bad_fields.status_code == 422
    assert bad_fields.json()["status"] == "invalid"

    missing = client.get("/applications/999")
    assert missing.status_code == 404
    assert missing.json()["status"] == "invalid"
    assert missing.json()["probability"] is None


def test_application_detail_reports_duplicate_ids(client: TestClient) -> None:
    class DuplicateRepository:
        available = True

        def get_full_by_id(self, application_id: int) -> Any:
            raise DuplicateApplicationIdError(
                f"Multiple rows found for application {application_id}."
            )

    client.app.state.container.repository = DuplicateRepository()
    response = client.get("/applications/1")
    assert response.status_code == 409
    assert response.json()["status"] == "invalid"
    assert "Multiple rows" in response.json()["message"]


def test_repository_rejects_duplicate_ids_on_load(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.csv"
    pd.DataFrame({"SK_ID_CURR": [1, 1], "AMT_CREDIT": [10, 20]}).to_csv(
        path, index=False
    )
    with pytest.raises(DuplicateApplicationIdError):
        ApplicationRepository(path)


def test_evaluation_rows_default_to_test_split(client: TestClient) -> None:
    response = client.get("/model/evaluation")
    assert response.status_code == 200
    payload = response.json()
    assert payload["split"] == "test"
    assert payload["provenance"]["threshold"] == 0.5
    assert payload["provenance"]["source"] == "evaluation_results.json"
    assert payload["provenance"]["lstm_available"] is False
    assert payload["provenance"]["n_samples"] == 2
    assert {row["model"] for row in payload["rows"]} == {"XGBoost"}
    values = {row["metric"]: row["value"] for row in payload["rows"]}
    assert values == {
        "ROC-AUC": 0.9,
        "PR-AUC": 0.75,
        "Precision": 0.65,
        "Recall": 0.55,
        "F1-score": 0.6,
    }
    assert payload["confusion_matrix"]["tn"] == 1
    assert payload["confusion_matrix"]["tp"] == 1

    validation = client.get("/model/evaluation", params={"split": "validation"})
    assert validation.status_code == 200
    assert validation.json()["split"] == "validation"
    assert validation.json()["provenance"]["n_samples"] == 2


def test_evaluation_rejects_unsupported_split(client: TestClient) -> None:
    response = client.get("/model/evaluation", params={"split": "train"})
    assert response.status_code == 422
    assert response.json()["status"] == "invalid"


def test_missing_evaluation_artifact_returns_model_unavailable(
    contract_artifacts: tuple[Settings, pd.DataFrame], tmp_path: Path
) -> None:
    settings, _ = contract_artifacts
    copied_model_dir = tmp_path / "copied_model"
    shutil.copytree(settings.model_dir, copied_model_dir)
    (copied_model_dir / "evaluation_results.json").unlink()
    missing_settings = Settings(
        model_dir=copied_model_dir,
        application_data_path=settings.application_data_path,
        cors_origins=settings.cors_origins,
        log_level="CRITICAL",
    )
    with TestClient(create_app(missing_settings)) as test_client:
        response = test_client.get("/model/evaluation")
        assert response.status_code == 503
        assert response.json()["status"] == "model_unavailable"
        assert response.json()["probability"] is None


def test_prediction_status_cases(client: TestClient) -> None:
    success = client.post("/predict", json={"application_id": 2})
    assert success.status_code == 200
    payload = success.json()
    assert payload["status"] == "available"
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["explanation"] is None

    explained = client.post(
        "/predict",
        params={"include_explanation": True},
        json={"application_id": 2},
    )
    assert explained.status_code == 200
    assert explained.json()["status"] == "available"
    assert explained.json()["probability"] == payload["probability"]

    missing = client.get("/applications/999/prediction")
    assert missing.status_code == 404
    assert missing.json()["status"] == "no_prediction"
    assert missing.json()["probability"] is None

    invalid = client.post(
        "/predict",
        json={"application_id": 1, "features": {"AMT_CREDIT": 1}},
    )
    assert invalid.status_code == 422
    assert invalid.json()["status"] == "invalid"

    client.app.state.container.predictor = _StubPredictor(
        {"status": "success", "probability": float("nan")}
    )
    invalid_probability = client.post("/predict", json={"application_id": 1})
    assert invalid_probability.status_code == 500
    assert invalid_probability.json()["status"] == "invalid"
    assert invalid_probability.json()["probability"] is None

    client.app.state.container.predictor = _StubPredictor(
        {"status": "no_prediction", "message": "No score for this record."}
    )
    no_prediction = client.post("/predict", json={"application_id": 1})
    assert no_prediction.status_code == 200
    assert no_prediction.json()["status"] == "no_prediction"
    assert no_prediction.json()["probability"] is None

    client.app.state.container.predictor = _StubPredictor(
        PredictionError("Inference failed.")
    )
    inference_failure = client.post("/predict", json={"application_id": 1})
    assert inference_failure.status_code == 422
    assert inference_failure.json()["status"] == "invalid"


def test_prediction_summary_uses_saved_prediction_records(
    client: TestClient,
) -> None:
    records = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "y_true": [0, 0, 1, 1],
            "probability": [0.1, 0.4, 0.599, 0.9],
            "prediction": [0, 0, 1, 1],
        }
    )
    client.app.state.container.prediction_records["test"] = records
    response = client.get("/model/prediction-summary", params={"split": "test"})
    assert response.status_code == 200
    assert response.json() == {
        "split": "test",
        "prediction_count": 4,
        "predicted_outcomes": {
            "no_defined_repayment_difficulty": 2,
            "repayment_difficulty": 2,
        },
        "risk_distribution": {"LOW": 1, "MEDIUM": 2, "HIGH": 1},
        "risk_bands": {"low_upper": 0.3, "medium_upper": 0.6},
        "decision_threshold": 0.5,
        "source": "test_predictions.csv",
    }


def test_missing_data_and_model_statuses(
    contract_artifacts: tuple[Settings, pd.DataFrame], tmp_path: Path
) -> None:
    settings, _ = contract_artifacts
    missing_data_settings = Settings(
        model_dir=settings.model_dir,
        application_data_path=tmp_path / "missing.csv",
        cors_origins=settings.cors_origins,
        log_level="CRITICAL",
    )
    with TestClient(create_app(missing_data_settings)) as test_client:
        prediction = test_client.post("/predict", json={"application_id": 1})
        assert prediction.status_code == 503
        assert prediction.json()["status"] == "unavailable"
        assert prediction.json()["probability"] is None
        applications = test_client.get("/applications")
        assert applications.status_code == 503
        assert applications.json()["status"] == "unavailable"

    missing_model_settings = Settings(
        model_dir=tmp_path / "missing_model",
        application_data_path=settings.application_data_path,
        cors_origins=settings.cors_origins,
        log_level="CRITICAL",
    )
    with TestClient(create_app(missing_model_settings)) as test_client:
        prediction = test_client.post("/predict", json={"application_id": 1})
        assert prediction.status_code == 503
        assert prediction.json()["status"] == "model_unavailable"
        assert prediction.json()["probability"] is None


def test_every_error_uses_the_same_shape(client: TestClient) -> None:
    responses = [
        client.get("/applications/999"),
        client.get("/applications", params={"fields": "NOPE"}),
        client.get("/applications", params={"limit": 0}),
        client.get("/applications/999/prediction"),
        client.post("/predict", json={}),
        client.get("/does-not-exist"),
    ]
    for response in responses:
        assert response.status_code >= 400
        body = response.json()
        assert set(body) == ERROR_KEYS
        assert body["status"] in {
            "invalid",
            "no_prediction",
            "unavailable",
            "model_unavailable",
        }
        assert body["probability"] is None


def test_dashboard_normalizer_round_trip(client: TestClient) -> None:
    payload = client.post("/predict", json={"application_id": 2}).json()
    normalized = normalize_prediction_response(payload)
    assert normalized.status == "available"
    assert normalized.probability == payload["probability"]
    assert math.isfinite(normalized.probability)


def test_explanation_does_not_change_probability(client: TestClient) -> None:
    plain = client.post("/predict", json={"application_id": 2}).json()
    explained = client.post(
        "/predict",
        params={"include_explanation": True},
        json={"application_id": 2},
    ).json()
    assert explained["probability"] == plain["probability"]
    if explained["explanation"] is not None:
        assert explained["explanation"]["probability"] == plain["probability"]


def test_openapi_export_is_present(project_root: Path) -> None:
    exported = json.loads(
        (project_root / "docs" / "openapi.json").read_text(encoding="utf-8")
    )
    assert "/applications" in exported["paths"]
    assert "/applications/{application_id}" in exported["paths"]
    assert "/model/evaluation" in exported["paths"]
    assert "/model/prediction-summary" in exported["paths"]
