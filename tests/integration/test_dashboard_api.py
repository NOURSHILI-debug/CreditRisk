from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.main import create_app
from agent import tools
from dashboard import data_access
from dashboard.api_client import APIClient


@pytest.fixture(scope="module")
def connected_dashboard_client(
    project_root: Path, model_dir: Path
) -> tuple[TestClient, APIClient]:
    data_path = project_root / "data" / "raw" / "application_train.csv"
    if not data_path.exists():
        pytest.skip("Raw application_train.csv is missing.")
    settings = Settings(
        model_dir=model_dir,
        application_data_path=data_path,
        cors_origins=("http://localhost:8501",),
        log_level="CRITICAL",
    )
    with TestClient(create_app(settings)) as api:
        yield api, APIClient("http://testserver", client=api)


def test_dashboard_client_reads_real_application_records(
    connected_dashboard_client: tuple[TestClient, APIClient],
) -> None:
    api, client = connected_dashboard_client
    page = client.list_applications(
        limit=2,
        offset=0,
        fields=["SK_ID_CURR", "NAME_INCOME_TYPE", "AMT_CREDIT"],
    )
    assert page["total"] == 307511
    assert len(page["items"]) == 2
    assert all("TARGET" not in row for row in page["items"])

    application = client.get_application(120321)
    assert application["application_id"] == 120321
    assert application["application"]["SK_ID_CURR"] == 120321
    assert "TARGET" not in application["application"]
    assert api.get("/applications/120321", params={"fields": "TARGET"}).status_code == 422


def test_dashboard_client_reads_real_prediction_and_shap(
    connected_dashboard_client: tuple[TestClient, APIClient],
) -> None:
    _, client = connected_dashboard_client
    prediction = client.predict(120321, include_explanation=True)
    assert prediction["status"] == "available"
    assert prediction["probability"] == pytest.approx(0.4915030896663666)
    assert prediction["risk_level"] == "MEDIUM"
    assert prediction["explanation"]["method"] == "tree_shap"
    assert prediction["explanation"]["top_features"]
    application = client.get_application(120321)["application"]
    matched_features = [
        feature
        for feature in prediction["explanation"]["top_features"]
        if feature["raw_feature"] in application
    ]
    assert matched_features
    assert matched_features[0]["raw_value"] == application[
        matched_features[0]["raw_feature"]
    ]
    assert prediction["explanation"]["probability"] == prediction["probability"]


def test_agent_prediction_tool_runs_real_inference_without_shap(
    connected_dashboard_client: tuple[TestClient, APIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, client = connected_dashboard_client
    monkeypatch.setattr(data_access, "_client", lambda: client)

    prediction = tools.get_prediction(120321)

    assert prediction["status"] == "available"
    assert prediction["application_id"] == 120321
    assert prediction["probability"] == pytest.approx(0.4915030896663666)
    assert prediction["risk_level"] == "MEDIUM"
    assert prediction["model_name"] == "XGBoost"
    assert prediction["model_version"]
    assert prediction["prediction_timestamp"]
    assert "explanation" not in prediction
    assert "TARGET" not in prediction


def test_agent_explanation_tool_returns_filtered_real_shap(
    connected_dashboard_client: tuple[TestClient, APIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, client = connected_dashboard_client
    monkeypatch.setattr(data_access, "_client", lambda: client)

    explanation = tools.explain_prediction(120321)

    assert explanation["status"] == "available"
    assert explanation["application_id"] == 120321
    assert explanation["prediction_probability"] == pytest.approx(
        0.4915030896663666
    )
    assert explanation["explanation_method"] == "tree_shap"
    assert explanation["output_space"] == "log_odds"
    assert explanation["top_features"]
    assert all(
        set(feature) == {"feature", "shap_value", "effect"}
        for feature in explanation["top_features"]
    )
    assert all(
        feature["feature"] not in {"TARGET", "SK_ID_CURR"}
        for feature in explanation["top_features"]
    )
    assert "raw_value" not in str(explanation)
    assert "transformed_feature" not in str(explanation)


def test_dashboard_client_reads_saved_prediction_and_evaluation_artifacts(
    connected_dashboard_client: tuple[TestClient, APIClient],
    model_dir: Path,
) -> None:
    _, client = connected_dashboard_client
    records = pd.read_csv(model_dir / "test_predictions.csv")
    summary = client.prediction_summary("test")
    assert summary["prediction_count"] == len(records)

    evaluation = client.evaluation("test")
    assert evaluation["split"] == "test"
    metrics = {row["metric"]: row["value"] for row in evaluation["rows"]}
    assert metrics["ROC-AUC"] == pytest.approx(0.7560700206065816)
    assert metrics["PR-AUC"] == pytest.approx(0.23680797485855454)
    assert evaluation["confusion_matrix"]["tn"] == 33397
    assert evaluation["confusion_matrix"]["tp"] == 2154


def test_dashboard_data_access_adapters_use_the_connected_real_api(
    connected_dashboard_client: tuple[TestClient, APIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, client = connected_dashboard_client
    monkeypatch.setattr(data_access, "_client", lambda: client)

    applications, total = data_access.get_applications_page(
        limit=3, offset=0, fields=["SK_ID_CURR", "AMT_CREDIT"]
    )
    prediction = data_access.get_credit_risk_prediction(120321)
    evaluation = data_access.get_model_evaluation_results("test")
    assert total == 307511
    assert len(applications) == 3
    assert "TARGET" not in applications.columns
    assert prediction.status == "available"
    assert prediction.explanation["top_features"]
    assert evaluation.attrs["split"] == "test"
