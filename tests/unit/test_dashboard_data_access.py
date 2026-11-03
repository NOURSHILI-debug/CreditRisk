from __future__ import annotations

import pandas as pd
import pytest

from dashboard import data_access


class FixtureProvider:
    def __init__(
        self,
        applications: object = None,
        prediction: object = None,
        evaluations: object = None,
    ) -> None:
        self.applications = (
            pd.DataFrame({"SK_ID_CURR": [1001]})
            if applications is None
            else applications
        )
        self.prediction = prediction
        self.evaluations = evaluations

    def get_applications(self) -> object:
        return self.applications

    def get_prediction(self, application_id: int) -> object:
        assert application_id == 1001
        return self.prediction

    def get_model_evaluation_results(self) -> object:
        return self.evaluations


def test_mock_provider_is_selected_by_default_and_applications_are_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CREDIT_RISK_DATA_PROVIDER", raising=False)

    assert data_access.get_provider_name() == "mock"
    assert data_access.is_mock_mode()

    applications = data_access.get_applications()
    applications.loc[0, "SK_ID_CURR"] = -1
    assert data_access.get_applications().loc[0, "SK_ID_CURR"] == 1001


def test_unknown_provider_is_unavailable_without_fabricated_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CREDIT_RISK_DATA_PROVIDER", "future-real-provider")

    with pytest.raises(data_access.DataSourceUnavailableError):
        data_access.get_applications()
    with pytest.raises(data_access.DataSourceUnavailableError):
        data_access.get_model_evaluation_results()
    assert data_access.get_credit_risk_prediction(1001).status == "unavailable"
    assert not data_access.is_mock_mode()

    monkeypatch.setenv("CREDIT_RISK_DATA_PROVIDER", " ")
    assert data_access.get_provider_name() == ""
    with pytest.raises(data_access.DataSourceUnavailableError, match="must not be empty"):
        data_access.get_applications()


def test_application_retrieval_rejects_malformed_provider_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_access, "get_data_provider", lambda: FixtureProvider(applications=[])
    )

    with pytest.raises(data_access.MalformedProviderResponseError):
        data_access.get_applications()


def test_application_lookup_distinguishes_invalid_missing_duplicate_and_unavailable() -> None:
    applications = pd.DataFrame(
        {"SK_ID_CURR": [1001, 1002, 1002], "AMT_CREDIT": [1, 2, 3]}
    )

    found = data_access.get_application_by_id("1001", applications)
    assert found.status == "found"
    assert found.application is not None
    assert found.application["AMT_CREDIT"] == 1
    assert data_access.get_application_by_id("bad", applications).status == "invalid_id"
    assert data_access.get_application_by_id("9999", applications).status == "not_found"
    assert data_access.get_application_by_id("1002", applications).status == "duplicate"
    assert (
        data_access.get_application_by_id(1001, pd.DataFrame({"AMT_CREDIT": [1]})).status
        == "unavailable"
    )


def test_overview_metrics_use_supplied_snapshot_and_handle_empty_or_missing_columns() -> None:
    applications = pd.DataFrame(
        {
            "TARGET": [0, 1, None, "invalid"],
            "RISK_CATEGORY": ["Low", "High", "Low", None],
        }
    )
    assert data_access.get_overview_metrics(applications) == {
        "total_applications": 4,
        "target_0": 1,
        "target_1": 1,
        "difficulty_rate": 0.5,
        "unlabeled": 2,
        "risk_distribution": {"Low": 2, "High": 1},
    }

    empty_metrics = data_access.get_overview_metrics(pd.DataFrame())
    assert empty_metrics["total_applications"] == 0
    assert empty_metrics["target_0"] is None
    assert empty_metrics["target_1"] is None
    assert empty_metrics["difficulty_rate"] is None
    assert empty_metrics["risk_distribution"] == {}

    missing_columns = data_access.get_overview_metrics(pd.DataFrame({"AMT_CREDIT": [10]}))
    assert missing_columns["total_applications"] == 1
    assert missing_columns["target_0"] is None
    assert missing_columns["unlabeled"] == 1


def test_overview_metrics_reject_non_dataframe_input() -> None:
    with pytest.raises(data_access.MalformedProviderResponseError):
        data_access.get_overview_metrics([])  # type: ignore[arg-type]


def test_prediction_uses_provider_contract_and_rejects_malformed_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FixtureProvider(
        prediction={
            "probability": 0.25,
            "model_name": "Fixture",
            "model_version": "v1",
        }
    )
    monkeypatch.setattr(data_access, "get_data_provider", lambda: provider)

    prediction = data_access.get_credit_risk_prediction("1001")
    assert prediction.status == "available"
    assert prediction.probability == 0.25
    assert prediction.model_name == "Fixture"

    provider.prediction = ["malformed"]
    assert data_access.get_credit_risk_prediction(1001).status == "invalid"
    assert data_access.get_credit_risk_prediction(True).status == "invalid"


def test_model_evaluation_retrieval_preserves_rows_and_handles_absent_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = pd.DataFrame(
        [{"model": "XGBoost", "metric": "roc_auc", "value": 0.8}]
    )
    provider = FixtureProvider(evaluations=results)
    monkeypatch.setattr(data_access, "get_data_provider", lambda: provider)

    retrieved = data_access.get_model_evaluation_results()
    assert retrieved is not None
    assert retrieved.equals(results)
    retrieved.loc[0, "value"] = 0.1
    assert provider.evaluations.loc[0, "value"] == 0.8

    provider.evaluations = None
    assert data_access.get_model_evaluation_results() is None


def test_model_evaluation_retrieval_and_normalization_report_malformed_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FixtureProvider(evaluations=[])
    monkeypatch.setattr(data_access, "get_data_provider", lambda: provider)
    with pytest.raises(data_access.MalformedProviderResponseError):
        data_access.get_model_evaluation_results()

    values, statuses, issues = data_access.normalize_model_evaluation_results(
        pd.DataFrame([{"model": "XGBoost", "score": 0.8}])
    )
    assert statuses == {"XGBoost": "Invalid results", "LSTM": "Invalid results"}
    assert all(value is None for model in values.values() for value in model.values())
    assert "missing required columns" in issues[0]
