from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent import tools
from dashboard import data_access


def _assert_json(value: object) -> None:
    json.dumps(value, allow_nan=False)


def test_application_details_returns_serializable_fields_and_nulls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools.data_access, "is_mock_mode", lambda: True)
    monkeypatch.setattr(
        tools.data_access,
        "get_application_by_id",
        lambda _: data_access.ApplicationLookupResult(
            status="found",
            application=pd.Series(
                {
                    "SK_ID_CURR": np.int64(1001),
                    "AMT_CREDIT": np.float64(50000),
                    "NAME_INCOME_TYPE": "Working",
                    "DAYS_BIRTH": pd.NA,
                    "RISK_PROBABILITY": np.float64(0.9),
                    "RISK_CATEGORY": "High Risk",
                }
            ),
        ),
    )

    result = tools.get_application_details(" 1001 ")

    assert result == {
        "status": "found",
        "application_id": "1001",
        "fields": {
            "SK_ID_CURR": 1001,
            "AMT_CREDIT": 50000.0,
            "NAME_INCOME_TYPE": "Working",
        },
        "null_fields": ["DAYS_BIRTH"],
        "source_mode": "mock",
    }
    _assert_json(result)


@pytest.mark.parametrize(
    ("application_id", "status"),
    [
        ("bad", "invalid_id"),
        ("9999", "not_found"),
        ("1002", "duplicate"),
        ("1001", "unavailable"),
    ],
)
def test_application_details_preserves_lookup_statuses(
    monkeypatch: pytest.MonkeyPatch,
    application_id: str,
    status: str,
) -> None:
    monkeypatch.setattr(
        tools.data_access,
        "get_application_by_id",
        lambda _: data_access.ApplicationLookupResult(status=status),
    )

    result = tools.get_application_details(application_id)

    assert result["status"] == status
    assert result["fields"] == {}
    _assert_json(result)


def test_application_details_reports_unavailable_and_malformed_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_: object) -> data_access.ApplicationLookupResult:
        raise data_access.DataSourceUnavailableError("provider unavailable")

    monkeypatch.setattr(tools.data_access, "get_application_by_id", unavailable)
    result = tools.get_application_details(1001)
    assert result["status"] == "unavailable"
    assert result["message"] == "provider unavailable"

    def malformed(_: object) -> data_access.ApplicationLookupResult:
        raise data_access.MalformedProviderResponseError("malformed source")

    monkeypatch.setattr(tools.data_access, "get_application_by_id", malformed)
    result = tools.get_application_details(1001)
    assert result["status"] == "invalid"
    assert result["message"] == "malformed source"
    _assert_json(result)


def test_dashboard_metrics_returns_shared_values_and_handles_empty_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {
        "total_applications": 3,
        "prediction_count": 3,
        "prediction_split": "test",
        "predicted_outcomes": {"0": 2, "1": 1},
        "risk_distribution": {"LOW": 1, "MEDIUM": 1, "HIGH": 1},
        "risk_bands": {"low_upper": 0.3, "medium_upper": 0.6},
        "decision_threshold": 0.5,
        "source": "test_predictions.csv",
    }
    monkeypatch.setattr(
        tools.data_access,
        "get_overview_metrics",
        lambda: metrics,
    )
    result = tools.get_dashboard_metrics()
    assert result["status"] == "available"
    assert result["metrics"] == metrics
    _assert_json(result)

    monkeypatch.setattr(
        tools.data_access,
        "get_overview_metrics",
        lambda: {**metrics, "total_applications": 0},
    )
    empty = tools.get_dashboard_metrics()
    assert empty["status"] == "unavailable"
    assert empty["metrics"]["total_applications"] == 0
    _assert_json(empty)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (data_access.DataSourceUnavailableError("offline"), "unavailable"),
        (data_access.MalformedProviderResponseError("malformed"), "invalid"),
    ],
)
def test_dashboard_metrics_reports_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: data_access.DataAccessError,
    status: str,
) -> None:
    def fail() -> dict:
        raise error

    monkeypatch.setattr(tools.data_access, "get_overview_metrics", fail)
    result = tools.get_dashboard_metrics()
    assert result["status"] == status
    assert result["metrics"] is None
    assert result["message"] == str(error)
    _assert_json(result)


def test_model_comparison_returns_only_normalized_actual_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = pd.DataFrame(
        [
            {"model": "XGBoost", "metric": "roc_auc", "value": 0.81},
            {"model": "XGBoost", "metric": "pr_auc", "value": 0.42},
        ]
    )
    monkeypatch.setattr(
        tools.data_access, "get_model_evaluation_results", lambda: rows
    )
    result = tools.get_model_comparison()
    assert result["status"] == "partial"
    assert result["models"]["XGBoost"]["ROC-AUC"] == 0.81
    assert result["models"]["XGBoost"]["Recall"] is None
    assert result["statuses"]["LSTM"] == "Not available"
    assert result["comparison_available"] is False
    assert result["evaluations"][1]["evaluation_status"] == "unavailable"
    assert result["evaluations"][1]["metrics"] is None
    assert "LSTM evaluation results are not provided" in result["comparison_note"]
    assert "not available in the API" in (
        result["evaluations"][1]["unavailable_reason"]
    )
    _assert_json(result)


def test_model_comparison_returns_available_complete_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = pd.DataFrame(
        [
            {"model": model, "metric": metric, "value": index / 10}
            for model in ("XGBoost", "LSTM")
            for index, metric in enumerate(data_access.MODEL_METRICS, start=1)
        ]
    )
    monkeypatch.setattr(
        tools.data_access, "get_model_evaluation_results", lambda: rows
    )

    result = tools.get_model_comparison()

    assert result["status"] == "available"
    assert result["statuses"] == {"XGBoost": "Available", "LSTM": "Available"}
    assert result["models"]["XGBoost"]["ROC-AUC"] == 0.1
    assert result["models"]["LSTM"]["F1-score"] == 0.5
    assert result["comparison_available"] is False
    assert "cannot be confirmed" in result["comparison_note"]
    _assert_json(result)


def test_model_comparison_enables_comparison_only_with_shared_protocol_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = pd.DataFrame(
        [
            {"model": model, "metric": metric, "value": index / 10}
            for model in ("XGBoost", "LSTM")
            for index, metric in enumerate(data_access.MODEL_METRICS, start=1)
        ]
    )
    rows.attrs["split"] = "test"
    rows.attrs["provenance"] = {
        "target": {
            "name": "TARGET",
            "positive_class": 1,
            "positive_label": "repayment difficulty",
            "probability_semantics": "P(TARGET=1)",
        },
        "metric_definitions": {
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "precision": "Precision at threshold.",
            "recall": "Recall at threshold.",
            "f1": "F1 at threshold.",
        },
        "threshold": 0.5,
        "n_samples": 100,
        "class_counts": {"0": 80, "1": 20},
        "split_protocol": {"method": "stratified train/test"},
        "source": "evaluation_results.json",
    }
    monkeypatch.setattr(
        tools.data_access, "get_model_evaluation_results", lambda: rows
    )

    result = tools.get_model_comparison()

    assert result["comparison_available"] is True
    assert "same reported split" in result["comparison_note"]
    assert [item["dataset_split"] for item in result["evaluations"]] == [
        "test",
        "test",
    ]
    assert result["evaluation_metadata"]["target"]["positive_class"] == 1
    _assert_json(result)


@pytest.mark.parametrize(
    "rows",
    [None, pd.DataFrame(columns=["model", "metric", "value"])],
)
def test_model_comparison_reports_missing_evaluation_results(
    monkeypatch: pytest.MonkeyPatch,
    rows: pd.DataFrame | None,
) -> None:
    monkeypatch.setattr(
        tools.data_access, "get_model_evaluation_results", lambda: rows
    )
    result = tools.get_model_comparison()
    assert result["status"] == "unavailable"
    assert result["statuses"] == {
        "XGBoost": "Not available",
        "LSTM": "Not available",
    }
    assert result["comparison_available"] is False
    assert all(
        item["evaluation_status"] == "unavailable"
        for item in result["evaluations"]
    )
    assert all(
        value is None
        for metrics in result["models"].values()
        for value in metrics.values()
    )
    _assert_json(result)


def test_model_comparison_reports_malformed_results_and_unavailable_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools.data_access,
        "get_model_evaluation_results",
        lambda: pd.DataFrame([{"model": "XGBoost", "score": 0.8}]),
    )
    malformed = tools.get_model_comparison()
    assert malformed["status"] == "invalid"
    assert "missing required columns" in malformed["issues"][0]
    _assert_json(malformed)

    def unavailable() -> None:
        raise data_access.DataSourceUnavailableError("evaluation source offline")

    monkeypatch.setattr(
        tools.data_access, "get_model_evaluation_results", unavailable
    )
    result = tools.get_model_comparison()
    assert result["status"] == "unavailable"
    assert result["message"] == "Evaluation results are unavailable from the API."
    assert "evaluation source offline" not in str(result)
    _assert_json(result)


def test_model_comparison_handles_unexpected_execution_errors_without_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise RuntimeError("private path and response payload")

    monkeypatch.setattr(
        tools.data_access, "get_model_evaluation_results", fail
    )

    result = tools.get_model_comparison()

    assert result["status"] == "error"
    assert result["comparison_available"] is False
    assert "private path" not in str(result)
    _assert_json(result)


def test_prediction_preserves_task_9_contract_and_unavailable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []

    def prediction(
        application_id: int, *, include_explanation: bool
    ) -> data_access.CreditRiskPrediction:
        calls.append((application_id, include_explanation))
        return data_access.CreditRiskPrediction(
            status="available",
            probability=0.25,
            risk_level="LOW",
            model_name="XGBoost",
            model_version="v2",
            prediction_timestamp="2026-11-03T10:00:00Z",
            explanation={"details": "must not reach the agent"},
        )

    monkeypatch.setattr(
        tools.data_access,
        "get_credit_risk_prediction",
        prediction,
    )
    result = tools.get_prediction("1001")
    assert result == {
        "status": "available",
        "application_id": 1001,
        "probability": 0.25,
        "risk_level": "LOW",
        "model_name": "XGBoost",
        "model_version": "v2",
        "prediction_timestamp": "2026-11-03T10:00:00Z",
        "message": None,
        "source_mode": "configured",
    }
    assert calls == [(1001, False)]
    assert "explanation" not in result
    _assert_json(result)

    def unavailable_prediction(
        _application_id: int, *, include_explanation: bool
    ) -> data_access.CreditRiskPrediction:
        assert _application_id == 1001
        assert include_explanation is False
        return data_access.CreditRiskPrediction(
            status="unavailable", message="no prediction provider"
        )

    monkeypatch.setattr(
        tools.data_access,
        "get_credit_risk_prediction",
        unavailable_prediction,
    )
    unavailable = tools.get_prediction(1001)
    assert unavailable["status"] == "unavailable"
    assert unavailable["application_id"] == 1001
    assert unavailable["probability"] is None
    assert unavailable["message"] == "no prediction provider"
    _assert_json(unavailable)


def test_prediction_preserves_missing_optional_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_optional_metadata(
        _application_id: int, *, include_explanation: bool
    ) -> data_access.CreditRiskPrediction:
        assert _application_id == 1001
        assert include_explanation is False
        return data_access.CreditRiskPrediction(
            status="available", probability=0.0
        )

    monkeypatch.setattr(
        tools.data_access,
        "get_credit_risk_prediction",
        missing_optional_metadata,
    )

    result = tools.get_prediction(1001)

    assert result["application_id"] == 1001
    assert result["probability"] == 0.0
    assert result["risk_level"] is None
    assert result["model_name"] is None
    assert result["model_version"] is None
    assert result["prediction_timestamp"] is None
    assert result["message"] is None
    _assert_json(result)


def test_prediction_rejects_invalid_id_before_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    called_ids: list[object] = []

    def forbidden(
        application_id: object, *, include_explanation: bool
    ) -> None:
        nonlocal called
        assert include_explanation is False
        called_ids.append(application_id)
        called = True

    monkeypatch.setattr(
        tools.data_access, "get_credit_risk_prediction", forbidden
    )

    result = tools.get_prediction(True)

    assert result["status"] == "invalid"
    assert result["probability"] is None
    assert result["application_id"] is None
    assert not called
    assert called_ids == []
    _assert_json(result)


@pytest.mark.parametrize(
    ("prediction_status", "expected_status"),
    [
        ("not_found", "not_found"),
        ("no_prediction", "no_prediction"),
        ("model_unavailable", "model_unavailable"),
        ("unavailable", "unavailable"),
        ("error", "error"),
        ("invalid", "invalid"),
    ],
)
def test_prediction_preserves_explicit_statuses(
    monkeypatch: pytest.MonkeyPatch,
    prediction_status: str,
    expected_status: str,
) -> None:
    def prediction_with_status(
        _application_id: int, *, include_explanation: bool
    ) -> data_access.CreditRiskPrediction:
        assert _application_id == 1001
        assert include_explanation is False
        return data_access.CreditRiskPrediction(
            status=prediction_status, message="status-specific message"
        )

    monkeypatch.setattr(
        tools.data_access,
        "get_credit_risk_prediction",
        prediction_with_status,
    )

    result = tools.get_prediction(1001)

    assert result["status"] == expected_status
    assert result["application_id"] == 1001
    assert result["probability"] is None
    assert result["message"] == "status-specific message"
    _assert_json(result)


def test_prediction_reports_data_access_errors_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(
        _: object, *, include_explanation: bool
    ) -> data_access.CreditRiskPrediction:
        assert include_explanation is False
        raise data_access.DataSourceUnavailableError("inference offline")

    monkeypatch.setattr(
        tools.data_access, "get_credit_risk_prediction", fail
    )
    result = tools.get_prediction(1001)
    assert result["status"] == "unavailable"
    assert result["message"] == "Prediction service is unavailable."
    _assert_json(result)

    def malformed(
        _: object, *, include_explanation: bool
    ) -> data_access.CreditRiskPrediction:
        assert include_explanation is False
        raise data_access.MalformedProviderResponseError(
            "C:\\internal\\secret\\response.json"
        )

    monkeypatch.setattr(
        tools.data_access, "get_credit_risk_prediction", malformed
    )
    result = tools.get_prediction(1001)
    assert result["status"] == "invalid"
    assert result["message"] == "The prediction service returned an invalid response."
    assert "secret" not in str(result)

    def unexpected(
        _: object, *, include_explanation: bool
    ) -> data_access.CreditRiskPrediction:
        assert include_explanation is False
        raise data_access.DataAccessError("internal detail")

    monkeypatch.setattr(
        tools.data_access, "get_credit_risk_prediction", unexpected
    )
    result = tools.get_prediction(1001)
    assert result["status"] == "error"
    assert result["message"] == "Prediction retrieval failed."
    assert "internal detail" not in str(result)
    _assert_json(result)


def test_explain_prediction_returns_valid_minimized_shap_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []
    explanation = {
        "method": "tree_shap",
        "output_space": "log_odds",
        "application_id": 1001,
        "base_value": 0.12,
        "base_probability": 0.53,
        "prediction_margin": 1.26,
        "contribution_sum": 1.14,
        "probability": 0.78,
        "summary": "Risk-increasing and decreasing features.",
        "top_features": [
            {
                "feature": "EXT_SOURCE_2",
                "raw_feature": "EXT_SOURCE_2",
                "transformed_feature": "num__EXT_SOURCE_2",
                "value": 0.2,
                "raw_value": 0.2,
                "shap_value": 0.42,
                "effect": "increases_risk",
            },
            {
                "feature": "AMT_CREDIT",
                "raw_feature": "AMT_CREDIT",
                "transformed_feature": "num__AMT_CREDIT",
                "value": 90000.0,
                "raw_value": 90000.0,
                "shap_value": -0.18,
                "effect": "decreases_risk",
            },
        ],
    }

    def prediction(
        application_id: int, *, include_explanation: bool
    ) -> data_access.CreditRiskPrediction:
        calls.append((application_id, include_explanation))
        return data_access.CreditRiskPrediction(
            status="available",
            probability=0.78,
            risk_level="HIGH",
            model_name="XGBoost",
            model_version="actual-v2",
            prediction_timestamp="2026-11-06T10:00:00Z",
            explanation=explanation,
        )

    monkeypatch.setattr(
        tools.data_access, "get_credit_risk_prediction", prediction
    )

    result = tools.explain_prediction(" 1001 ")

    assert calls == [(1001, True)]
    assert result == {
        "status": "available",
        "application_id": 1001,
        "prediction_probability": 0.78,
        "risk_level": "HIGH",
        "model_name": "XGBoost",
        "model_version": "actual-v2",
        "explanation_method": "tree_shap",
        "output_space": "log_odds",
        "base_value": 0.12,
        "base_probability": 0.53,
        "prediction_margin": 1.26,
        "contribution_sum": 1.14,
        "top_features": [
            {
                "feature": "EXT_SOURCE_2",
                "shap_value": 0.42,
                "effect": "increases_risk",
            },
            {
                "feature": "AMT_CREDIT",
                "shap_value": -0.18,
                "effect": "decreases_risk",
            },
        ],
        "message": "Explanation retrieved successfully.",
        "source_mode": "configured",
    }
    assert "raw_value" not in str(result)
    assert "transformed_feature" not in str(result)
    assert "application_id" not in result["top_features"][0]
    _assert_json(result)


@pytest.mark.parametrize(
    "explanation",
    [
        None,
        {},
        {"method": "tree_shap", "output_space": "log_odds"},
        {
            "method": "tree_shap",
            "output_space": "log_odds",
            "top_features": [{"feature": "AMT_CREDIT", "shap_value": float("nan"), "effect": "increases_risk"}],
        },
        {
            "method": "tree_shap",
            "output_space": "log_odds",
            "top_features": [{"feature": "AMT_CREDIT", "shap_value": 0.2, "effect": "decreases_risk"}],
        },
        {
            "method": "tree_shap",
            "output_space": "log_odds",
            "probability": 0.1,
            "top_features": [],
        },
        {
            "method": "other",
            "output_space": "log_odds",
            "top_features": [],
        },
        {
            "method": "tree_shap",
            "output_space": "log_odds",
            "top_features": [{"feature": "TARGET", "shap_value": 0.2, "effect": "increases_risk"}],
        },
    ],
)
def test_explain_prediction_rejects_malformed_explanations(
    monkeypatch: pytest.MonkeyPatch,
    explanation: object,
) -> None:
    monkeypatch.setattr(
        tools.data_access,
        "get_credit_risk_prediction",
        lambda *_args, **_kwargs: data_access.CreditRiskPrediction(
            status="available",
            probability=0.78,
            explanation=explanation,
        ),
    )

    result = tools.explain_prediction(1001)

    expected_status = (
        "explanation_unavailable" if explanation is None else "invalid"
    )
    assert result["status"] == expected_status
    assert result["top_features"] == []
    if expected_status == "invalid":
        assert result["prediction_probability"] is None
    _assert_json(result)


def test_explain_prediction_rejects_invalid_id_without_api_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        tools.data_access, "get_credit_risk_prediction", forbidden
    )

    result = tools.explain_prediction(True)

    assert result["status"] == "invalid"
    assert result["application_id"] is None
    assert not called
    _assert_json(result)


@pytest.mark.parametrize(
    ("prediction_status", "expected_status"),
    [
        ("not_found", "not_found"),
        ("no_prediction", "no_prediction"),
        ("model_unavailable", "model_unavailable"),
        ("unavailable", "unavailable"),
    ],
)
def test_explain_prediction_preserves_prediction_unavailable_status(
    monkeypatch: pytest.MonkeyPatch,
    prediction_status: str,
    expected_status: str,
) -> None:
    monkeypatch.setattr(
        tools.data_access,
        "get_credit_risk_prediction",
        lambda *_args, **_kwargs: data_access.CreditRiskPrediction(
            status=prediction_status,
            model_name="XGBoost",
            model_version="v2",
        ),
    )

    result = tools.explain_prediction(1001)

    assert result["status"] == expected_status
    assert result["model_name"] == "XGBoost"
    assert result["model_version"] == "v2"
    assert result["top_features"] == []
    assert result["prediction_probability"] is None
    _assert_json(result)


def test_explain_prediction_handles_data_access_and_execution_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise data_access.DataSourceUnavailableError("private connection detail")

    monkeypatch.setattr(
        tools.data_access, "get_credit_risk_prediction", unavailable
    )
    result = tools.explain_prediction(1001)
    assert result["status"] == "unavailable"
    assert "private connection detail" not in str(result)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("private traceback detail")

    monkeypatch.setattr(
        tools.data_access, "get_credit_risk_prediction", unexpected
    )
    result = tools.explain_prediction(1001)
    assert result["status"] == "error"
    assert "private traceback detail" not in str(result)
    assert result["top_features"] == []
    _assert_json(result)


class FakeStructuredRouter:
    def __init__(
        self,
        tool: str,
        tools: list[str] | None = None,
        needs_clarification: bool = False,
    ) -> None:
        self.tool = tool
        self.tools = tools
        self.needs_clarification = needs_clarification

    def invoke(self, _: object) -> dict[str, object]:
        if self.tools is not None:
            return {
                "tools": self.tools,
                "needs_clarification": self.needs_clarification,
            }
        return {"tool": self.tool}


class FakeChatModel:
    def __init__(
        self,
        tool: str,
        response: str = "Grounded response.",
        *,
        tools: list[str] | None = None,
        needs_clarification: bool = False,
    ) -> None:
        self.tool = tool
        self.response = response
        self.tools = tools
        self.needs_clarification = needs_clarification
        self.response_calls = 0

    def with_structured_output(self, _: object) -> FakeStructuredRouter:
        return FakeStructuredRouter(
            self.tool,
            tools=self.tools,
            needs_clarification=self.needs_clarification,
        )

    def invoke(self, _: object) -> object:
        self.response_calls += 1
        return {"content": self.response}


@pytest.mark.parametrize(
    ("tool", "question", "expected_arguments"),
    [
        ("get_dashboard_metrics", "Show dashboard metrics.", {}),
        (
            "get_application_details",
            "Show application details for application ID 1001.",
            {"application_id": 1001},
        ),
        ("get_model_comparison", "Compare XGBoost and LSTM.", {}),
        (
            "get_prediction",
            "Get the prediction for application 1001.",
            {"application_id": 1001},
        ),
        (
            "explain_prediction",
            "Explain the prediction for application 1001.",
            {"application_id": 1001},
        ),
    ],
)
def test_graph_routes_to_only_the_selected_allowed_tool(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    question: str,
    expected_arguments: dict[str, int | str],
) -> None:
    from agent import graph

    called: list[str] = []

    def fake_tool(**_kwargs: object) -> dict[str, object]:
        called.append(tool)
        assert _kwargs == expected_arguments
        if tool == "get_application_details":
            return {"status": "found", "fields": {"SK_ID_CURR": 1001}}
        if tool == "get_prediction":
            return {"status": "unavailable", "message": "No prediction provider."}
        if tool == "explain_prediction":
            return {
                "status": "available",
                "application_id": 1001,
                "top_features": [
                    {
                        "feature": "EXT_SOURCE_2",
                        "shap_value": 0.42,
                        "effect": "increases_risk",
                    }
                ],
            }
        if tool == "get_model_comparison":
            return {"status": "unavailable", "message": "No evaluation results."}
        return {"status": "available", "metrics": {"total_applications": 120}}

    monkeypatch.setitem(graph._ALLOWED_TOOLS, tool, fake_tool)
    model = FakeChatModel(tool, response="Tool result summarized.")

    state = graph.run_agent(question, model=model)  # type: ignore[arg-type]

    assert state["selected_tool"] == tool
    assert state["tool_arguments"] == expected_arguments
    assert called == [tool]
    expected_status = (
        "unavailable"
        if tool in {"get_prediction", "get_model_comparison"}
        else "answered"
    )
    assert state["status"] == expected_status
    assert state["final_response"]
    assert state["messages"][-1] == {
        "role": "assistant",
        "content": state["final_response"],
    }


def test_graph_executes_prediction_and_explanation_once_for_same_applicant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    calls: list[tuple[str, int]] = []

    def prediction(application_id: int) -> dict[str, object]:
        calls.append(("get_prediction", application_id))
        return {
            "status": "available",
            "application_id": application_id,
            "probability": 0.62,
            "risk_level": "HIGH",
        }

    def explanation(application_id: int) -> dict[str, object]:
        calls.append(("explain_prediction", application_id))
        return {
            "status": "available",
            "application_id": application_id,
            "top_features": [
                {
                    "feature": "EXT_SOURCE_2",
                    "shap_value": 0.4,
                    "effect": "increases_risk",
                }
            ],
        }

    monkeypatch.setitem(graph._ALLOWED_TOOLS, "get_prediction", prediction)
    monkeypatch.setitem(graph._ALLOWED_TOOLS, "explain_prediction", explanation)
    model = FakeChatModel(
        "general",
        tools=["get_prediction", "explain_prediction", "get_prediction"],
        response="The retrieved prediction and explanation are consistent.",
    )

    state = graph.run_agent(
        "Predict the credit risk for applicant 100001 and explain why.",
        model=model,  # type: ignore[arg-type]
    )

    assert calls == [
        ("get_prediction", 100001),
        ("explain_prediction", 100001),
    ]
    assert state["status"] == "answered"
    assert state["tool_statuses"] == {
        "get_prediction": "answered",
        "explain_prediction": "answered",
    }
    assert state["tool_results"]["get_prediction"]["application_id"] == 100001
    assert state["tool_results"]["explain_prediction"]["application_id"] == 100001
    assert model.response_calls == 1


def test_graph_preserves_partial_success_when_one_requested_tool_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    monkeypatch.setitem(
        graph._ALLOWED_TOOLS,
        "get_prediction",
        lambda application_id: {
            "status": "available",
            "application_id": application_id,
            "probability": 0.62,
        },
    )

    def fail_explanation(application_id: int) -> dict[str, object]:
        raise RuntimeError(f"failure for {application_id}")

    monkeypatch.setitem(
        graph._ALLOWED_TOOLS, "explain_prediction", fail_explanation
    )
    model = FakeChatModel(
        "general",
        tools=["get_prediction", "explain_prediction"],
        response="A fabricated explanation must not be used.",
    )

    state = graph.run_agent(
        "Predict applicant 100001 and explain it.",
        model=model,  # type: ignore[arg-type]
    )

    assert state["status"] == "partial"
    assert state["tool_statuses"] == {
        "get_prediction": "answered",
        "explain_prediction": "error",
    }
    assert state["tool_results"]["get_prediction"]["probability"] == 0.62
    assert state["tool_results"]["explain_prediction"]["status"] == "error"
    assert "failure for 100001" not in str(state)
    assert "fabricated explanation" not in state["final_response"]
    assert model.response_calls == 0


def test_graph_runs_prediction_and_model_comparison_without_duplicate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    calls: list[str] = []
    monkeypatch.setitem(
        graph._ALLOWED_TOOLS,
        "get_prediction",
        lambda application_id: (
            calls.append(f"prediction:{application_id}")
            or {
                "status": "available",
                "application_id": application_id,
                "probability": 0.62,
            }
        ),
    )
    monkeypatch.setitem(
        graph._ALLOWED_TOOLS,
        "get_model_comparison",
        lambda: (
            calls.append("comparison")
            or {
                "status": "partial",
                "models": {"XGBoost": {"ROC-AUC": 0.76}, "LSTM": None},
                "comparison_available": False,
                "comparison_note": "LSTM evaluation is unavailable.",
            }
        ),
    )
    model = FakeChatModel(
        "general",
        tools=["get_prediction", "get_model_comparison", "get_model_comparison"],
    )

    state = graph.run_agent(
        "Predict applicant 100001 and tell me how the model performs overall.",
        model=model,  # type: ignore[arg-type]
    )

    assert calls == ["prediction:100001", "comparison"]
    assert state["status"] == "partial"
    assert state["tool_statuses"] == {
        "get_prediction": "answered",
        "get_model_comparison": "partial",
    }
    assert "direct model comparison is not justified" in state["final_response"]
    assert model.response_calls == 0


def test_agent_session_context_resolves_followup_and_new_id_replaces_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph
    from agent.state import AgentConversationContext

    calls: list[tuple[str, int]] = []

    def prediction(application_id: int) -> dict[str, object]:
        calls.append(("get_prediction", application_id))
        return {
            "status": "available",
            "application_id": application_id,
            "probability": 0.6,
        }

    def explanation(application_id: int) -> dict[str, object]:
        calls.append(("explain_prediction", application_id))
        return {
            "status": "available",
            "application_id": application_id,
            "top_features": [],
        }

    monkeypatch.setitem(graph._ALLOWED_TOOLS, "get_prediction", prediction)
    monkeypatch.setitem(graph._ALLOWED_TOOLS, "explain_prediction", explanation)
    context = AgentConversationContext()
    first = graph.run_agent(
        "Predict applicant 100001.",
        model=FakeChatModel("get_prediction"),  # type: ignore[arg-type]
        context=context,
    )
    followup = graph.run_agent(
        "Can you explain why?",
        model=FakeChatModel("explain_prediction"),  # type: ignore[arg-type]
        context=context,
    )
    next_applicant = graph.run_agent(
        "What about applicant 100002?",
        model=FakeChatModel("get_prediction"),  # type: ignore[arg-type]
        context=context,
    )

    assert first["tool_result"]["application_id"] == 100001
    assert followup["tool_result"]["application_id"] == 100001
    assert followup["tool_arguments"] == {"application_id": 100001}
    assert next_applicant["tool_arguments"] == {"application_id": 100002}
    assert context.current_application_id == 100002
    assert calls == [
        ("get_prediction", 100001),
        ("explain_prediction", 100001),
        ("get_prediction", 100002),
    ]


def test_agent_session_context_isolation_and_ambiguous_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph
    from agent.state import AgentConversationContext

    calls: list[int] = []

    def explanation(application_id: int) -> dict[str, object]:
        calls.append(application_id)
        return {
            "status": "available",
            "application_id": application_id,
            "top_features": [],
        }

    monkeypatch.setitem(graph._ALLOWED_TOOLS, "explain_prediction", explanation)
    first_session = AgentConversationContext(current_application_id=100001)
    isolated_session = AgentConversationContext()
    no_context = graph.run_agent(
        "Can you explain why?",
        model=FakeChatModel("explain_prediction"),  # type: ignore[arg-type]
        context=isolated_session,
    )
    explicit_reference = graph.run_agent(
        "Explain why for this applicant.",
        model=FakeChatModel("explain_prediction"),  # type: ignore[arg-type]
        context=first_session,
    )

    assert no_context["status"] == "needs_clarification"
    assert no_context["tool_result"] is None
    assert explicit_reference["tool_arguments"] == {"application_id": 100001}
    assert calls == [100001]


def test_invalid_new_application_id_clears_previous_session_context() -> None:
    from agent import graph
    from agent.state import AgentConversationContext

    context = AgentConversationContext(
        current_application_id=100001,
        last_intent="get_prediction",
    )
    state = graph.run_agent(
        "Predict applicant -5.",
        model=FakeChatModel("get_prediction"),  # type: ignore[arg-type]
        context=context,
    )

    assert state["status"] == "invalid_input"
    assert context.current_application_id is None
    assert context.last_intent == "invalid_input"


def test_agent_does_not_reuse_previous_tool_result_as_fresh_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph
    from agent.state import AgentConversationContext

    seen: list[int] = []

    def prediction(application_id: int) -> dict[str, object]:
        seen.append(application_id)
        return {
            "status": "available",
            "application_id": application_id,
            "probability": 0.4 if len(seen) == 1 else 0.7,
        }

    monkeypatch.setitem(graph._ALLOWED_TOOLS, "get_prediction", prediction)
    context = AgentConversationContext()
    first = graph.run_agent(
        "Predict applicant 100001.",
        model=FakeChatModel("get_prediction"),  # type: ignore[arg-type]
        context=context,
    )
    second = graph.run_agent(
        "What about this applicant?",
        model=FakeChatModel("get_prediction"),  # type: ignore[arg-type]
        context=context,
    )

    assert first["tool_result"]["probability"] == 0.4
    assert second["tool_result"]["probability"] == 0.7
    assert seen == [100001, 100001]


@pytest.mark.parametrize(
    ("tool_status", "expected_status"),
    [("available", "answered"), ("success", "success")],
)
def test_graph_executes_prediction_tool_and_preserves_its_result(
    monkeypatch: pytest.MonkeyPatch,
    tool_status: str,
    expected_status: str,
) -> None:
    from agent import graph

    prediction = {
        "status": tool_status,
        "application_id": 1001,
        "probability": 0.25,
        "model_name": "XGBoost",
        "model_version": "v2",
    }
    monkeypatch.setitem(graph._ALLOWED_TOOLS, "get_prediction", lambda **_: prediction)
    model = FakeChatModel("get_prediction", response="Prediction retrieved.")

    state = graph.run_agent(
        "Get prediction for application 1001.", model=model  # type: ignore[arg-type]
    )

    assert state["status"] == expected_status
    assert state["tool_result"] == prediction
    assert state["selected_tool"] == "get_prediction"
    assert model.response_calls == 1


@pytest.mark.parametrize(
    ("tool", "tool_status", "expected_status", "answer_fragment"),
    [
        ("get_prediction", "not_found", "not_found", "No application was found"),
        (
            "get_prediction",
            "no_prediction",
            "no_prediction",
            "No prediction is available",
        ),
        (
            "get_prediction",
            "model_unavailable",
            "model_unavailable",
            "model is unavailable",
        ),
        (
            "explain_prediction",
            "explanation_unavailable",
            "explanation_unavailable",
            "SHAP explanation is unavailable",
        ),
        ("get_prediction", "unavailable", "unavailable", "unavailable"),
        ("get_prediction", "invalid", "invalid", "invalid input"),
        ("get_prediction", "error", "error", "couldn't safely complete"),
    ],
)
def test_graph_preserves_prediction_failure_statuses_without_llm_fabrication(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    tool_status: str,
    expected_status: str,
    answer_fragment: str,
) -> None:
    from agent import graph

    result = {
        "status": tool_status,
        "application_id": 1001,
        "probability": None,
        "message": "The prediction service returned no score.",
    }
    monkeypatch.setitem(graph._ALLOWED_TOOLS, tool, lambda **_: result)
    model = FakeChatModel(
        tool,
        response="A fabricated answer must not be used.",
    )

    state = graph.run_agent(
        "Explain prediction for application 1001.",
        model=model,  # type: ignore[arg-type]
    )

    assert state["status"] == expected_status
    assert state["tool_result"] == result
    assert answer_fragment.lower() in state["final_response"].lower()
    assert "fabricated answer" not in state["final_response"].lower()
    assert model.response_calls == 0


@pytest.mark.parametrize(
    ("tool", "question"),
    [
        ("get_application_details", "Show me the application details."),
        ("get_prediction", "What prediction is available?"),
        ("explain_prediction", "Explain the prediction for this application."),
    ],
)
def test_graph_asks_for_missing_application_id_without_calling_tool(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    question: str,
) -> None:
    from agent import graph

    called = False

    def forbidden_tool(**kwargs: object) -> dict[str, object]:
        nonlocal called
        assert not kwargs
        called = True
        raise AssertionError("Tool must not run without an application ID.")

    monkeypatch.setitem(graph._ALLOWED_TOOLS, tool, forbidden_tool)
    state = graph.run_agent(question, model=FakeChatModel(tool))  # type: ignore[arg-type]

    assert state["status"] == "needs_clarification"
    assert "application ID" in state["final_response"]
    assert not called


@pytest.mark.parametrize("application_id", ["0", "-2", "1.5", "1e4"])
@pytest.mark.parametrize("tool", ["get_prediction", "explain_prediction"])
def test_graph_rejects_invalid_application_ids_before_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
    application_id: str,
    tool: str,
) -> None:
    from agent import graph

    called = False

    def forbidden_tool(**kwargs: object) -> dict[str, object]:
        nonlocal called
        assert not kwargs
        called = True
        raise AssertionError("Invalid application ID reached a tool.")

    monkeypatch.setitem(graph._ALLOWED_TOOLS, tool, forbidden_tool)
    request = (
        "Explain prediction"
        if tool == "explain_prediction"
        else "Get prediction"
    )
    state = graph.run_agent(
        f"{request} for application {application_id}.",
        model=FakeChatModel(tool),  # type: ignore[arg-type]
    )

    assert state["status"] == "invalid_input"
    assert "positive integer" in state["final_response"]
    assert not called


def test_unavailable_tool_result_is_communicated_without_llm_fabrication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    model = FakeChatModel(
        "get_prediction",
        response="This fabricated answer must not be used.",
    )
    def unavailable_prediction(**kwargs: object) -> dict[str, object]:
        assert kwargs == {"application_id": 1001}
        return {
            "status": "unavailable",
            "message": "Inference provider is not configured.",
            "source_mode": "mock",
        }

    monkeypatch.setitem(
        graph._ALLOWED_TOOLS, "get_prediction", unavailable_prediction
    )

    state = graph.run_agent("Prediction for application 1001?", model=model)  # type: ignore[arg-type]

    assert state["status"] == "unavailable"
    assert "unavailable" in state["final_response"]
    assert "mock/demo" in state["final_response"]
    assert "Inference provider is not configured." in state["final_response"]
    assert model.response_calls == 0


def test_partial_results_do_not_let_llm_fill_missing_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    model = FakeChatModel(
        "get_model_comparison",
        response="XGBoost PR-AUC is 0.99 and LSTM ROC-AUC is 0.98.",
    )

    def partial_comparison(**kwargs: object) -> dict[str, object]:
        assert not kwargs
        return {
            "status": "partial",
            "models": {
                "XGBoost": {"ROC-AUC": 0.81, "PR-AUC": None},
                "LSTM": {"ROC-AUC": None, "PR-AUC": None},
            },
            "statuses": {"XGBoost": "Partial", "LSTM": "Not available"},
            "comparison_available": False,
            "comparison_note": "Only XGBoost evaluation results are available.",
            "source_mode": "real",
        }

    monkeypatch.setitem(
        graph._ALLOWED_TOOLS, "get_model_comparison", partial_comparison
    )

    state = graph.run_agent(
        "Compare XGBoost and LSTM.",
        model=model,  # type: ignore[arg-type]
    )

    assert state["status"] == "partial"
    assert "direct model comparison is not justified" in state["final_response"]
    assert "Only XGBoost evaluation results are available." in state["final_response"]
    assert '"PR-AUC": null' in state["final_response"]
    assert "0.99" not in state["final_response"]
    assert "0.98" not in state["final_response"]
    assert model.response_calls == 0


def test_graph_does_not_rank_complete_metrics_without_comparable_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    model = FakeChatModel(
        "get_model_comparison",
        response="LSTM is the winner.",
    )
    result = {
        "status": "available",
        "models": {
            "XGBoost": {"ROC-AUC": 0.8},
            "LSTM": {"ROC-AUC": 0.7},
        },
        "comparison_available": False,
        "comparison_note": "Evaluation protocol metadata is incomplete.",
        "source_mode": "configured",
    }
    monkeypatch.setitem(
        graph._ALLOWED_TOOLS, "get_model_comparison", lambda **_: result
    )

    state = graph.run_agent(
        "Which model performs better?",
        model=model,  # type: ignore[arg-type]
    )

    assert state["tool_result"] == result
    assert "direct model comparison is not justified" in state["final_response"]
    assert "LSTM is the winner" not in state["final_response"]
    assert model.response_calls == 0


def test_tool_exception_is_logged_and_returns_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    def fail(**kwargs: object) -> dict[str, object]:
        assert not kwargs
        raise RuntimeError("sensitive provider detail")

    monkeypatch.setitem(graph._ALLOWED_TOOLS, "get_dashboard_metrics", fail)
    state = graph.run_agent(
        "Show dashboard metrics.",
        model=FakeChatModel("get_dashboard_metrics"),  # type: ignore[arg-type]
    )

    assert state["status"] == "error"
    assert state["error"] == "tool_failed:RuntimeError"
    assert "sensitive provider detail" not in state["final_response"]


def test_unknown_request_does_not_execute_a_tool() -> None:
    from agent import graph

    state = graph.run_agent(
        "Write and run arbitrary Python code.",
        model=FakeChatModel("unsupported"),  # type: ignore[arg-type]
    )

    assert state["status"] == "unsupported_request"
    assert state["selected_tool"] is None
    assert state["tool_result"] is None


def test_general_platform_question_uses_approved_context_only() -> None:
    from agent import graph

    model = FakeChatModel("general", response="Mock data is synthetic demo data.")
    state = graph.run_agent(
        "What does TARGET=1 mean in this platform?",
        model=model,  # type: ignore[arg-type]
    )

    assert state["status"] == "answered"
    assert state["selected_tool"] is None
    assert state["tool_result"] is None
    assert "synthetic" in state["final_response"]
    assert model.response_calls == 1


def test_successful_mock_results_are_labeled_even_if_model_omits_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    def mock_metrics(**kwargs: object) -> dict[str, object]:
        assert not kwargs
        return {
            "status": "available",
            "metrics": {"total_applications": 120},
            "source_mode": "mock",
        }

    monkeypatch.setitem(
        graph._ALLOWED_TOOLS, "get_dashboard_metrics", mock_metrics
    )
    state = graph.run_agent(
        "Show dashboard metrics.",
        model=FakeChatModel(
            "get_dashboard_metrics",
            response="There are 120 applications.",
        ),  # type: ignore[arg-type]
    )

    assert state["status"] == "answered"
    assert state["final_response"].startswith("Mock/demo data:")
    assert "120 applications" in state["final_response"]


def test_routing_and_response_generation_failures_are_explicit() -> None:
    from agent import graph

    class RoutingFailureModel(FakeChatModel):
        def with_structured_output(self, _: object) -> FakeStructuredRouter:
            raise RuntimeError("routing service failure")

    route_failure = graph.run_agent(
        "Show dashboard metrics.",
        model=RoutingFailureModel("get_dashboard_metrics"),  # type: ignore[arg-type]
    )
    assert route_failure["status"] == "error"
    assert route_failure["error"] == "routing_failed:RuntimeError"
    assert "couldn't safely complete" in route_failure["final_response"]

    class ResponseFailureModel(FakeChatModel):
        def invoke(self, _: object) -> object:
            raise RuntimeError("response service failure")

    response_failure = graph.run_agent(
        "What does this platform do?",
        model=ResponseFailureModel("general"),  # type: ignore[arg-type]
    )
    assert response_failure["status"] == "error"
    assert response_failure["error"] == "response_failed:RuntimeError"
    assert "couldn't safely generate" in response_failure["final_response"]


def test_llm_requires_credentials_only_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.graph import AgentConfigurationError, run_agent

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AgentConfigurationError, match="OPENAI_API_KEY"):
        run_agent("What does this platform do?")

    # Injected test models avoid credential lookup and external calls.
    state = run_agent(
        "What does this platform do?",
        model=FakeChatModel("general"),  # type: ignore[arg-type]
    )
    assert state["status"] == "answered"


def test_credit_decision_request_is_refused_without_tool_execution() -> None:
    from agent import graph

    state = graph.run_agent(
        "Should I approve application 1001?",
        model=FakeChatModel("get_application_details"),  # type: ignore[arg-type]
    )

    assert state["status"] == "unsupported_request"
    assert state["selected_tool"] is None