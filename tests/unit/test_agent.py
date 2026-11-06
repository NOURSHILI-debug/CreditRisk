from __future__ import annotations

import json
import sys
from types import ModuleType

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
    def __init__(self, tool: str, captured: list[object] | None = None) -> None:
        self.tool = tool
        self.captured = captured

    def invoke(self, messages: object) -> dict[str, str]:
        if self.captured is not None:
            self.captured.append(messages)
        return {"tool": self.tool}


class FakeChatModel:
    def __init__(self, tool: str, response: str = "Grounded response.") -> None:
        self.tool = tool
        self.response = response
        self.response_calls = 0
        self.route_inputs: list[object] = []
        self.response_inputs: list[object] = []

    def with_structured_output(self, _: object, **_kwargs: object) -> FakeStructuredRouter:
        return FakeStructuredRouter(self.tool, self.route_inputs)

    def invoke(self, messages: object) -> object:
        self.response_calls += 1
        self.response_inputs.append(messages)
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
    assert model.response_calls == 0
    assert "0.25" in state["final_response"]
    assert "1001" not in state["final_response"]


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


def test_ollama_provider_initializes_without_openai_key_and_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    captured: dict[str, object] = {}

    class FakeOllama:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    module = ModuleType("langchain_ollama")
    module.ChatOllama = FakeOllama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_ollama", module)
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    model = graph._create_chat_model()

    assert isinstance(model, FakeOllama)
    assert captured == {
        "model": "llama3.1:8b",
        "base_url": "http://localhost:11434",
        "temperature": 0,
    }


def test_ollama_provider_accepts_model_and_base_url_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    captured: dict[str, object] = {}

    class FakeOllama:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    module = ModuleType("langchain_ollama")
    module.ChatOllama = FakeOllama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_ollama", module)
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("AGENT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")

    graph._create_chat_model()

    assert captured["model"] == "llama3.1:8b-instruct-q4_K_M"
    assert captured["base_url"] == "http://127.0.0.1:11435"
    assert captured["temperature"] == 0


@pytest.mark.parametrize(
    ("url", "allow_remote", "expected"),
    [
        ("http://localhost:11434", "false", "http://localhost:11434"),
        ("http://127.0.0.1:11435/", "false", "http://127.0.0.1:11435"),
        ("http://[::1]:11434", "false", "http://[::1]:11434"),
        ("https://ollama.example.test", "true", "https://ollama.example.test"),
    ],
)
def test_ollama_endpoint_policy_accepts_local_and_explicit_https_remote(
    url: str, allow_remote: str, expected: str
) -> None:
    from agent.graph import _validated_ollama_base_url

    assert _validated_ollama_base_url(url, allow_remote=allow_remote) == expected


@pytest.mark.parametrize(
    ("url", "allow_remote"),
    [
        ("ollama.example.test", "false"),
        ("https://ollama.example.test", "false"),
        ("http://ollama.example.test", "true"),
        ("ftp://localhost:11434", "true"),
        ("http://localhost:99999", "false"),
        ("http://localhost:abc", "false"),
        ("http://user:password@localhost:11434", "false"),
        ("http://localhost:11434/api", "false"),
        ("http://localhost:11434?token=secret", "false"),
        ("http://localhost.example.test:11434", "false"),
        ("https://ollama..example.test", "true"),
        ("https://ollama.example.test:", "true"),
    ],
)
def test_ollama_endpoint_policy_rejects_unsafe_or_malformed_urls(
    url: str, allow_remote: str
) -> None:
    from agent.graph import AgentConfigurationError, _validated_ollama_base_url

    with pytest.raises(AgentConfigurationError):
        _validated_ollama_base_url(url, allow_remote=allow_remote)


def test_ollama_remote_setting_rejects_unrecognized_values() -> None:
    from agent.graph import AgentConfigurationError, _validated_ollama_base_url

    with pytest.raises(AgentConfigurationError, match="OLLAMA_ALLOW_REMOTE"):
        _validated_ollama_base_url("http://localhost:11434", allow_remote="yes")


@pytest.mark.parametrize(
    ("tool", "question", "result", "visible"),
    [
        (
            "get_application_details",
            "What is AMT_CREDIT for application ID 1001?",
            {
                "status": "found",
                "application_id": 1001,
                "fields": {
                    "SK_ID_CURR": 1001,
                    "AMT_CREDIT": 45000,
                    "AMT_INCOME_TOTAL": 90000,
                    "DAYS_BIRTH": -12000,
                },
            },
            '"value": 45000',
        ),
        (
            "get_prediction",
            "Get the prediction for application ID 1001.",
            {
                "status": "available",
                "application_id": 1001,
                "probability": 0.37,
                "risk_level": "MEDIUM",
            },
            '"probability": 0.37',
        ),
        (
            "explain_prediction",
            "Explain the prediction for application ID 1001.",
            {
                "status": "available",
                "application_id": 1001,
                "top_features": [{"feature": "AMT_CREDIT", "shap_value": 0.2}],
            },
            '"feature": "AMT_CREDIT"',
        ),
    ],
)
@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_applicant_results_are_minimized_and_never_passed_to_response_llm(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    question: str,
    result: dict[str, object],
    visible: str,
    provider: str,
) -> None:
    from agent import graph

    monkeypatch.setenv("AGENT_LLM_PROVIDER", provider)
    monkeypatch.setitem(graph._ALLOWED_TOOLS, tool, lambda **_: result)
    model = FakeChatModel(tool)

    state = graph.run_agent(question, model=model)  # type: ignore[arg-type]

    assert state["tool_result"] == result
    assert visible in state["final_response"]
    assert "1001" not in str(model.route_inputs)
    assert "1001" not in str(model.response_inputs)
    assert "90000" not in state["final_response"]
    assert "-12000" not in state["final_response"]
    assert model.response_calls == 0


def test_general_application_lookup_does_not_return_the_full_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    result = {
        "status": "found",
        "application_id": 1001,
        "fields": {"AMT_CREDIT": 45000, "AMT_INCOME_TOTAL": 90000},
    }
    monkeypatch.setitem(
        graph._ALLOWED_TOOLS, "get_application_details", lambda **_: result
    )
    model = FakeChatModel("get_application_details")

    state = graph.run_agent(
        "Show application details for application ID 1001.", model=model  # type: ignore[arg-type]
    )

    assert '"status": "not_authorized"' in state["final_response"]
    assert "45000" not in state["final_response"]
    assert "90000" not in state["final_response"]
    assert model.response_calls == 0


def test_all_six_agent_tools_remain_registered_and_allowlisted() -> None:
    from agent import graph

    assert set(graph._ALLOWED_TOOLS) == {
        "get_application_details",
        "get_dashboard_metrics",
        "get_model_comparison",
        "get_prediction",
        "explain_prediction",
        "generate_report",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"tool": "delete_all_records"},
        {"tool": "get_dashboard_metrics", "tools": ["get_application_details"]},
        {"tool": 123},
    ],
)
def test_malformed_or_unauthorized_routing_fails_without_fallback(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
) -> None:
    from agent import graph

    class StructuredRouter:
        def invoke(self, _: object) -> dict[str, object]:
            return payload

    class MalformedModel:
        def with_structured_output(self, _: object, **_kwargs: object) -> StructuredRouter:
            return StructuredRouter()

        def invoke(self, _: object) -> object:
            raise AssertionError("Response generation must not run after bad routing.")

    called: list[str] = []
    for name in graph._ALLOWED_TOOLS:
        monkeypatch.setitem(
            graph._ALLOWED_TOOLS,
            name,
            lambda **_: called.append(name) or {"status": "available"},
        )

    result = graph.run_agent(
        "Show dashboard metrics.", model=MalformedModel()  # type: ignore[arg-type]
    )

    assert result["status"] == "error"
    assert result["selected_tool"] is None
    assert called == []


def test_allowlist_rejects_a_tool_removed_from_registry_without_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    calls: list[str] = []
    monkeypatch.delitem(graph._ALLOWED_TOOLS, "get_dashboard_metrics")
    monkeypatch.setitem(
        graph._ALLOWED_TOOLS,
        "get_application_details",
        lambda **_: calls.append("application") or {"status": "found", "fields": {}},
    )

    result = graph.run_agent(
        "Show dashboard metrics.",
        model=FakeChatModel("get_dashboard_metrics"),  # type: ignore[arg-type]
    )

    assert result["status"] == "needs_clarification"
    assert result["selected_tool"] is None
    assert calls == []


def test_ollama_route_uses_json_schema_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    class OllamaRouter:
        def __init__(self) -> None:
            self.schema: object | None = None
            self.options: dict[str, object] = {}

        def with_structured_output(
            self, schema: object, **kwargs: object
        ) -> FakeStructuredRouter:
            self.schema = schema
            self.options = kwargs
            return FakeStructuredRouter("get_dashboard_metrics")

    model = OllamaRouter()
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "ollama")

    decision = graph._decision_from_model(model, "Show dashboard metrics.")  # type: ignore[arg-type]

    assert decision.tool == "get_dashboard_metrics"
    assert model.schema is graph.RouteDecision
    assert model.options == {"method": "json_schema"}


def test_route_decision_normalizes_redundant_general_marker() -> None:
    from agent.graph import RouteDecision

    decision = RouteDecision(
        tool="general", tools=["get_dashboard_metrics"]
    )

    assert decision.tool is None
    assert decision.tools == ["get_dashboard_metrics"]


def test_route_decision_normalizes_only_an_exact_duplicated_tool() -> None:
    from agent.graph import RouteDecision

    decision = RouteDecision(
        tool="get_application_details", tools=["get_application_details"]
    )

    assert decision.tool == "get_application_details"
    assert decision.tools == []


def test_unsupported_route_cannot_include_tools() -> None:
    from agent.graph import RouteDecision

    with pytest.raises(ValueError, match="Unsupported routes cannot select tools"):
        RouteDecision(tool="unsupported", tools=["get_dashboard_metrics"])


def test_openai_provider_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import graph

    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    module = ModuleType("langchain_openai")
    module.ChatOpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    model = graph._create_chat_model()

    assert isinstance(model, FakeOpenAI)
    assert captured == {
        "model": "gpt-4o-mini",
        "api_key": "test-key",
        "temperature": 0,
    }


@pytest.mark.parametrize("provider", ["", "gemini", "unknown"])
def test_unsupported_llm_provider_is_rejected(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    from agent.graph import AgentConfigurationError, _create_chat_model

    monkeypatch.setenv("AGENT_LLM_PROVIDER", provider)
    with pytest.raises(AgentConfigurationError, match="supported providers"):
        _create_chat_model()


def test_credit_decision_request_is_refused_without_tool_execution() -> None:
    from agent import graph

    state = graph.run_agent(
        "Should I approve application 1001?",
        model=FakeChatModel("get_application_details"),  # type: ignore[arg-type]
    )

    assert state["status"] == "unsupported_request"
    assert state["selected_tool"] is None
