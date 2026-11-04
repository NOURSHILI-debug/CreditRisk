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
    assert result["message"] == "evaluation source offline"
    _assert_json(result)


def test_prediction_preserves_task_9_contract_and_unavailable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools.data_access,
        "get_credit_risk_prediction",
        lambda _: data_access.CreditRiskPrediction(
            status="available",
            probability=0.25,
            model_name="XGBoost",
            model_version="v2",
            prediction_timestamp="2026-11-03T10:00:00Z",
            explanation={"details": "provider supplied"},
        ),
    )
    result = tools.get_prediction("1001")
    assert result == {
        "status": "available",
        "probability": 0.25,
        "model_name": "XGBoost",
        "model_version": "v2",
        "prediction_timestamp": "2026-11-03T10:00:00Z",
        "explanation": {"details": "provider supplied"},
        "message": None,
        "source_mode": "configured",
    }
    _assert_json(result)

    monkeypatch.setattr(
        tools.data_access,
        "get_credit_risk_prediction",
        lambda _: data_access.CreditRiskPrediction(
            status="unavailable", message="no prediction provider"
        ),
    )
    unavailable = tools.get_prediction(1001)
    assert unavailable["status"] == "unavailable"
    assert unavailable["probability"] is None
    assert unavailable["message"] == "no prediction provider"
    _assert_json(unavailable)


def test_prediction_preserves_invalid_malformed_and_invalid_id_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools.data_access,
        "get_credit_risk_prediction",
        lambda _: data_access.CreditRiskPrediction(
            status="invalid", message="Malformed provider response."
        ),
    )
    malformed = tools.get_prediction(1001)
    assert malformed["status"] == "invalid"
    assert malformed["message"] == "Malformed provider response."
    _assert_json(malformed)

    monkeypatch.setattr(
        tools.data_access,
        "get_credit_risk_prediction",
        lambda _: data_access.CreditRiskPrediction(
            status="invalid", message="A valid application ID is required."
        ),
    )
    invalid_id = tools.get_prediction(True)
    assert invalid_id["status"] == "invalid"
    _assert_json(invalid_id)


def test_prediction_reports_data_access_errors_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_: object) -> data_access.CreditRiskPrediction:
        raise data_access.DataSourceUnavailableError("inference offline")

    monkeypatch.setattr(
        tools.data_access, "get_credit_risk_prediction", fail
    )
    result = tools.get_prediction(1001)
    assert result["status"] == "unavailable"
    assert result["message"] == "inference offline"
    _assert_json(result)


class FakeStructuredRouter:
    def __init__(self, tool: str) -> None:
        self.tool = tool

    def invoke(self, _: object) -> dict[str, str]:
        return {"tool": self.tool}


class FakeChatModel:
    def __init__(self, tool: str, response: str = "Grounded response.") -> None:
        self.tool = tool
        self.response = response
        self.response_calls = 0

    def with_structured_output(self, _: object) -> FakeStructuredRouter:
        return FakeStructuredRouter(self.tool)

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
    ("tool", "question"),
    [
        ("get_application_details", "Show me the application details."),
        ("get_prediction", "What prediction is available?"),
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
def test_graph_rejects_invalid_application_ids_before_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
    application_id: str,
) -> None:
    from agent import graph

    called = False

    def forbidden_tool(**kwargs: object) -> dict[str, object]:
        nonlocal called
        assert not kwargs
        called = True
        raise AssertionError("Invalid application ID reached a tool.")

    monkeypatch.setitem(
        graph._ALLOWED_TOOLS, "get_prediction", forbidden_tool
    )
    state = graph.run_agent(
        f"Get prediction for application {application_id}.",
        model=FakeChatModel("get_prediction"),  # type: ignore[arg-type]
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
    assert "Missing values are unavailable" in state["final_response"]
    assert '"PR-AUC": null' in state["final_response"]
    assert "0.99" not in state["final_response"]
    assert "0.98" not in state["final_response"]
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