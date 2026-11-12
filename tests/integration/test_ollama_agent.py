"""Opt-in smoke test for local Ollama routing and response generation."""
from __future__ import annotations

import os
from typing import Any

import pytest

from agent import graph

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_INTEGRATION") != "1",
    reason="Set RUN_OLLAMA_INTEGRATION=1 to call the local Ollama server.",
)


@pytest.mark.parametrize(
    ("tool", "question", "expected_arguments", "stub_result", "response_anchor"),
    [
        (
            "get_application_details",
            "What is AMT_CREDIT for application ID 1001?",
            {"application_id": 1001},
            {"status": "found", "application_id": 1001, "fields": {"AMT_CREDIT": 25000}},
            '"value": 25000',
        ),
        (
            "get_dashboard_metrics",
            "Show dashboard metrics.",
            {},
            {"status": "available", "metrics": {"total_applications": 23}},
            "23",
        ),
        (
            "get_model_comparison",
            "Compare the available model evaluation results.",
            {},
            {
                "status": "available",
                "models": {"XGBoost": {"ROC-AUC": 0.81}, "LSTM": {"ROC-AUC": 0.79}},
                "comparison_available": False,
                "comparison_note": "The protocols are not comparable.",
            },
            "not justified",
        ),
        (
            "get_prediction",
            "Get the prediction for application ID 1001.",
            {"application_id": 1001},
            {"status": "available", "application_id": 1001, "probability": 0.37, "risk_level": "MEDIUM"},
            '"probability": 0.37',
        ),
        (
            "explain_prediction",
            "Show the SHAP explanation for application ID 1001.",
            {"application_id": 1001},
            {"status": "available", "application_id": 1001, "top_features": [{"feature": "AMT_CREDIT", "shap_value": 0.2}]},
            "AMT_CREDIT",
        ),
        (
            "generate_report",
            "Generate a 30 day credit risk report.",
            {"period": "30d"},
            {
                "status": "complete",
                "application_statistics": {"status": "available", "total_applications": 23},
                "prediction_statistics": {"status": "available", "prediction_count": 10},
                "model_evaluation": {"status": "available", "XGBoost": {"ROC-AUC": 0.81}},
                "data_quality": {"missing_data": []},
            },
            "23",
        ),
    ],
)
def test_local_ollama_structured_route_executes_each_allowlisted_tool(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    question: str,
    expected_arguments: dict[str, int | str],
    stub_result: dict[str, Any],
    response_anchor: str,
) -> None:
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("AGENT_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls: list[dict[str, int | str]] = []

    def stub_tool(**kwargs: int | str) -> dict[str, Any]:
        calls.append(kwargs)
        return stub_result

    monkeypatch.setitem(graph._ALLOWED_TOOLS, tool, stub_tool)

    result = graph.run_agent(question)

    assert result["selected_tool"] == tool, result
    assert calls == [expected_arguments]
    assert result["tool_result"] == stub_result
    assert result["final_response"]
    assert response_anchor.casefold() in result["final_response"].casefold()


def test_local_ollama_unavailable_tool_result_is_not_fabricated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("AGENT_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(
        graph._ALLOWED_TOOLS,
        "get_dashboard_metrics",
        lambda: {
            "status": "unavailable",
            "metrics": None,
            "message": "Dashboard metrics are unavailable.",
        },
    )

    result = graph.run_agent("Show dashboard metrics.")

    assert result["selected_tool"] == "get_dashboard_metrics"
    assert result["status"] == "unavailable"
    assert result["tool_result"]["metrics"] is None
    assert "unavailable" in result["final_response"].casefold()
    assert "total_applications" not in result["final_response"]


@pytest.mark.parametrize(
    ("report_status", "missing_data", "expected_status"),
    [
        ("complete", [], "success"),
        ("partial", ["model evaluation"], "partial"),
        ("unavailable", ["application statistics", "saved prediction summary", "model evaluation"], "unavailable"),
    ],
)
def test_local_ollama_report_preserves_section_availability(
    monkeypatch: pytest.MonkeyPatch,
    report_status: str,
    missing_data: list[str],
    expected_status: str,
) -> None:
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("AGENT_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    report = {
        "status": report_status,
        "message": "Report data is unavailable." if report_status == "unavailable" else None,
        "application_statistics": {"status": "unavailable" if report_status == "unavailable" else "available", "total_applications": None if report_status == "unavailable" else 23},
        "prediction_statistics": {"status": "unavailable" if report_status == "unavailable" else "available", "prediction_count": None if report_status == "unavailable" else 10},
        "model_evaluation": {"status": "unavailable" if report_status != "complete" else "available", "ROC-AUC": None if report_status != "complete" else 0.81},
        "data_quality": {"missing_data": missing_data},
    }
    monkeypatch.setitem(graph._ALLOWED_TOOLS, "generate_report", lambda **_: report)

    result = graph.run_agent("Generate a 30 day credit risk report.")

    assert result["selected_tool"] == "generate_report"
    assert result["status"] == expected_status
    assert result["tool_result"] == report
    if expected_status == "partial":
        assert '"ROC-AUC": null' in result["final_response"]
    if expected_status == "unavailable":
        assert "unavailable" in result["final_response"].casefold()
