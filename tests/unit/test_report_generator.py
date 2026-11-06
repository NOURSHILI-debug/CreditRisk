from __future__ import annotations

import json

import pandas as pd
import pytest

from agent import report_generator, tools
from dashboard import data_access


class _ReportRouter:
    def __init__(self) -> None:
        self.calls = 0

    def with_structured_output(self, _: object) -> "_ReportRouter":
        return self

    def invoke(self, _: object) -> object:
        self.calls += 1
        if self.calls == 1:
            return {"tool": "generate_report"}
        return type("Response", (), {"content": "The report covers the available sources."})()


def _sources(monkeypatch: pytest.MonkeyPatch) -> None:
    overview = {
        "total_applications": 12,
        "prediction_count": 8,
        "prediction_split": "test",
        "predicted_outcomes": {"no_defined_repayment_difficulty": 5, "repayment_difficulty": 3},
        "risk_distribution": {"LOW": 2, "MEDIUM": 3, "HIGH": 3},
        "source": "test_predictions.csv",
    }
    monkeypatch.setattr(data_access, "get_applications_page", lambda **_: (pd.DataFrame(), overview["total_applications"]))
    monkeypatch.setattr(data_access, "get_prediction_summary", lambda: {
        "prediction_count": 8, "split": "test", "predicted_outcomes": overview["predicted_outcomes"],
        "risk_distribution": overview["risk_distribution"], "source": "test_predictions.csv",
    })
    frame = pd.DataFrame([
        {"model": "XGBoost", "metric": metric, "value": 0.8}
        for metric in data_access.MODEL_METRICS
    ])
    frame.attrs["split"] = "test"
    frame.attrs["provenance"] = {"source": "evaluation_results.json"}
    monkeypatch.setattr(data_access, "get_model_evaluation_results", lambda: frame)
    monkeypatch.setattr(data_access, "get_provider_name", lambda: "fastapi")


@pytest.mark.parametrize("period", ["7d", "30d", "90d", "all"])
def test_generate_report_valid_periods_are_explicit_about_coverage(
    monkeypatch: pytest.MonkeyPatch, period: str
) -> None:
    _sources(monkeypatch)
    result = report_generator.generate_report(period)
    assert result["metadata"]["requested_period"] == period
    assert result["metadata"]["period_filtering_supported"] is False
    assert result["prediction_statistics"]["provenance"] == "saved_historical_split_predictions"
    assert "not a rolling window" in result["metadata"]["actual_coverage"]
    assert result["application_statistics"]["target_counts"] is None
    json.dumps(result, allow_nan=False)


def test_invalid_period_rejected() -> None:
    with pytest.raises(ValueError, match="period must be one of"):
        report_generator.generate_report("30 days")


def test_agent_routes_report_tool_and_passes_period(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import graph

    captured: list[str] = []

    def report(*, period: str) -> dict[str, object]:
        captured.append(period)
        return {"status": "complete", "metadata": {"requested_period": period}}

    monkeypatch.setitem(graph._ALLOWED_TOOLS, "generate_report", report)
    state = graph.run_agent(
        "Generate a report for the last 30 days.",
        model=_ReportRouter(),  # type: ignore[arg-type]
    )
    assert state["selected_tool"] == "generate_report"
    assert captured == ["30d"]
    assert state["tool_result"]["metadata"]["requested_period"] == "30d"


def test_agent_rejects_unsupported_report_period(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import graph

    calls: list[str] = []
    monkeypatch.setitem(graph._ALLOWED_TOOLS, "generate_report", lambda *, period: calls.append(period) or tools.generate_report(period))
    state = graph.run_agent(
        "Generate a report for 14d.",
        model=_ReportRouter(),  # type: ignore[arg-type]
    )
    assert calls == ["14d"]
    assert state["tool_statuses"]["generate_report"] == "invalid"


def test_partial_source_failures_preserve_available_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_access, "get_applications_page", lambda **_: (pd.DataFrame(), 6))
    monkeypatch.setattr(data_access, "get_prediction_summary", lambda: (_ for _ in ()).throw(ConnectionError()))
    monkeypatch.setattr(data_access, "get_model_evaluation_results", lambda: (_ for _ in ()).throw(ConnectionError()))
    result = report_generator.generate_report("all")
    assert result["status"] == "partial"
    assert result["application_statistics"]["total_applications"] == 6
    assert result["prediction_statistics"]["prediction_count"] is None
    assert result["model_evaluation"]["models"]["XGBoost"]["ROC-AUC"] is None
    assert "TARGET class counts" in result["markdown"]


def test_all_sources_unavailable_and_tool_validates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_access, "get_applications_page", lambda **_: (_ for _ in ()).throw(ConnectionError()))
    monkeypatch.setattr(data_access, "get_prediction_summary", lambda: (_ for _ in ()).throw(ConnectionError()))
    monkeypatch.setattr(data_access, "get_model_evaluation_results", lambda: (_ for _ in ()).throw(ConnectionError()))
    assert report_generator.generate_report("all")["status"] == "unavailable"
    assert tools.generate_report("week")["status"] == "invalid"


def test_missing_metrics_remain_null_and_lstm_not_evaluated(monkeypatch: pytest.MonkeyPatch) -> None:
    _sources(monkeypatch)
    frame = pd.DataFrame([{"model": "XGBoost", "metric": "ROC-AUC", "value": 0.8}])
    frame.attrs["split"] = "test"
    frame.attrs["provenance"] = {}
    monkeypatch.setattr(data_access, "get_model_evaluation_results", lambda: frame)
    result = report_generator.generate_report("all")
    assert result["model_evaluation"]["models"]["XGBoost"]["PR-AUC"] is None
    assert result["model_evaluation"]["statuses"]["LSTM"] == "Not available"
    assert result["model_comparison"]["available"] is False

