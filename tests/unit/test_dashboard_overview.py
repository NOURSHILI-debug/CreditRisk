from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard import data_access
from dashboard.pages.overview import render_overview


def _summary() -> dict[str, object]:
    return {
        "total_applications": 307511,
        "prediction_count": 46127,
        "prediction_split": "test",
        "predicted_outcomes": {
            "no_defined_repayment_difficulty": 33972,
            "repayment_difficulty": 12155,
        },
        "risk_distribution": {"LOW": 10, "MEDIUM": 20, "HIGH": 30},
        "risk_bands": {"low_upper": 0.3, "medium_upper": 0.6},
        "decision_threshold": 0.5,
        "source": "test_predictions.csv",
    }


def _evaluation() -> pd.DataFrame:
    results = pd.DataFrame(
        [
            {"model": "XGBoost", "metric": "ROC-AUC", "value": 0.75},
            {"model": "XGBoost", "metric": "PR-AUC", "value": 0.24},
        ]
    )
    results.attrs["split"] = "test"
    results.attrs["provenance"] = {
        "n_samples": 46127,
        "threshold": 0.5,
        "generated_at": "2026-11-04T12:00:00Z",
        "lstm_available": False,
    }
    results.attrs["confusion_matrix"] = {
        "tn": 10,
        "fp": 2,
        "fn": 3,
        "tp": 4,
    }
    return results


def _health() -> dict[str, object]:
    return {"status": "ok", "data_available": True, "model_loaded": True}


def _unavailable(*_args, **_kwargs):
    raise data_access.DataSourceUnavailableError("API unavailable")


def test_overview_displays_only_api_provided_values(monkeypatch) -> None:
    monkeypatch.setattr(
        data_access,
        "get_api_health",
        lambda: {
            "status": "degraded",
            "data_available": True,
            "model_loaded": False,
        },
    )
    monkeypatch.setattr(data_access, "get_overview_metrics", _summary)
    monkeypatch.setattr(
        data_access, "get_model_evaluation_results", lambda split: _evaluation()
    )
    monkeypatch.setattr(
        data_access,
        "get_model_info",
        lambda: {
            "model_name": "XGBoost",
            "model_version": "baseline-v2-local",
            "model_hash": "real-artifact-hash",
            "training_date": "2026-11-03",
            "thresholds": {"decision_threshold": 0.5, "risk_bands": {}},
        },
    )
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from dashboard.pages.overview import render_overview\n"
        "render_overview()\n"
    ).run(timeout=30)
    assert not app.exception
    metric_values = {metric.label: metric.value for metric in app.metric}
    assert metric_values["Applications available"] == "307,511"
    assert metric_values["Test prediction records"] == "46,127"
    assert metric_values["Low-risk predictions"] == "10"
    assert metric_values["High-risk predictions"] == "30"
    assert "TARGET" not in " ".join(
        element.value for element in app.subheader
    )
    assert any("test_predictions.csv" in item.value for item in app.caption)
    assert "Connected" in [item.value for item in app.markdown]
    assert "Available" in [item.value for item in app.markdown]
    assert "Unavailable" in [item.value for item in app.markdown]
    assert "LSTM evaluation: not available" in " ".join(item.value for item in app.caption)


def test_overview_api_unavailable_has_no_fallback_data(monkeypatch) -> None:
    def unavailable():
        raise data_access.DataSourceUnavailableError(
            "Credit-risk API connection refused"
        )

    monkeypatch.setattr(data_access, "get_api_health", unavailable)
    monkeypatch.setattr(data_access, "get_overview_metrics", unavailable)
    monkeypatch.setattr(data_access, "get_model_evaluation_results", _unavailable)
    monkeypatch.setattr(data_access, "get_model_info", unavailable)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from dashboard.pages.overview import render_overview\n"
        "render_overview()\n"
    ).run()
    assert not app.exception
    assert not app.metric
    assert any("Portfolio summary is unavailable" in item.value for item in app.warning)
    assert "No values are substituted" in app.warning[0].value


def test_evaluation_failure_does_not_hide_available_portfolio_summary(monkeypatch) -> None:
    monkeypatch.setattr(data_access, "get_api_health", _health)
    monkeypatch.setattr(data_access, "get_overview_metrics", _summary)

    def unavailable(_split: str):
        raise data_access.DataSourceUnavailableError("private API detail")

    monkeypatch.setattr(data_access, "get_model_evaluation_results", unavailable)
    monkeypatch.setattr(
        data_access,
        "get_model_info",
        lambda: {"model_name": "XGBoost", "thresholds": {}},
    )
    app = AppTest.from_string(
        "from dashboard.pages.overview import render_overview\nrender_overview()\n"
    ).run()

    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric}["Applications available"] == "307,511"
    assert any("Evaluation metrics are unavailable" in item.value for item in app.warning)
    assert "private API detail" not in str(app)


def test_model_metadata_failure_does_not_hide_summary_or_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(data_access, "get_api_health", _health)
    monkeypatch.setattr(data_access, "get_overview_metrics", _summary)
    monkeypatch.setattr(data_access, "get_model_evaluation_results", lambda _split: _evaluation())

    def unavailable():
        raise data_access.DataSourceUnavailableError("private metadata error")

    monkeypatch.setattr(data_access, "get_model_info", unavailable)
    app = AppTest.from_string(
        "from dashboard.pages.overview import render_overview\nrender_overview()\n"
    ).run()

    assert not app.exception
    assert any(metric.label == "Applications available" for metric in app.metric)
    assert len(app.dataframe) >= 2
    assert any("Model metadata is unavailable" in item.value for item in app.warning)
    assert "private metadata error" not in str(app)


def test_empty_evaluation_is_not_shown_as_zero(monkeypatch) -> None:
    monkeypatch.setattr(data_access, "get_api_health", _health)
    monkeypatch.setattr(data_access, "get_overview_metrics", _summary)
    monkeypatch.setattr(data_access, "get_model_info", lambda: {"model_name": "XGBoost"})
    empty = pd.DataFrame(columns=["model", "metric", "value"])
    empty.attrs["split"] = "test"
    empty.attrs["provenance"] = {"lstm_available": False}
    monkeypatch.setattr(data_access, "get_model_evaluation_results", lambda _split: empty)

    app = AppTest.from_string(
        "from dashboard.pages.overview import render_overview\nrender_overview()\n"
    ).run()

    assert not app.exception
    assert any("No metrics were produced" in item.value for item in app.caption)
    assert "0.0%" not in str(app)
    assert "LSTM evaluation: not available" in " ".join(item.value for item in app.caption)
