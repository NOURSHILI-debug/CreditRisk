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
    }
    results.attrs["confusion_matrix"] = {
        "tn": 10,
        "fp": 2,
        "fn": 3,
        "tp": 4,
    }
    return results


def test_overview_displays_only_api_provided_values(monkeypatch) -> None:
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
    ).run()
    assert not app.exception
    metric_values = {metric.label: metric.value for metric in app.metric}
    assert metric_values["Applications in dataset"] == "307,511"
    assert metric_values["Test predictions"] == "46,127"
    assert metric_values["Low-risk predictions"] == "10"
    assert metric_values["High-risk predictions"] == "30"
    assert "TARGET" not in " ".join(
        element.value for element in app.subheader
    )
    assert any("test_predictions.csv" in item.value for item in app.caption)


def test_overview_api_unavailable_has_no_fallback_data(monkeypatch) -> None:
    def unavailable():
        raise data_access.DataSourceUnavailableError(
            "Credit-risk API connection refused"
        )

    monkeypatch.setattr(data_access, "get_overview_metrics", unavailable)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from dashboard.pages.overview import render_overview\n"
        "render_overview()\n"
    ).run()
    assert not app.exception
    assert any(
        "Credit-risk API connection refused" in error.value for error in app.error
    )
    assert any("No sample data is shown" in item.value for item in app.info)
