from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard import data_access
from dashboard.pages.model_comparison import render_model_comparison


def _results() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"model": "XGBoost", "metric": "ROC-AUC", "value": 0.75},
            {"model": "XGBoost", "metric": "PR-AUC", "value": 0.24},
            {"model": "XGBoost", "metric": "Precision", "value": 0.19},
            {"model": "XGBoost", "metric": "Recall", "value": 0.57},
            {"model": "XGBoost", "metric": "F1-score", "value": 0.29},
        ]
    )
    frame.attrs["split"] = "test"
    frame.attrs["provenance"] = {
        "threshold": 0.5,
        "n_samples": 46127,
        "class_counts": {"0": 42403, "1": 3724},
        "generated_at": "2026-11-04T12:00:00Z",
        "source": "evaluation_results.json",
    }
    frame.attrs["confusion_matrix"] = {
        "tn": 33397,
        "fp": 9006,
        "fn": 1570,
        "tp": 2154,
    }
    return frame


def test_model_comparison_renders_real_metrics_metadata_and_confusion_matrix(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        data_access, "get_model_evaluation_results", lambda split: _results()
    )
    monkeypatch.setattr(
        data_access,
        "get_model_info",
        lambda: {
            "model_name": "XGBoost",
            "model_version": "baseline-v2-local",
            "model_hash": "artifact-hash",
            "training_date": "2026-11-03",
        },
    )
    app = AppTest.from_string(
        "from dashboard.pages.model_comparison import render_model_comparison\n"
        "render_model_comparison()\n"
    ).run()
    assert not app.exception
    assert "XGBoost" in [item.value for item in app.subheader]
    assert any("baseline-v2-local" in item.value for item in app.markdown)
    assert len(app.dataframe) == 2
    assert "ROC-AUC" in str(app.dataframe[0].value)
    assert app.dataframe[1].value.iloc[0, 1] == 33397


def test_model_comparison_reports_api_unavailable_without_mock_metrics(
    monkeypatch,
) -> None:
    def unavailable(split: str):
        raise data_access.DataSourceUnavailableError(
            "Evaluation API connection refused"
        )

    monkeypatch.setattr(
        data_access, "get_model_evaluation_results", unavailable
    )
    app = AppTest.from_string(
        "from dashboard.pages.model_comparison import render_model_comparison\n"
        "render_model_comparison()\n"
    ).run()
    assert not app.exception
    assert any(
        "Evaluation API connection refused" in error.value
        for error in app.error
    )
    assert any("No cached or illustrative metrics" in item.value for item in app.info)


def test_evaluation_normalizer_does_not_fill_missing_model_scores() -> None:
    values, statuses, issues = data_access.normalize_model_evaluation_results(
        pd.DataFrame(
            [{"model": "XGBoost", "metric": "ROC-AUC", "value": 0.75}]
        )
    )
    assert values["XGBoost"]["ROC-AUC"] == 0.75
    assert values["XGBoost"]["PR-AUC"] is None
    assert statuses["XGBoost"].startswith("Partially available")
    assert statuses["LSTM"] == "Not available"
    assert issues == []
