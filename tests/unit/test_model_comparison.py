from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard.pages.model_comparison import (
    METRICS,
    MODELS,
    _metric_table,
    _normalize_results,
)

_VALID_ROWS = [
    {"model": model, "metric": metric, "value": index / 10}
    for model in MODELS
    for index, metric in enumerate(METRICS, start=1)
]


def test_both_models_with_valid_metrics_are_available() -> None:
    values, statuses, issues = _normalize_results(pd.DataFrame(_VALID_ROWS))

    assert statuses == {"XGBoost": "Available", "LSTM": "Available"}
    assert values["XGBoost"]["ROC-AUC"] == 0.1
    assert values["LSTM"]["F1-score"] == 0.5
    assert not issues

    table = _metric_table(values, statuses)
    assert table.loc[0, "ROC-AUC"] == "10.0%"
    assert table.loc[1, "F1-score"] == "50.0%"


def test_only_xgboost_results_marks_lstm_unavailable() -> None:
    results = pd.DataFrame([row for row in _VALID_ROWS if row["model"] == "XGBoost"])

    values, statuses, _ = _normalize_results(results)

    assert statuses == {"XGBoost": "Available", "LSTM": "Not available"}
    assert all(value is None for value in values["LSTM"].values())


def test_only_lstm_results_marks_xgboost_unavailable() -> None:
    results = pd.DataFrame([row for row in _VALID_ROWS if row["model"] == "LSTM"])

    values, statuses, _ = _normalize_results(results)

    assert statuses == {"XGBoost": "Not available", "LSTM": "Available"}
    assert all(value is None for value in values["XGBoost"].values())


def test_none_and_empty_results_produce_unavailable_models() -> None:
    for results in (None, pd.DataFrame(columns=["model", "metric", "value"])):
        values, statuses, issues = _normalize_results(results)

        assert statuses == {"XGBoost": "Not available", "LSTM": "Not available"}
        assert all(
            value is None
            for model_values in values.values()
            for value in model_values.values()
        )
        assert not issues


def test_partial_results_keep_missing_metrics_unavailable() -> None:
    results = pd.DataFrame(
        [
            {"model": "XGBoost", "metric": "roc_auc", "value": 0.87},
            {"model": "XGBoost", "metric": "precision", "value": 0.66},
        ]
    )

    values, statuses, _ = _normalize_results(results)

    assert statuses == {
        "XGBoost": "Partially available (2/5 metrics)",
        "LSTM": "Not available",
    }
    assert values["XGBoost"]["ROC-AUC"] == 0.87
    assert values["XGBoost"]["PR-AUC"] is None
    table = _metric_table(values, statuses)
    assert table.loc[0, "PR-AUC"] == "N/A"


def test_null_non_numeric_non_finite_and_out_of_range_values_are_invalid() -> None:
    results = pd.DataFrame(
        [
            {"model": "XGBoost", "metric": "roc_auc", "value": None},
            {"model": "XGBoost", "metric": "pr_auc", "value": "bad"},
            {"model": "XGBoost", "metric": "precision", "value": float("inf")},
            {"model": "XGBoost", "metric": "recall", "value": -0.1},
            {"model": "XGBoost", "metric": "f1_score", "value": 1.1},
        ]
    )

    values, statuses, issues = _normalize_results(results)

    assert statuses == {"XGBoost": "Invalid or incomplete results", "LSTM": "Not available"}
    assert all(value is None for value in values["XGBoost"].values())
    assert len(issues) == 5


def test_duplicate_metric_is_n_a_and_unexpected_model_is_reported() -> None:
    results = pd.DataFrame(
        [
            {"model": "XGBoost", "metric": "roc_auc", "value": 0.8},
            {"model": "XGBoost", "metric": "ROC-AUC", "value": 0.8},
            {"model": "RandomForest", "metric": "recall", "value": 0.7},
        ]
    )

    values, statuses, issues = _normalize_results(results)

    assert values["XGBoost"]["ROC-AUC"] is None
    assert statuses["XGBoost"] == "Invalid or incomplete results"
    assert statuses["LSTM"] == "Not available"
    assert any("Duplicate evaluation rows" in issue for issue in issues)
    assert any("Unrecognized model identifier" in issue for issue in issues)


def test_page_shows_unavailable_notice_and_does_not_render_empty_chart() -> None:
    app = _render_page(None)

    assert not app.exception
    assert any("Evaluation results not yet available" in item.value for item in app.warning)
    assert app.session_state["captured_charts"] == []
    assert len(app.dataframe) == 1
    assert "N/A" in app.dataframe[0].value.to_string()
    assert app.dataframe[0].value["Availability"].tolist() == [
        "Not available",
        "Not available",
    ]


def test_empty_results_dataframe_shows_unavailable_state() -> None:
    app = _render_page(pd.DataFrame(columns=["model", "metric", "value"]))

    assert not app.exception
    assert any("Evaluation results not yet available" in item.value for item in app.warning)
    assert app.session_state["captured_charts"] == []


def test_page_shows_single_available_model_without_fake_second_model() -> None:
    for model in MODELS:
        results = pd.DataFrame([row for row in _VALID_ROWS if row["model"] == model])
        app = _render_page(results)

        assert not app.exception
        table = app.dataframe[0].value.set_index("Model")
        other_model = "LSTM" if model == "XGBoost" else "XGBoost"
        assert table.loc[model, "Availability"] == "Available"
        assert table.loc[other_model, "Availability"] == "Not available"
        chart = app.session_state["captured_charts"][0]
        assert {trace["name"] for trace in chart["data"]} == {model}


def test_page_displays_partial_metrics_and_keeps_unavailable_values_as_na() -> None:
    results = pd.DataFrame(
        [
            {"model": "XGBoost", "metric": "roc_auc", "value": 0.87},
            {"model": "XGBoost", "metric": "precision", "value": 0.66},
        ]
    )
    app = _render_page(results)

    assert not app.exception
    table = app.dataframe[0].value.set_index("Model")
    assert table.loc["XGBoost", "Availability"] == "Partially available (2/5 metrics)"
    assert table.loc["XGBoost", "ROC-AUC"] == "87.0%"
    assert table.loc["XGBoost", "PR-AUC"] == "N/A"
    chart = app.session_state["captured_charts"][0]
    assert {
        metric
        for trace in chart["data"]
        for metric in trace["x"]
    } == {"ROC-AUC", "Precision"}


def test_page_displays_both_models_with_valid_metrics() -> None:
    app = _render_page(pd.DataFrame(_VALID_ROWS))

    assert not app.exception
    table = app.dataframe[0].value.set_index("Model")
    assert table["Availability"].to_dict() == {
        "XGBoost": "Available",
        "LSTM": "Available",
    }
    chart = app.session_state["captured_charts"][0]
    assert {trace["name"] for trace in chart["data"]} == {"XGBoost", "LSTM"}


def test_page_renders_only_available_scores_in_chart() -> None:
    results = pd.DataFrame(
        [
            {"model": "XGBoost", "metric": "roc_auc", "value": 0.84},
            {"model": "LSTM", "metric": "pr_auc", "value": 0.71},
        ]
    )
    app = _render_page(results)
    assert not app.exception
    assert not app.exception
    chart = app.session_state["captured_charts"][0]
    assert len(chart["data"]) == 2
    assert {trace["name"] for trace in chart["data"]} == {"XGBoost", "LSTM"}
    assert {trace["x"][0] for trace in chart["data"]} == {"ROC-AUC", "PR-AUC"}


def test_page_reports_malformed_contract_without_crashing() -> None:
    app = _render_page(pd.DataFrame([{"model": "XGBoost", "score": 0.8}]))

    assert not app.exception
    assert any("missing required columns" in item.value for item in app.warning)
    assert app.session_state["captured_charts"] == []


def _render_page(results: pd.DataFrame | None) -> AppTest:
    serialized_rows = results.to_json(orient="records") if results is not None else "null"
    app_source = (
        "import json\n"
        "import pandas as pd\n"
        "import streamlit as st\n"
        "from dashboard.pages import model_comparison\n"
        "captured_charts = []\n"
        "st.plotly_chart = lambda figure, **kwargs: captured_charts.append(figure.to_dict())\n"
        f"rows = json.loads({serialized_rows!r})\n"
        "result = None if rows is None else pd.DataFrame(rows)\n"
        "model_comparison.get_model_evaluation_results = lambda: result\n"
        "model_comparison.render_model_comparison()\n"
        "st.session_state['captured_charts'] = captured_charts\n"
    )
    return AppTest.from_string(app_source).run()
