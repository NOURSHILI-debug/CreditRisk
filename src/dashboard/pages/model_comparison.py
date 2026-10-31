from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.theme import BLUE, TEAL, style_chart
from dashboard.data_access import get_model_evaluation_results

MODELS = ("XGBoost", "LSTM")
METRICS = ("ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-score")
_MODEL_IDS = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}
_METRIC_IDS = {
    "rocauc": "ROC-AUC",
    "prauc": "PR-AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1score": "F1-score",
}
_MODEL_COLORS = {"XGBoost": BLUE, "LSTM": TEAL}


def _identifier(value: object) -> str:
    """Normalize external model and metric IDs for matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _normalize_results(
    results: pd.DataFrame | None,
) -> tuple[dict[str, dict[str, float | None]], dict[str, str], list[str]]:
    """Validate canonical evaluation rows and report availability by model."""
    values = {model: {metric: None for metric in METRICS} for model in MODELS}
    supplied_models: set[str] = set()
    issues: list[str] = []

    if results is None or results.empty:
        return values, {model: "Not available" for model in MODELS}, issues

    required_columns = {"model", "metric", "value"}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        issues.append(
            "Evaluation results are missing required columns: "
            + ", ".join(sorted(missing_columns))
            + ". Expected model, metric, and value."
        )
        return values, {model: "Invalid results" for model in MODELS}, issues

    occurrences: Counter[tuple[str, str]] = Counter()
    parsed_values: dict[tuple[str, str], float | None] = {}

    for row in results.loc[:, ["model", "metric", "value"]].itertuples(index=False):
        model = _MODEL_IDS.get(_identifier(row.model))
        if model is None:
            issues.append(f"Unrecognized model identifier: {row.model!r}.")
            continue

        supplied_models.add(model)
        metric = _METRIC_IDS.get(_identifier(row.metric))
        if metric is None:
            issues.append(f"Unrecognized metric identifier for {model}: {row.metric!r}.")
            continue

        key = (model, metric)
        occurrences[key] += 1
        try:
            numeric_value = float(row.value)
        except (TypeError, ValueError):
            numeric_value = math.nan

        parsed_values[key] = (
            numeric_value
            if math.isfinite(numeric_value) and 0.0 <= numeric_value <= 1.0
            else None
        )

    for (model, metric), count in occurrences.items():
        if count > 1:
            parsed_values[(model, metric)] = None
            issues.append(f"Duplicate evaluation rows for {model} / {metric}; value marked N/A.")
        elif parsed_values[(model, metric)] is None:
            issues.append(
                f"Invalid or missing value for {model} / {metric}; expected a finite score from 0 to 1."
            )
        else:
            values[model][metric] = parsed_values[(model, metric)]

    statuses: dict[str, str] = {}
    for model in MODELS:
        available_count = sum(value is not None for value in values[model].values())
        if model not in supplied_models:
            statuses[model] = "Not available"
        elif available_count == len(METRICS):
            statuses[model] = "Available"
        elif available_count:
            statuses[model] = f"Partially available ({available_count}/{len(METRICS)} metrics)"
        else:
            statuses[model] = "Invalid or incomplete results"

    return values, statuses, issues


def _metric_table(
    values: dict[str, dict[str, float | None]],
    statuses: dict[str, str],
) -> pd.DataFrame:
    """Format all required metrics while preserving unavailable values as N/A."""
    rows = []
    for model in MODELS:
        row: dict[str, str] = {"Model": model, "Availability": statuses[model]}
        row.update(
            {
                metric: f"{value:.1%}" if value is not None else "N/A"
                for metric, value in values[model].items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=["Model", "Availability", *METRICS])


def _render_comparison_chart(values: dict[str, dict[str, float | None]]) -> None:
    rows = [
        {"Metric": metric, "Model": model, "Score": value}
        for model in MODELS
        for metric, value in values[model].items()
        if value is not None
    ]
    if not rows:
        st.info("Comparison chart unavailable until valid evaluation metrics are supplied.")
        return

    chart_data = pd.DataFrame(rows)
    fig = px.bar(
        chart_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        category_orders={"Metric": list(METRICS)},
        color_discrete_map=_MODEL_COLORS,
        text=chart_data["Score"].map(lambda value: f"{value:.1%}"),
        custom_data=["Model"],
        title="Evaluation metrics by model",
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<br>%{x}: %{y:.1%}<extra></extra>",
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Score",
        yaxis={"range": [0, 1.12], "tickformat": ".0%"},
        legend_title=None,
    )
    style_chart(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison() -> None:
    """Render supplied XGBoost and LSTM evaluation results without fabricated values."""
    st.title("Model Comparison")
    st.caption("Compare evaluation metrics supplied by the model evaluation pipeline.")

    raw_results = get_model_evaluation_results()
    results = raw_results if isinstance(raw_results, pd.DataFrame) else None
    values, statuses, issues = _normalize_results(results)
    valid_result_count = sum(
        value is not None
        for model_values in values.values()
        for value in model_values.values()
    )

    if valid_result_count:
        st.success("Evaluation results supplied through the dashboard data-access interface.")
    else:
        st.warning(
            "**Evaluation results not yet available.** No evaluation output is configured. "
            "No model scores are being shown as actual results."
        )

    if issues:
        with st.expander("Data quality notes"):
            for issue in dict.fromkeys(issues):
                st.warning(issue)

    model_columns = st.columns(2, gap="large")
    for column, model in zip(model_columns, MODELS):
        with column:
            with st.container(border=True):
                st.subheader(model)
                status = statuses[model]
                if status == "Available":
                    st.success(status)
                elif status.startswith("Partially available"):
                    st.warning(status)
                elif status == "Not available":
                    st.info(status)
                else:
                    st.error(status)

    with st.container(border=True):
        st.subheader("Evaluation metrics")
        st.caption("All metrics are shown as percentages. Missing or invalid values are marked N/A.")
        st.dataframe(
            _metric_table(values, statuses),
            use_container_width=True,
            hide_index=True,
        )

    _render_comparison_chart(values)

    st.caption(
        "Expected data-access interface: one row per model and metric with columns "
        "`model`, `metric`, and `value` (a finite score from 0 to 1)."
    )
