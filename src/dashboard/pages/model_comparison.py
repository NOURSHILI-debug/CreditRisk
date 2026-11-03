from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_access
from dashboard.components.theme import BLUE, TEAL, style_chart
from dashboard.data_access import get_model_evaluation_results

MODELS = data_access.MODEL_NAMES
METRICS = data_access.MODEL_METRICS
_MODEL_COLORS = {"XGBoost": BLUE, "LSTM": TEAL}


def _normalize_results(
    results: pd.DataFrame | None,
) -> tuple[dict[str, dict[str, float | None]], dict[str, str], list[str]]:
    """Compatibility wrapper for shared evaluation-result validation."""
    return data_access.normalize_model_evaluation_results(results)


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

    provider_error = None
    try:
        raw_results = get_model_evaluation_results()
    except data_access.DataAccessError as error:
        raw_results = None
        provider_error = str(error)
    results = raw_results if isinstance(raw_results, pd.DataFrame) else None
    values, statuses, issues = _normalize_results(results)
    if provider_error is not None:
        issues.append(provider_error)
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
    if provider_error is not None:
        st.error(f"Evaluation data is unavailable: {provider_error}")

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
