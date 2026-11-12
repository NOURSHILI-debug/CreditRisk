from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_access
from dashboard.components.theme import BLUE, TEAL, style_chart
from dashboard.components.ui import render_page_header, render_section_heading, render_status

_MODEL_COLORS = {"XGBoost": BLUE, "LSTM": TEAL}
_METRIC_HELP = {
    "ROC-AUC": "Ranking ability across classification thresholds.",
    "PR-AUC": "Precision-recall trade-off, useful when positive cases are less common.",
    "Precision": "Share of predicted positive cases that were positive in the evaluation labels.",
    "Recall": "Share of positive evaluation labels identified by the model.",
    "F1-score": "Harmonic mean of precision and recall.",
}


def _render_metrics(results: pd.DataFrame) -> None:
    if results.empty:
        render_status("No evaluation metrics were produced for this split.")
        return
    table = results.loc[:, ["model", "metric", "value"]].copy()
    table["value"] = pd.to_numeric(table["value"], errors="coerce").map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
    )
    display = table.rename(columns={"model": "Model", "metric": "Metric", "value": "Score"})
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="small"),
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Score": st.column_config.TextColumn("Score", width="small"),
        },
    )

    chart = results.copy()
    chart["value"] = pd.to_numeric(chart["value"], errors="coerce")
    chart = chart.dropna(subset=["value"])
    if not chart.empty:
        figure = px.bar(
            chart,
            x="metric",
            y="value",
            color="model",
            barmode="group",
            text=chart["value"].map(lambda value: f"{value:.1%}"),
            color_discrete_map=_MODEL_COLORS,
            title="Evaluation scores by metric",
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Score",
            yaxis={"range": [0, 1.12], "tickformat": ".0%"},
            legend_title="Model",
        )
        figure.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{fullData.name}<br>%{x}: %{y:.1%}<extra></extra>",
        )
        style_chart(figure, height=370)
        st.plotly_chart(figure, use_container_width=True)

    with st.expander("Metric definitions"):
        for metric, description in _METRIC_HELP.items():
            st.markdown(f"**{metric}.** {description}")


def _render_confusion_matrix(confusion: object, threshold: object) -> None:
    if not isinstance(confusion, dict):
        render_status("No confusion matrix was produced for this evaluation.")
        return
    rows = [
        {
            "Actual class": "No defined repayment difficulty",
            "Predicted: no difficulty": confusion.get("tn"),
            "Predicted: repayment difficulty": confusion.get("fp"),
        },
        {
            "Actual class": "Repayment difficulty",
            "Predicted: no difficulty": confusion.get("fn"),
            "Predicted: repayment difficulty": confusion.get("tp"),
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Rows are actual classes; columns are predicted classes. Saved probability threshold: {threshold}."
    )


def render_model_comparison() -> None:
    render_page_header(
        "Model Performance",
        "Compare saved validation or test results. Only metrics present in the API are displayed.",
    )
    split = st.selectbox("Evaluation split", ["test", "validation"], key="model_evaluation_split")
    render_section_heading(f"{split.title()} evaluation", "Scores and confusion-matrix counts come from the selected saved evaluation artifact.")

    results: pd.DataFrame | None = None
    try:
        with st.spinner("Loading model evaluation..."):
            results = data_access.get_model_evaluation_results(split)
    except data_access.DataAccessError:
        render_status(
            "Evaluation data is unavailable.",
            "No cached or illustrative metrics are shown. Try refreshing after the local API is available.",
            kind="warning",
        )

    if results is not None:
        provenance = results.attrs.get("provenance", {})
        if results.empty:
            render_status(f"No evaluation metrics were recorded for the {split} split.")
        else:
            _render_metrics(results)
        render_section_heading("Confusion matrix")
        _render_confusion_matrix(
            results.attrs.get("confusion_matrix"),
            provenance.get("threshold", "not recorded"),
        )
    else:
        provenance = {}

    render_section_heading("Model and evaluation provenance")
    try:
        model = data_access.get_model_info()
    except data_access.DataAccessError:
        render_status(
            "Model metadata is unavailable.",
            "Evaluation scores remain visible if their artifact was retrieved.",
            kind="warning",
        )
        model = None

    if model is not None:
        st.markdown(
            f"**Model:** {model.get('model_name', 'Not recorded')}  \n"
            f"**Version:** {model.get('model_version', 'Not recorded')}  \n"
            f"**Training date:** {model.get('training_date') or 'Not recorded'}"
        )
    if results is not None:
        with st.expander("Evaluation provenance"):
            st.json(
                {
                    "split": results.attrs.get("split"),
                    "sample_count": provenance.get("n_samples"),
                    "class_counts": provenance.get("class_counts"),
                    "generated_at": provenance.get("generated_at"),
                    "threshold": provenance.get("threshold"),
                    "split_protocol": provenance.get("split_protocol"),
                    "source": provenance.get("source"),
                    "model_hash": model.get("model_hash") if model else None,
                }
            )

    st.caption(
        "Only XGBoost validation and test artifacts are currently available. "
        "No LSTM evaluation scores are present in the API."
    )
