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


def _render_metrics(
    results: pd.DataFrame,
    *,
    comparison_available: bool,
) -> None:
    values, statuses, issues = data_access.normalize_model_evaluation_results(results)
    table = pd.DataFrame(
        [
            {"model": model, "metric": metric, "value": values[model][metric]}
            for model in ("XGBoost", "LSTM")
            for metric in data_access.MODEL_METRICS
        ]
    )
    table["value"] = table["value"].map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "Not available"
    )
    display = table.rename(columns={"model": "Model", "metric": "Metric", "value": "Score"})
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="small"),
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Score": st.column_config.TextColumn("Score", width="small"),
        },
    )

    for model, status in statuses.items():
        st.caption(f"{model}: {status}")
    for issue in issues:
        st.warning(issue)

    # A grouped chart visually implies a head-to-head comparison. Show it only
    # when the data-access contract confirms compatible evaluation provenance.
    if not comparison_available:
        return
    chart = table.copy()
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
        st.plotly_chart(figure, width="stretch")

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
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        f"Rows are actual classes; columns are predicted classes. Saved probability threshold: {threshold}."
    )


def render_model_comparison() -> None:
    mock_mode = data_access.is_mock_mode()
    render_page_header(
        "Model Performance",
        "Review the selected evaluation results and their provenance.",
    )
    if mock_mode:
        st.warning(
            "Synthetic mock applications are for UI demonstration only. No trained-model evaluation metrics are available in mock mode."
        )
    split = st.selectbox("Evaluation split", ["test", "validation"], key="model_evaluation_split")
    render_section_heading(
        f"{split.title()} evaluation",
        "No trained-model evaluation metrics are provided by the synthetic mock provider."
        if mock_mode
        else "Scores and confusion-matrix counts come from the selected saved evaluation artifact.",
    )

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
        raw_provenance = results.attrs.get("provenance", {})
        provenance = raw_provenance if isinstance(raw_provenance, dict) else {}
        _, statuses, _ = data_access.normalize_model_evaluation_results(results)
        comparison_available, comparison_note = data_access.model_comparison_validity(
            statuses,
            results.attrs.get("split"),
            provenance,
            synthetic=mock_mode,
        )
        render_section_heading("Individual model results")
        st.info(
            "A direct XGBoost/LSTM comparison is available."
            if comparison_available
            else f"No valid head-to-head comparison: {comparison_note}"
        )
        if results.empty:
            render_status(f"No evaluation metrics were recorded for the {split} split.")
        _render_metrics(results, comparison_available=comparison_available)
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
        "Synthetic mock applications are illustrative only; no trained model evaluation results are available."
        if mock_mode
        else "Only XGBoost validation and test artifacts are currently available. "
        "No LSTM evaluation scores are present in the API."
    )
