from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_access
from dashboard.components.theme import BLUE, RED, TEAL, style_chart
from dashboard.components.ui import (
    render_metric_grid,
    render_page_header,
    render_section_heading,
    render_status,
)

_RISK_COLORS = {"LOW": TEAL, "MEDIUM": BLUE, "HIGH": RED}


def _render_risk_distribution(summary: dict[str, Any]) -> None:
    distribution = summary.get("risk_distribution")
    if not isinstance(distribution, dict) or not any(distribution.values()):
        render_status(
            "No saved prediction records are available for this split.",
            "Risk distribution will appear when the API supplies prediction records.",
        )
        return

    order = [category for category in ("LOW", "MEDIUM", "HIGH") if category in distribution]
    chart_data = pd.DataFrame(
        [
            {"Risk category": category, "Prediction records": distribution[category]}
            for category in order
        ]
    )
    fig = px.bar(
        chart_data,
        x="Risk category",
        y="Prediction records",
        color="Risk category",
        category_orders={"Risk category": order},
        color_discrete_map=_RISK_COLORS,
        text="Prediction records",
        title=f"Risk categories · {summary['prediction_split'].title()} split",
    )
    fig.update_layout(showlegend=False, yaxis_title="Saved prediction records")
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>%{y:,} records<extra></extra>")
    style_chart(fig, height=330)
    st.plotly_chart(fig, use_container_width=True)


def _render_evaluation(results: pd.DataFrame) -> None:
    if results.empty:
        render_status("No evaluation metrics were produced for this split.")
        return
    display = results.loc[:, ["model", "metric", "value"]].copy()
    display["value"] = display["value"].map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
    )
    st.dataframe(
        display.rename(columns={"model": "Model", "metric": "Metric", "value": "Score"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="small"),
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Score": st.column_config.TextColumn("Score", width="small"),
        },
    )


def _render_confusion_matrix(confusion: object, provenance: dict[str, Any]) -> None:
    if not isinstance(confusion, dict):
        render_status("No confusion matrix was produced for this evaluation.")
        return
    matrix = pd.DataFrame(
        [
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
    )
    st.dataframe(matrix, use_container_width=True, hide_index=True)
    threshold = confusion.get("threshold", provenance.get("threshold"))
    threshold_label = f"{threshold:.3g}" if isinstance(threshold, (int, float)) else "not recorded"
    st.caption(f"Actual classes are rows; model predictions are columns. Saved probability threshold: {threshold_label}.")


def render_overview() -> None:
    render_page_header(
        "Overview",
        "Portfolio coverage, saved prediction outputs, and model evaluation from the local API.",
    )

    metrics: dict[str, Any] | None = None
    try:
        with st.spinner("Loading portfolio summary..."):
            # This existing adapter combines application count and prediction summary.
            metrics = data_access.get_overview_metrics()
    except data_access.DataAccessError:
        render_status(
            "Portfolio summary is unavailable.",
            "Application and prediction totals could not be retrieved. Check that both sources are available in the local API. No values are substituted.",
            kind="warning",
        )

    render_section_heading(
        "Portfolio summary",
        "Prediction counts and risk bands describe saved model output, not every application in the dataset.",
    )
    if metrics is not None:
        summary = metrics
        distribution = summary["risk_distribution"]
        prediction_count = summary["prediction_count"]
        render_metric_grid(
            [
                (
                    "Applications available",
                    f"{metrics['total_applications']:,}",
                    "Application records available through the API.",
                ),
                (
                    f"{summary['prediction_split'].title()} prediction records",
                    f"{prediction_count:,}",
                    "Rows represented in the saved prediction artifact; not the full application dataset.",
                ),
                (
                    "Low-risk predictions",
                    f"{distribution['LOW']:,}",
                    "Saved records in the current artifact's LOW model band.",
                ),
                (
                    "High-risk predictions",
                    f"{distribution['HIGH']:,}",
                    "Saved records in the current artifact's HIGH model band.",
                ),
            ]
        )
        bands = summary["risk_bands"]
        st.caption(
            f"Source: `{summary['source']}`. Provisional model bands: LOW < "
            f"{bands['low_upper']:.0%}, MEDIUM < {bands['medium_upper']:.0%}, "
            "HIGH otherwise. These categories are not lending decisions."
        )
        risk_column, outcomes_column = st.columns(2, gap="large")
        with risk_column:
            render_section_heading("Risk distribution")
            _render_risk_distribution(summary)
        with outcomes_column:
            render_section_heading("Predicted outcomes")
            outcomes = summary["predicted_outcomes"]
            outcome_data = pd.DataFrame(
                [
                    {"Model outcome": "No defined repayment difficulty", "Records": outcomes["no_defined_repayment_difficulty"]},
                    {"Model outcome": "Repayment difficulty", "Records": outcomes["repayment_difficulty"]},
                ]
            )
            st.dataframe(outcome_data, use_container_width=True, hide_index=True)
            threshold = summary["decision_threshold"]
            threshold_label = f"{threshold:.2f}" if threshold is not None else "unavailable"
            st.caption(
                f"Counts use the saved model threshold ({threshold_label}). These are model classifications, not credit decisions."
            )

    render_section_heading("Evaluation performance")
    evaluation_split = metrics["prediction_split"] if metrics else "test"
    results: pd.DataFrame | None = None
    try:
        with st.spinner("Loading saved evaluation metrics..."):
            results = data_access.get_model_evaluation_results(evaluation_split)
    except data_access.DataAccessError:
        render_status(
            "Evaluation metrics are unavailable.",
            "No cached or illustrative scores are shown.",
            kind="warning",
        )
    if results is not None:
        split_label = results.attrs.get("split", evaluation_split)
        st.caption(f"Saved evaluation artifact · {str(split_label).title()} split")
        _render_evaluation(results)
        provenance = results.attrs.get("provenance", {})
        confusion = results.attrs.get("confusion_matrix")
        if isinstance(confusion, dict):
            with st.expander("Confusion matrix"):
                _render_confusion_matrix(confusion, provenance)
    else:
        provenance = {}

    render_section_heading("Model metadata")
    try:
        model = data_access.get_model_info()
    except data_access.DataAccessError:
        render_status(
            "Model metadata is unavailable.",
            "The other Overview sections remain available when their data sources respond.",
            kind="warning",
        )
        return
    thresholds = model.get("thresholds", {})
    metadata = {
        "Model": model.get("model_name"),
        "Version": model.get("model_version"),
        "Artifact hash": model.get("model_hash"),
        "Training date": model.get("training_date"),
        "Evaluation split": results.attrs.get("split") if results is not None else None,
        "Evaluation generated": provenance.get("generated_at"),
        "Decision threshold": thresholds.get("decision_threshold"),
        "Risk bands": thresholds.get("risk_bands"),
    }
    with st.expander("View metadata and evaluation provenance"):
        st.json(metadata)
