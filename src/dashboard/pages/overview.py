from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_access
from dashboard.components.metrics import render_metric
from dashboard.components.theme import BLUE, RED, TEAL, style_chart

_RISK_COLORS = {"LOW": TEAL, "MEDIUM": BLUE, "HIGH": RED}


def _render_risk_distribution(summary: dict[str, Any]) -> None:
    distribution = summary.get("risk_distribution")
    if not isinstance(distribution, dict) or not any(distribution.values()):
        st.info("No saved prediction records are available for this split.")
        return
    chart_data = pd.DataFrame(
        [
            {"Risk category": category, "Predictions": count}
            for category, count in distribution.items()
        ]
    )
    fig = px.bar(
        chart_data,
        x="Risk category",
        y="Predictions",
        color="Risk category",
        color_discrete_map=_RISK_COLORS,
        text="Predictions",
        title=f"Risk distribution — {summary['prediction_split']} prediction records",
    )
    fig.update_layout(showlegend=False, yaxis_title="Saved model predictions")
    style_chart(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)


def _render_evaluation(results: pd.DataFrame) -> None:
    if results.empty:
        st.info("No evaluation metrics are available for this split.")
        return
    display = results.loc[:, ["model", "metric", "value"]].copy()
    display["value"] = display["value"].map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
    )
    st.dataframe(
        display.rename(
            columns={"model": "Model", "metric": "Metric", "value": "Score"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_overview() -> None:
    st.title("Overview")
    st.caption(
        "Live portfolio size, saved model predictions, and evaluation results "
        "provided by the local FastAPI service."
    )
    if st.button("Refresh live data"):
        st.rerun()

    try:
        with st.spinner("Loading data from the credit-risk API..."):
            metrics = data_access.get_overview_metrics()
            results = data_access.get_model_evaluation_results(
                metrics["prediction_split"]
            )
            model = data_access.get_model_info()
    except data_access.DataAccessError as error:
        st.error(f"Live data could not be retrieved from the API: {error}")
        st.info(
            "Start the FastAPI service on 127.0.0.1:8000 and refresh this page. "
            "No sample data is shown while the API is unavailable."
        )
        return

    summary = metrics
    distribution = summary["risk_distribution"]
    prediction_count = summary["prediction_count"]
    columns = st.columns(4, gap="large")
    metric_values = (
        ("Applications in dataset", f"{metrics['total_applications']:,}"),
        (f"{summary['prediction_split'].title()} predictions", f"{prediction_count:,}"),
        ("Low-risk predictions", f"{distribution['LOW']:,}"),
        ("High-risk predictions", f"{distribution['HIGH']:,}"),
    )
    for column, (label, value) in zip(columns, metric_values):
        with column:
            render_metric(label, value)

    bands = summary["risk_bands"]
    st.caption(
        f"Risk distribution uses actual saved {summary['prediction_split']} model output "
        f"from `{summary['source']}`; it covers {prediction_count:,} scored "
        "records, not every application in the source dataset. Risk bands are "
        f"provisional model categories (LOW < {bands['low_upper']:.0%}, "
        f"MEDIUM < {bands['medium_upper']:.0%}, HIGH otherwise), not lending decisions."
    )

    risk_column, outcomes_column = st.columns(2, gap="large")
    with risk_column:
        with st.container(border=True):
            st.subheader("Risk distribution")
            _render_risk_distribution(summary)
    with outcomes_column:
        with st.container(border=True):
            st.subheader("Predicted outcome counts")
            outcomes = summary["predicted_outcomes"]
            outcome_data = pd.DataFrame(
                [
                    {
                        "Model outcome": "No defined repayment difficulty",
                        "Predictions": outcomes[
                            "no_defined_repayment_difficulty"
                        ],
                    },
                    {
                        "Model outcome": "Repayment difficulty",
                        "Predictions": outcomes["repayment_difficulty"],
                    },
                ]
            )
            st.dataframe(outcome_data, use_container_width=True, hide_index=True)
            threshold = summary["decision_threshold"]
            threshold_label = (
                f"{threshold:.2f}" if threshold is not None else "unavailable"
            )
            st.caption(
                f"Binary predictions use the saved model decision threshold "
                f"({threshold_label}); these are model "
                "classifications, not approval or rejection decisions."
            )

    st.subheader("Model evaluation")
    st.caption(
        f"Metrics are read from the backend evaluation artifact for the "
        f"{results.attrs.get('split', summary['prediction_split'])} split."
    )
    _render_evaluation(results)

    provenance = results.attrs.get("provenance", {})
    confusion = results.attrs.get("confusion_matrix")
    if isinstance(confusion, dict):
        with st.expander("Confusion matrix"):
            matrix = pd.DataFrame(
                [
                    {
                        "Actual": "No defined repayment difficulty",
                        "Predicted no difficulty": confusion.get("tn"),
                        "Predicted repayment difficulty": confusion.get("fp"),
                    },
                    {
                        "Actual": "Repayment difficulty",
                        "Predicted no difficulty": confusion.get("fn"),
                        "Predicted repayment difficulty": confusion.get("tp"),
                    },
                ]
            )
            st.dataframe(matrix, use_container_width=True, hide_index=True)
            st.caption(
                "Counts are from the saved evaluation confusion matrix at "
                f"threshold {confusion.get('threshold', provenance.get('threshold', 'N/A'))}."
            )

    st.subheader("Model metadata")
    thresholds = model.get("thresholds", {})
    st.json(
        {
            "model_name": model.get("model_name"),
            "model_version": model.get("model_version"),
            "model_hash": model.get("model_hash"),
            "training_date": model.get("training_date"),
            "evaluation_split": results.attrs.get("split"),
            "evaluation_generated_at": provenance.get("generated_at"),
            "decision_threshold": thresholds.get("decision_threshold"),
            "risk_bands": thresholds.get("risk_bands"),
        }
    )
