from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_access
from dashboard.components.theme import BLUE, style_chart


def _render_metrics(results: pd.DataFrame) -> None:
    if results.empty:
        st.info("The API returned no evaluation metrics for this split.")
        return
    table = results.loc[:, ["model", "metric", "value"]].copy()
    table["value"] = pd.to_numeric(table["value"], errors="coerce").map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
    )
    st.dataframe(
        table.rename(
            columns={"model": "Model", "metric": "Metric", "value": "Score"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    chart = results.copy()
    chart["value"] = pd.to_numeric(chart["value"], errors="coerce")
    chart = chart.dropna(subset=["value"])
    if chart.empty:
        return
    figure = px.bar(
        chart,
        x="metric",
        y="value",
        color="model",
        barmode="group",
        text=chart["value"].map(lambda value: f"{value:.1%}"),
        color_discrete_sequence=[BLUE],
        title="Saved evaluation metrics",
    )
    figure.update_layout(
        xaxis_title=None,
        yaxis_title="Score",
        yaxis={"range": [0, 1.12], "tickformat": ".0%"},
        legend_title=None,
    )
    figure.update_traces(textposition="outside", cliponaxis=False)
    style_chart(figure, height=400)
    st.plotly_chart(figure, use_container_width=True)


def _render_confusion_matrix(confusion: object, threshold: object) -> None:
    if not isinstance(confusion, dict):
        st.info("No confusion matrix is available in the saved evaluation artifact.")
        return
    rows = [
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Saved confusion matrix at probability threshold {threshold}.")


def render_model_comparison() -> None:
    st.title("Model Performance")
    st.caption(
        "Metrics and model metadata are read from the backend's saved evaluation "
        "artifacts. The dashboard does not calculate substitute scores."
    )
    split = st.selectbox("Evaluation split", ["test", "validation"])

    try:
        with st.spinner("Loading model evaluation from the API..."):
            results = data_access.get_model_evaluation_results(split)
            model = data_access.get_model_info()
    except data_access.DataAccessError as error:
        st.error(f"Evaluation data is unavailable: {error}")
        st.info("No cached or illustrative metrics are shown.")
        return

    provenance = results.attrs.get("provenance", {})
    st.subheader(str(model.get("model_name", "Model")))
    st.write(
        f"Version: `{model.get('model_version', 'unknown')}`  \n"
        f"Training date: {model.get('training_date') or 'not recorded'}  \n"
        f"Model hash: `{model.get('model_hash', 'not recorded')}`"
    )
    if not results.empty:
        _render_metrics(results)
    else:
        st.info(f"No evaluation metrics were recorded for the {split} split.")

    st.subheader("Confusion matrix")
    _render_confusion_matrix(
        results.attrs.get("confusion_matrix"),
        provenance.get("threshold", "not recorded"),
    )

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
            }
        )
    st.info(
        "Only XGBoost validation and test artifacts are currently available. "
        "Training-set metrics and LSTM evaluation results are not present in the API."
    )
