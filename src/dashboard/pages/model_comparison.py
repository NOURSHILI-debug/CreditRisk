from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.theme import BLUE, TEAL, style_chart
from dashboard.data_access import get_model_metrics


def render_model_comparison() -> None:
    """Render an initial comparison page for the XGBoost and LSTM models."""
    st.title("Model Comparison")
    st.caption("Illustrative model metrics for the dashboard scaffolding. These values are synthetic.")

    metrics = get_model_metrics()
    if not metrics:
        st.warning("No model metrics are available yet. Add evaluation outputs when the training pipeline is ready.")
        return

    model_rows = []
    for model_name, metric_map in metrics.items():
        for metric_name, metric_value in metric_map.items():
            if metric_name == "source":
                continue
            model_rows.append({
                "Model": model_name,
                "Metric": metric_name,
                "Value": float(metric_value),
            })

    comparison_df = pd.DataFrame(model_rows)
    with st.container(border=True):
        st.subheader("Evaluation metrics")
        st.caption("Higher values indicate stronger performance for each metric; shown values are simulated.")
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model"),
                "Metric": st.column_config.TextColumn("Metric"),
                "Value": st.column_config.NumberColumn("Value", format="%.3f"),
            },
        )

    metrics_of_interest = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-score"]
    for row_start in range(0, len(metrics_of_interest), 2):
        chart_columns = st.columns(2, gap="large")
        for column, metric_name in zip(chart_columns, metrics_of_interest[row_start:row_start + 2]):
            chart_df = pd.DataFrame(
                [
                    {"Model": model_name, metric_name: float(metric_values.get(metric_name, 0.0))}
                    for model_name, metric_values in metrics.items()
                ]
            )
            fig = px.bar(
                chart_df,
                x="Model",
                y=metric_name,
                color="Model",
                title=f"{metric_name} comparison",
                range_y=[0.0, 1.0],
                color_discrete_map={"XGBoost": BLUE, "LSTM": TEAL},
                text_auto=".2f",
            )
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=metric_name)
            fig.update_traces(textposition="outside", cliponaxis=False)
            style_chart(fig, height=320)
            with column:
                st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Mock model metrics are displayed for interface validation only. They are not actual evaluation results "
        "from a trained XGBoost or LSTM model."
    )
