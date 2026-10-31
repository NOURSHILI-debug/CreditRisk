from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_access
from dashboard.components.metrics import render_metric
from dashboard.components.theme import BLUE, RED, TEAL, style_chart

_TARGET_LABELS = {
    0: "No defined repayment difficulty (0)",
    1: "Repayment difficulty (1)",
}
_CATEGORY_COLUMNS = (
    "NAME_INCOME_TYPE",
    "NAME_CONTRACT_TYPE",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
)
_NUMERIC_COLUMNS = ("AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY")


def _target_values(applications: pd.DataFrame) -> pd.Series | None:
    """Return recognized binary target values, or None when no usable target exists."""
    if "TARGET" not in applications.columns:
        return None

    target = pd.to_numeric(applications["TARGET"], errors="coerce")
    target = target.where(target.isin([0, 1])).dropna()
    return target if not target.empty else None


def _summarize_applications(applications: pd.DataFrame) -> dict[str, Any]:
    """Calculate application and target statistics from the supplied data."""
    target = _target_values(applications)
    if target is None:
        return {
            "total": len(applications),
            "target_0": None,
            "target_1": None,
            "difficulty_rate": None,
            "unlabeled": len(applications),
        }

    target_0 = int(target.eq(0).sum())
    target_1 = int(target.eq(1).sum())
    labeled_count = target_0 + target_1

    return {
        "total": len(applications),
        "target_0": target_0,
        "target_1": target_1,
        "difficulty_rate": target_1 / labeled_count if labeled_count else None,
        "unlabeled": len(applications) - labeled_count,
    }


def _target_distribution(applications: pd.DataFrame) -> pd.DataFrame:
    """Build count and percentage rows for the two defined target values."""
    target = _target_values(applications)
    if target is None:
        return pd.DataFrame(columns=["Target", "Applications", "Share"])

    counts = target.value_counts().reindex([0, 1], fill_value=0)
    labeled_count = int(counts.sum())
    return pd.DataFrame(
        [
            {
                "Target": _TARGET_LABELS[target_value],
                "Applications": int(counts[target_value]),
                "Share": float(counts[target_value] / labeled_count),
            }
            for target_value in (0, 1)
        ]
    )


def _category_distribution(applications: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return category counts, retaining missing values as an explicit category."""
    if column not in applications.columns or applications.empty:
        return pd.DataFrame(columns=["Category", "Applications", "Share"])

    categories = applications[column].astype("string").fillna("Missing")
    counts = categories.value_counts(dropna=False)
    result = counts.rename_axis("Category").reset_index(name="Applications")
    result["Share"] = result["Applications"] / len(applications)

    if len(result) > 10:
        top_categories = result.nlargest(9, "Applications")
        other_count = int(result.loc[~result["Category"].isin(top_categories["Category"]), "Applications"].sum())
        result = top_categories.copy()
        result.loc[len(result)] = {
            "Category": "Other",
            "Applications": other_count,
            "Share": other_count / len(applications),
        }

    return result.sort_values("Applications", ascending=True)


def _target_by_category(applications: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summarize target counts by a categorical application feature."""
    target = _target_values(applications)
    if target is None or column not in applications.columns:
        return pd.DataFrame(columns=["Category", "TARGET", "Applications"])

    data = pd.DataFrame(
        {
            "Category": applications[column].astype("string").fillna("Missing"),
            "TARGET": pd.to_numeric(applications["TARGET"], errors="coerce"),
        },
        index=applications.index,
    )
    data = data.loc[data["TARGET"].isin([0, 1])]
    if data.empty:
        return pd.DataFrame(columns=["Category", "TARGET", "Applications"])

    result = (
        data.groupby(["Category", "TARGET"], observed=True)
        .size()
        .rename("Applications")
        .reset_index()
    )
    result["TARGET"] = result["TARGET"].map(_TARGET_LABELS)
    return result


def _numeric_values(applications: pd.DataFrame, column: str) -> pd.Series:
    """Return finite numeric values for a field, dropping invalid and missing data."""
    if column not in applications.columns:
        return pd.Series(dtype="float64")

    values = pd.to_numeric(applications[column], errors="coerce")
    return values.loc[np.isfinite(values)]


def _render_target_distribution(applications: pd.DataFrame) -> None:
    distribution = _target_distribution(applications)
    if distribution.empty:
        st.info("Target distribution is unavailable because no valid TARGET values are present.")
        return

    fig = px.bar(
        distribution,
        x="Target",
        y="Applications",
        color="Target",
        color_discrete_map={
            _TARGET_LABELS[0]: TEAL,
            _TARGET_LABELS[1]: RED,
        },
        custom_data=["Share"],
        text=distribution.apply(
            lambda row: f"{row['Applications']:,} · {row['Share']:.1%}",
            axis=1,
        ),
        title="Target distribution",
    )
    fig.update_traces(
        hovertemplate="%{x}<br>Applications: %{y:,}<br>Share: %{customdata[0]:.1%}<extra></extra>",
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Applications")
    style_chart(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)


def _render_category_distribution(applications: pd.DataFrame, column: str) -> None:
    distribution = _category_distribution(applications, column)
    if distribution.empty:
        st.info("No categorical application field is available for this distribution.")
        return

    fig = px.bar(
        distribution,
        x="Applications",
        y="Category",
        orientation="h",
        color_discrete_sequence=[BLUE],
        custom_data=["Share"],
        text=distribution.apply(
            lambda row: f"{row['Applications']:,} · {row['Share']:.1%}",
            axis=1,
        ),
        title=f"Applications by {column.replace('NAME_', '').replace('_', ' ').title()}",
    )
    fig.update_traces(
        hovertemplate="%{y}<br>Applications: %{x:,}<br>Share: %{customdata[0]:.1%}<extra></extra>",
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(xaxis_title="Applications", yaxis_title=None)
    style_chart(fig, height=max(320, 52 * len(distribution)))
    st.plotly_chart(fig, use_container_width=True)


def _render_target_by_category(applications: pd.DataFrame, column: str) -> None:
    distribution = _target_by_category(applications, column)
    if distribution.empty:
        st.info("Target-by-category distribution is unavailable for the current data.")
        return

    fig = px.bar(
        distribution,
        x="Category",
        y="Applications",
        color="TARGET",
        barmode="group",
        color_discrete_map={
            _TARGET_LABELS[0]: TEAL,
            _TARGET_LABELS[1]: RED,
        },
        title=f"Repayment difficulty by {column.replace('NAME_', '').replace('_', ' ').title()}",
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Applications", legend_title=None)
    fig.update_traces(
        hovertemplate="%{x}<br>%{fullData.name}<br>Applications: %{y:,}<extra></extra>"
    )
    style_chart(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


def _render_numeric_distribution(applications: pd.DataFrame, column: str) -> None:
    values = _numeric_values(applications, column)
    if values.empty:
        st.info(f"{column.replace('AMT_', '').replace('_', ' ').title()} distribution is unavailable.")
        return

    chart_data = pd.DataFrame({column: values})
    fig = px.histogram(
        chart_data,
        x=column,
        nbins=20,
        color_discrete_sequence=[BLUE],
        title=f"{column.replace('AMT_', '').replace('_', ' ').title()} distribution",
    )
    fig.update_layout(xaxis_title=column.replace("AMT_", "").replace("_", " ").title(), yaxis_title="Applications")
    fig.update_traces(hovertemplate="Amount: %{x:$,.0f}<br>Applications: %{y}<extra></extra>")
    style_chart(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


def render_overview() -> None:
    """Render dynamic application and repayment-difficulty statistics."""
    applications = data_access.get_applications()
    is_mock = data_access.MOCK_MODE

    st.title("Overview")
    st.caption("Application portfolio and repayment-difficulty statistics from the configured data source.")

    if is_mock:
        st.warning(
            "**DEMO DATA — Synthetic applications.** All figures on this page are illustrative and do not "
            "represent actual Home Credit applicants or model results."
        )
    else:
        st.info(
            "**CONFIGURED DATA SOURCE.** Figures are calculated from the configured application provider; "
            "verify source provenance before interpreting them as real-world results."
        )

    summary = _summarize_applications(applications)
    target_available = summary["target_0"] is not None
    metric_columns = st.columns(4, gap="large")
    with metric_columns[0]:
        render_metric(
            "Total applications",
            f"{summary['total']:,}",
            help_text="Number of application records returned by the configured data source.",
        )
    with metric_columns[1]:
        render_metric(
            "TARGET = 0",
            f"{summary['target_0']:,}" if target_available else "N/A",
            help_text="No defined repayment difficulties.",
        )
    with metric_columns[2]:
        render_metric(
            "TARGET = 1",
            f"{summary['target_1']:,}" if target_available else "N/A",
            help_text="Repayment difficulties.",
        )
    with metric_columns[3]:
        rate = summary["difficulty_rate"]
        render_metric(
            "Repayment-difficulty rate",
            f"{rate:.1%}" if rate is not None else "N/A",
            help_text="TARGET = 1 divided by applications with a valid TARGET value.",
        )

    if summary["unlabeled"]:
        st.caption(
            f"TARGET is missing or invalid for {summary['unlabeled']:,} of {summary['total']:,} applications; "
            "target percentages use only records with TARGET = 0 or TARGET = 1."
        )

    if applications.empty:
        st.info("No applications are available from the configured data source.")
        return

    st.subheader("Application and target distributions")
    target_column, category_column = st.columns(2, gap="large")
    with target_column:
        _render_target_distribution(applications)
    with category_column:
        categorical_field = next(
            (column for column in _CATEGORY_COLUMNS if column in applications.columns),
            None,
        )
        if categorical_field is None:
            st.info("No supported categorical application field is available.")
        else:
            _render_category_distribution(applications, categorical_field)

    st.subheader("Credit-risk patterns")
    available_category = next(
        (column for column in _CATEGORY_COLUMNS if column in applications.columns),
        None,
    )
    available_amount = next(
        (column for column in _NUMERIC_COLUMNS if column in applications.columns),
        None,
    )
    risk_columns = st.columns(2, gap="large")
    with risk_columns[0]:
        if available_category is not None:
            _render_target_by_category(applications, available_category)
        else:
            st.info("Target-by-category analysis is unavailable without a categorical application field.")
    with risk_columns[1]:
        if available_amount is not None:
            _render_numeric_distribution(applications, available_amount)
        else:
            st.info("Credit amount distribution is unavailable because no supported numeric field is present.")
