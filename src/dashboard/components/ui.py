"""Small, reusable presentation helpers for the Streamlit dashboard."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import streamlit as st

from dashboard.components.metrics import render_metric

StatusKind = Literal["info", "warning", "error"]


def render_page_header(title: str, description: str) -> None:
    """Render a consistent page title and its short explanation."""
    st.title(title)
    st.caption(description)


def render_status(
    title: str,
    detail: str | None = None,
    *,
    kind: StatusKind = "info",
) -> None:
    """Render a concise, consistent state without exposing provider internals."""
    message = title if not detail else f"{title}\n\n{detail}"
    if kind == "error":
        st.error(message)
    elif kind == "warning":
        st.warning(message)
    else:
        st.info(message)


def render_section_heading(title: str, description: str | None = None) -> None:
    st.subheader(title)
    if description:
        st.caption(description)


def render_metric_grid(
    items: Iterable[tuple[str, Any, str | None]], *, columns: int = 4
) -> None:
    """Render values in a responsive native Streamlit metric row."""
    values = list(items)
    if not values:
        return
    column_count = max(1, min(columns, len(values)))
    metric_columns = st.columns(column_count, gap="medium")
    for index, (label, value, help_text) in enumerate(values):
        with metric_columns[index % column_count]:
            render_metric(label, value, help_text)
