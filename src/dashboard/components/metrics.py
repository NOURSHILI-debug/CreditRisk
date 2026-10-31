from __future__ import annotations

from typing import Any

import streamlit as st


def render_metric(label: str, value: Any, help_text: str | None = None) -> None:
    """Render a consistent metric card across the dashboard."""
    st.metric(label=label, value=str(value), help=help_text)
