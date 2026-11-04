from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard.pages.application_explorer import render_application_explorer
from dashboard.pages.model_comparison import render_model_comparison
from dashboard.pages.overview import render_overview
from dashboard import data_access
from dashboard.components.theme import inject_styles


def main() -> None:
    """Initialize the dashboard application and handle page navigation."""
    st.set_page_config(
        page_title="Credit Risk Intelligence Platform",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.sidebar.markdown("## CREDIT RISK INTELLIGENCE")
    st.sidebar.markdown("### Live FastAPI data")
    try:
        health = data_access.get_api_health()
    except data_access.DataAccessError as error:
        st.sidebar.error(f"API disconnected: {error}")
    else:
        if health.get("status") == "ok":
            st.sidebar.success(
                f"API connected · {health.get('model_name') or 'model unavailable'} "
                f"{health.get('model_version') or ''}"
            )
        else:
            st.sidebar.warning(
                "API connected but degraded: model or application data is unavailable."
            )
    if st.sidebar.button("Refresh live data"):
        st.rerun()

    st.markdown(
        """
        <div style="border-bottom: 1px solid #E1E7EE; padding: 0 0 1.1rem; margin-bottom: 1.35rem;">
            <div style="color:#4B78A8; font-size:0.72rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase;">
                Credit Risk Intelligence Platform
            </div>
            <div style="color:#718096; font-size:0.91rem; margin-top:0.3rem;">
                Credit portfolio insights · Application analysis · Model evaluation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("Credit Risk Intelligence Platform")
    st.caption(
        "Applicant records, predictions, explanations, and evaluation results from the local FastAPI service."
    )

    try:
        pages = [
            st.Page(render_overview, title="Overview", icon="📊"),
            st.Page(render_application_explorer, title="Application Explorer", icon="🔎"),
            st.Page(render_model_comparison, title="Model Comparison", icon="📈"),
        ]
        navigation = st.navigation(pages)
        navigation.run()
        return
    except TypeError:
        page_map = {
            "Overview": render_overview,
            "Application Explorer": render_application_explorer,
            "Model Comparison": render_model_comparison,
        }
        selection = st.sidebar.radio("Navigate", list(page_map.keys()))
        page_map[selection]()


if __name__ == "__main__":
    main()
