from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard.pages.application_explorer import render_application_explorer
from dashboard.pages.assistant import render_assistant
from dashboard.pages.model_comparison import render_model_comparison
from dashboard.pages.overview import render_overview
from dashboard import data_access
from dashboard.components.theme import inject_styles


def _render_data_source_selector() -> str:
    try:
        default_mode = data_access.configured_data_mode()
    except data_access.DataModeConfigurationError as error:
        st.sidebar.error(str(error))
        st.stop()
    if st.session_state.get("credit_risk_data_mode") not in {"api", "mock"}:
        st.session_state["credit_risk_data_mode"] = default_mode
    mode = st.sidebar.selectbox(
        "Data source",
        options=["api", "mock"],
        format_func=lambda value: "Real API" if value == "api" else "Synthetic mock data",
        key="credit_risk_data_mode",
        help="Choose the real FastAPI backend or deterministic synthetic demo data.",
    )
    data_access.set_data_mode(mode)
    if mode == "mock":
        st.sidebar.warning(
            "Mock mode active. All displayed records and metrics are synthetic examples; no API requests are made."
        )
    return mode


def main() -> None:
    """Initialize the dashboard application and handle page navigation."""
    st.set_page_config(
        page_title="Credit Risk Intelligence Platform",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.sidebar.markdown("## CREDIT RISK")
    st.sidebar.caption("INTELLIGENCE PLATFORM")
    mode = _render_data_source_selector()
    if mode == "api":
        st.sidebar.markdown("### Local API status")
        try:
            health = data_access.get_api_health()
        except data_access.DataAccessError:
            st.sidebar.error("API disconnected. Start the local service and refresh.")
        else:
            if health.get("status") == "ok":
                st.sidebar.success(
                    f"API connected | {health.get('model_name') or 'model unavailable'} "
                    f"{health.get('model_version') or ''}"
                )
            else:
                st.sidebar.warning(
                    "API connected but degraded: model or application data is unavailable."
                )
    if st.sidebar.button("Refresh data", help="Reload this page and its live data"):
        st.rerun()

    try:
        pages = [
            st.Page(render_overview, title="Overview", icon=":material/dashboard:"),
            st.Page(render_application_explorer, title="Application Explorer", icon=":material/search:"),
            st.Page(render_model_comparison, title="Model Performance", icon=":material/monitoring:"),
            st.Page("pages/assistant.py", title="AI Assistant", icon=":material/chat:"),
        ]
        navigation = st.navigation(pages)
        navigation.run()
        return
    except TypeError:
        page_map = {
            "Overview": render_overview,
            "Application Explorer": render_application_explorer,
            "Model Performance": render_model_comparison,
            "AI Assistant": render_assistant,
        }
        selection = st.sidebar.radio("Navigate", list(page_map.keys()))
        page_map[selection]()


if __name__ == "__main__":
    main()
