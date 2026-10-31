from __future__ import annotations

import streamlit as st

from dashboard.components.theme import RED, TEAL, AMBER
from dashboard.data_access import get_application_by_id, get_applications


def render_application_explorer() -> None:
    """Allow inspection of individual synthetic applications."""
    st.title("Application Explorer")
    st.caption("Browse synthetic applicant records and inspect their illustrative risk profile.")

    applications = get_applications()
    if applications.empty:
        st.warning("No application data is available in mock mode.")
        return

    with st.container(border=True):
        st.subheader("Find an application")
        st.caption("Select a synthetic application by its identifier.")
        application_ids = sorted(applications["SK_ID_CURR"].tolist())
        selected_id = st.selectbox(
            "Select an application",
            application_ids,
            format_func=lambda value: f"SK_ID_CURR {value}",
        )

    selected_application = get_application_by_id(selected_id)
    if selected_application is None:
        st.warning("The selected application could not be found.")
        return

    with st.container(border=True):
        st.subheader("Applications")
        st.dataframe(
            applications[[
                "SK_ID_CURR",
                "AMT_INCOME_TOTAL",
                "AMT_CREDIT",
                "AMT_ANNUITY",
                "TARGET",
                "RISK_PROBABILITY",
                "RISK_CATEGORY",
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "SK_ID_CURR": st.column_config.NumberColumn("Application ID", format="%d"),
                "AMT_INCOME_TOTAL": st.column_config.NumberColumn("Income", format="$%,.0f"),
                "AMT_CREDIT": st.column_config.NumberColumn("Credit amount", format="$%,.0f"),
                "AMT_ANNUITY": st.column_config.NumberColumn("Annuity", format="$%,.0f"),
                "TARGET": st.column_config.NumberColumn("Target", format="%d"),
                "RISK_PROBABILITY": st.column_config.ProgressColumn(
                    "Illustrative risk probability",
                    min_value=0,
                    max_value=1,
                    format="%.1%%",
                ),
                "RISK_CATEGORY": st.column_config.TextColumn("Illustrative risk category"),
            },
        )

    st.subheader(f"Application {selected_id}")
    risk_category = selected_application["RISK_CATEGORY"]
    risk_color = {
        "Low Risk": TEAL,
        "Medium Risk": AMBER,
        "High Risk": RED,
    }.get(risk_category, "#718096")
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:inline-flex; align-items:center; gap:0.45rem; border:1px solid {risk_color}55;
                        color:{risk_color}; background:{risk_color}12; padding:0.35rem 0.65rem; border-radius:5px;
                        font-size:0.82rem; font-weight:700; margin-bottom:0.8rem;">
                {risk_category}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Synthetic profile details · Illustrative risk only")
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.metric("Income", f"${selected_application['AMT_INCOME_TOTAL']:,.0f}")
            st.metric("Credit amount", f"${selected_application['AMT_CREDIT']:,.0f}")
        with col2:
            st.metric("Annuity", f"${selected_application['AMT_ANNUITY']:,.0f}")
            st.metric("Target", selected_application["TARGET"])
        with col3:
            st.metric("Risk probability", f"{selected_application['RISK_PROBABILITY']:.2%}")
            st.progress(float(selected_application["RISK_PROBABILITY"]))
            st.metric("Risk category", selected_application["RISK_CATEGORY"])

    st.info(
        "This explorer shows mock applicant information only. The dashboard does not make lending decisions "
        "and does not connect to real data sources."
    )
