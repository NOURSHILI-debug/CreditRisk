from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from dashboard import data_access

_DETAIL_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Application information",
        (
            ("SK_ID_CURR", "Application ID"),
            ("NAME_CONTRACT_TYPE", "Contract type"),
            ("NAME_INCOME_TYPE", "Income type"),
            ("NAME_EDUCATION_TYPE", "Education type"),
            ("NAME_FAMILY_STATUS", "Family status"),
            ("NAME_HOUSING_TYPE", "Housing type"),
        ),
    ),
    (
        "Financial information",
        (
            ("AMT_INCOME_TOTAL", "Income amount"),
            ("AMT_CREDIT", "Credit amount"),
            ("AMT_ANNUITY", "Annuity amount"),
            ("AMT_GOODS_PRICE", "Goods price"),
        ),
    ),
    (
        "Demographic information",
        (
            ("DAYS_BIRTH", "Days birth"),
            ("DAYS_EMPLOYED", "Days employed"),
            ("CNT_CHILDREN", "Number of children"),
            ("CNT_FAM_MEMBERS", "Family members"),
        ),
    ),
    ("Target information", (("TARGET", "TARGET"),)),
)
_CURRENCY_FIELDS = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"}
_NON_APPLICATION_OUTPUT_FIELDS = {"RISK_PROBABILITY", "RISK_CATEGORY"}
_TABLE_FIELDS = (
    "SK_ID_CURR",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "TARGET",
)
_TABLE_LABELS = {
    "NAME_CONTRACT_TYPE": "Contract type",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_EDUCATION_TYPE": "Education type",
    "AMT_INCOME_TOTAL": "Income",
    "AMT_CREDIT": "Credit amount",
    "AMT_ANNUITY": "Annuity",
}


def _format_detail_value(field: str, value: Any) -> str:
    """Format a source value for display without substituting missing data."""
    if pd.isna(value):
        return "N/A"

    if field == "TARGET":
        try:
            target = float(value)
        except (TypeError, ValueError, OverflowError):
            return str(value)
        if not math.isfinite(target):
            return "N/A"
        if target == 1:
            return "1 — Repayment difficulty"
        if target == 0:
            return "0 — No defined repayment difficulty"
        return str(value)

    if field in _CURRENCY_FIELDS:
        try:
            amount = float(value)
        except (TypeError, ValueError, OverflowError):
            return str(value)
        if math.isfinite(amount):
            return f"${amount:,.0f}"
        return "N/A"

    return str(value)


def _render_application_details(application: pd.Series) -> None:
    """Render only source fields that are present on the selected record."""
    rendered_fields: set[str] = set()

    for group_title, configured_fields in _DETAIL_GROUPS:
        fields = [
            (field, label)
            for field, label in configured_fields
            if field in application.index
        ]
        if not fields:
            continue

        with st.container(border=True):
            st.subheader(group_title)
            columns = st.columns(min(3, len(fields)), gap="large")
            for index, (field, label) in enumerate(fields):
                with columns[index % len(columns)]:
                    st.metric(label, _format_detail_value(field, application[field]))
                rendered_fields.add(field)

            if "TARGET" in {field for field, _ in fields}:
                st.caption(
                    "TARGET = 1 indicates repayment difficulty; TARGET = 0 indicates "
                    "no defined repayment difficulty."
                )

    additional_fields = [
        field
        for field in application.index
        if field not in rendered_fields and field not in _NON_APPLICATION_OUTPUT_FIELDS
    ]
    if additional_fields:
        with st.container(border=True):
            st.subheader("Additional available information")
            columns = st.columns(min(3, len(additional_fields)), gap="large")
            for index, field in enumerate(additional_fields):
                with columns[index % len(columns)]:
                    label = field.replace("_", " ").title()
                    st.metric(label, _format_detail_value(field, application[field]))


def _render_application_table(applications: pd.DataFrame) -> None:
    table_fields = [
        field for field in _TABLE_FIELDS if field in applications.columns
    ]
    if not table_fields:
        return

    column_config: dict[str, Any] = {}
    for field in table_fields:
        numeric_field = pd.api.types.is_numeric_dtype(applications[field])
        if field == "SK_ID_CURR" and numeric_field:
            column_config[field] = st.column_config.NumberColumn(
                "Application ID",
                format="%d",
            )
        elif field in _CURRENCY_FIELDS and numeric_field:
            column_config[field] = st.column_config.NumberColumn(
                _TABLE_LABELS.get(field, field.replace("_", " ").title()),
                format="$%,.0f",
            )
        elif field == "TARGET" and numeric_field:
            column_config[field] = st.column_config.NumberColumn(
                "TARGET (0/1)",
                format="%d",
            )
        else:
            column_config[field] = st.column_config.TextColumn(
                _TABLE_LABELS.get(field, field.replace("_", " ").title())
            )

    with st.container(border=True):
        st.subheader("Applications")
        st.caption("Available application records from the configured data source.")
        st.dataframe(
            applications[table_fields],
            width="stretch",
            hide_index=True,
            column_config=column_config,
        )


def _render_prediction(application_id: int) -> None:
    """Render prediction output and explanation only when supplied by a provider."""
    prediction = data_access.get_credit_risk_prediction(application_id)
    is_valid_prediction = (
        prediction.status == "available"
        and prediction.probability is not None
        and math.isfinite(prediction.probability)
        and 0 <= prediction.probability <= 1
    )

    with st.container(border=True):
        st.subheader("Credit-Risk Prediction")
        if prediction.status == "available" and not is_valid_prediction:
            st.warning("Prediction unavailable: the returned probability is invalid.")
        elif not is_valid_prediction:
            status_messages = {
                "no_prediction": "No prediction was returned for this application.",
                "unavailable": "Prediction functionality is not yet available.",
                "model_unavailable": "The prediction model is unavailable.",
                "invalid": "The prediction response is invalid.",
            }
            message = prediction.message or status_messages.get(
                prediction.status,
                "Prediction functionality is not yet available.",
            )
            st.info(message)

        columns = st.columns(4, gap="large")
        values = (
            (
                "Credit-risk probability",
                f"{prediction.probability:.1%}" if is_valid_prediction else "N/A",
            ),
            ("Model name", prediction.model_name or "N/A" if is_valid_prediction else "N/A"),
            ("Model version", prediction.model_version or "N/A" if is_valid_prediction else "N/A"),
            (
                "Prediction timestamp",
                prediction.prediction_timestamp or "N/A" if is_valid_prediction else "N/A",
            ),
        )
        for column, (label, value) in zip(columns, values):
            with column:
                st.metric(label, value)

    with st.container(border=True):
        st.subheader("Prediction Explanation")
        explanation = prediction.explanation if is_valid_prediction else None
        if explanation is None or explanation == "":
            st.info("Prediction explanations are not available yet.")
        else:
            st.write(explanation)


def render_application_explorer() -> None:
    """Search for an application and inspect fields available in its source record."""
    st.title("Application Explorer")
    st.caption("Search by application ID to inspect the available source-record details.")

    if data_access.is_mock_mode():
        st.warning(
            "DEMO DATA — Synthetic applications. Figures are illustrative and do not "
            "represent actual Home Credit results."
        )
    else:
        st.info("Application records are provided by the configured data source.")

    try:
        applications = data_access.get_applications()
    except data_access.DataAccessError as error:
        st.error(f"Application data is unavailable: {error}")
        return
    with st.container(border=True):
        st.subheader("Find an application")
        st.caption("Enter the application ID as shown in the source data.")
        with st.form("application_explorer_search"):
            entered_id = st.text_input(
                "Application ID",
                placeholder="For example, 1001",
                key="application_explorer_id_input",
            )
            submitted = st.form_submit_button("Search application", type="primary")

    if submitted:
        st.session_state["application_explorer_last_query"] = entered_id.strip()

    if applications.empty:
        st.info("No application records are available from the configured data source.")
        return

    last_query = st.session_state.get("application_explorer_last_query")
    if last_query is not None:
        result = data_access.get_application_by_id(last_query, applications)
        if result.status == "invalid_id":
            if not last_query:
                st.warning("Enter an application ID to search.")
            else:
                st.warning("Enter a valid positive integer application ID.")
        elif result.status == "not_found":
            st.info(f"No application was found for ID {last_query}.")
        elif result.status == "duplicate":
            st.warning(
                f"Application ID {last_query} is duplicated in the source data. "
                "The record cannot be selected unambiguously."
            )
        elif result.status == "unavailable":
            st.warning("Application lookup is unavailable because the source has no usable ID field.")
        elif result.application is not None:
            st.subheader(f"Application {last_query}")
            _render_application_details(result.application)
            _render_prediction(int(result.application["SK_ID_CURR"]))

    if "SK_ID_CURR" not in applications.columns:
        st.warning("Application search requires the SK_ID_CURR identifier field.")
    _render_application_table(applications)

    if data_access.is_mock_mode():
        st.caption(
            "This explorer displays synthetic application information only. "
            "It does not generate risk predictions or make lending decisions."
        )
