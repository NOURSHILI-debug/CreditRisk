from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from dashboard import data_access
from dashboard.components.ui import (
    render_metric_grid,
    render_page_header,
    render_section_heading,
    render_status,
)

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
            ("DAYS_BIRTH", "Age at application"),
            ("DAYS_EMPLOYED", "Employment day offset"),
            ("CNT_CHILDREN", "Number of children"),
            ("CNT_FAM_MEMBERS", "Family members"),
        ),
    ),
)
_CURRENCY_FIELDS = {
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
}
_FILTERS = {
    "Income type": "NAME_INCOME_TYPE",
    "Contract type": "NAME_CONTRACT_TYPE",
    "Education type": "NAME_EDUCATION_TYPE",
}
_FIELD_LABELS = {
    "SK_ID_CURR": "Application ID",
    "NAME_CONTRACT_TYPE": "Contract type",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_EDUCATION_TYPE": "Education type",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_HOUSING_TYPE": "Housing type",
    "AMT_INCOME_TOTAL": "Income (source units)",
    "AMT_CREDIT": "Credit amount (source units)",
    "AMT_ANNUITY": "Annuity (source units)",
    "AMT_GOODS_PRICE": "Goods price (source units)",
    "DAYS_BIRTH": "Age at application (years)",
    "DAYS_EMPLOYED": "Employment day offset (source days)",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Family members",
}
_COMPACT_COLUMNS = (
    "SK_ID_CURR",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
)


def _format_detail_value(field: str, value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    if field in _CURRENCY_FIELDS:
        try:
            amount = float(value)
        except (TypeError, ValueError, OverflowError):
            return str(value)
        return f"{amount:,.0f}" if math.isfinite(amount) else "N/A"
    if field == "DAYS_BIRTH":
        try:
            days = float(value)
        except (TypeError, ValueError, OverflowError):
            return str(value)
        return f"{abs(days) / 365.25:.1f} years" if math.isfinite(days) else "N/A"
    if field == "DAYS_EMPLOYED":
        try:
            days = float(value)
        except (TypeError, ValueError, OverflowError):
            return str(value)
        return f"{days:,.0f} source days" if math.isfinite(days) else "N/A"
    return str(value)


def _render_application_details(application: pd.Series) -> None:
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
                    st.metric(
                        label,
                        _format_detail_value(field, application[field]),
                        help=(
                            "Age is calculated from the source day offset for display only."
                            if field == "DAYS_BIRTH"
                            else "Original source value; the API does not provide a currency code."
                            if field in _CURRENCY_FIELDS
                            else None
                        ),
                    )


def _render_application_table(applications: pd.DataFrame) -> None:
    if applications.empty:
        st.info("No applications match the current search or filters.")
        return
    display = applications.drop(columns=["TARGET"], errors="ignore")
    show_all_fields = st.checkbox(
        "Show all available application fields",
        value=False,
        help="The API response remains unchanged; this only changes the visible table columns.",
    )
    column_order = list(display.columns)
    if not show_all_fields:
        compact = [field for field in _COMPACT_COLUMNS if field in display.columns]
        column_order = compact + [field for field in display.columns if field not in compact]
        column_order = compact
    column_config = {
        field: st.column_config.Column(_FIELD_LABELS.get(field, field.replace("_", " ").title()))
        for field in display.columns
    }
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_order=column_order,
        column_config=column_config,
    )


def render_application_explorer() -> None:
    render_page_header(
        "Application Explorer",
        "Find an application, review permitted source fields, then inspect its model output and explanation.",
    )
    saved_filter_field = st.session_state.get("application_search_filter_field")
    saved_filter_label = next(
        (
            label
            for label, field in _FILTERS.items()
            if field == saved_filter_field
        ),
        "No category filter",
    )
    render_section_heading("Find an application", "Search by ID or narrow results by an exact category value.")
    with st.form("application_explorer_filters", border=True):
        id_text = st.text_input(
            "Search by application ID",
            value=st.session_state.get("application_search_id", ""),
            placeholder="Enter an application ID",
        )
        filter_label = st.selectbox(
            "Filter by",
            ["No category filter", *_FILTERS],
            index=["No category filter", *_FILTERS].index(saved_filter_label),
        )
        filter_value = st.text_input(
            "Exact category value",
            value=st.session_state.get("application_search_filter_value", ""),
            disabled=filter_label == "No category filter",
        )
        submitted = st.form_submit_button("Apply search and filters", type="primary")
    if submitted:
        st.session_state["application_search_id"] = id_text.strip()
        st.session_state["application_search_filter_field"] = _FILTERS.get(filter_label)
        st.session_state["application_search_filter_value"] = filter_value.strip()
        st.session_state["application_page_offset"] = 0
        st.session_state.pop("application_prediction", None)
        st.session_state.pop("selected_application", None)
        st.session_state.pop("selected_application_details", None)
        st.session_state.pop("selected_application_id", None)

    raw_id = st.session_state.get("application_search_id", "")
    application_id = None
    if raw_id:
        parsed_id = data_access.parse_application_id(raw_id)
        if parsed_id is None:
            render_status("Enter a positive integer application ID.", kind="warning")
            return
        application_id = parsed_id

    page_size = st.selectbox(
        "Applications per page", [25, 50, 100], index=0, key="application_page_size"
    )
    offset = st.session_state.get("application_page_offset", 0)
    filter_field = st.session_state.get("application_search_filter_field")
    filter_value = st.session_state.get("application_search_filter_value")
    if filter_field and not filter_value:
        filter_field = None

    try:
        with st.spinner("Retrieving applicant records from the API..."):
            applications, total = data_access.get_applications_page(
                limit=page_size,
                offset=offset,
                application_id=application_id,
                filter_field=filter_field,
                filter_value=filter_value,
            )
    except data_access.DataAccessError:
        render_status(
            "Applicant data is unavailable.",
            "Check that the local API and authorized application data are available. No sample records are shown.",
            kind="warning",
        )
        return

    _render_application_table(applications)
    start = offset + 1 if total else 0
    end = min(offset + len(applications), total)
    st.caption(f"Showing {start:,}–{end:,} of {total:,} matching records.")
    previous, next_page = st.columns(2)
    with previous:
        if st.button("Previous page", disabled=offset == 0):
            st.session_state["application_page_offset"] = max(0, offset - page_size)
            st.session_state.pop("selected_application", None)
            st.session_state.pop("selected_application_details", None)
            st.session_state.pop("application_prediction", None)
            st.session_state.pop("selected_application_id", None)
            st.rerun()
    with next_page:
        if st.button("Next page", disabled=offset + page_size >= total):
            st.session_state["application_page_offset"] = offset + page_size
            st.session_state.pop("selected_application", None)
            st.session_state.pop("selected_application_details", None)
            st.session_state.pop("application_prediction", None)
            st.session_state.pop("selected_application_id", None)
            st.rerun()

    if applications.empty:
        return
    application_ids = applications["SK_ID_CURR"].astype(int).tolist()
    previous_selection = st.session_state.get("selected_application")
    selected_index = (
        application_ids.index(previous_selection)
        if previous_selection in application_ids
        else 0
    )
    selected_id = st.selectbox(
        "Select an application for details and prediction",
        application_ids,
        index=selected_index,
        key="selected_application_id",
    )
    if st.button("Load applicant details"):
        try:
            result = data_access.get_application_by_id(selected_id)
        except data_access.DataAccessError:
            render_status(
                "Applicant details could not be retrieved.",
                "Check API availability and try again.",
                kind="error",
            )
        else:
            if result.status == "found" and result.application is not None:
                st.session_state["selected_application"] = selected_id
                st.session_state["selected_application_details"] = result.application
                st.session_state.pop("application_prediction", None)
            else:
                detail_messages = {
                    "not_found": "No application was found for this ID.",
                    "duplicate": "More than one application matched this ID.",
                    "unavailable": "Applicant details are currently unavailable.",
                    "invalid_id": "The application ID is invalid.",
                }
                render_status(
                    detail_messages.get(result.status, "Applicant details could not be retrieved."),
                    kind="warning",
                )

    if st.session_state.get("selected_application") == selected_id:
        application = st.session_state.get("selected_application_details")
        if isinstance(application, pd.Series):
            render_section_heading(f"Application {selected_id}", "Permitted application details from the API.")
            st.caption("Monetary amounts use source units; the API does not provide a currency code.")
            _render_application_details(application)
        if st.button("Run XGBoost prediction and SHAP explanation"):
            try:
                with st.spinner("Scoring application and computing explanation..."):
                    prediction = data_access.get_credit_risk_prediction(
                        selected_id, include_explanation=True
                    )
                st.session_state["application_prediction"] = prediction
            except data_access.DataAccessError as error:
                render_status(
                    "Prediction could not be completed.",
                    "Check model availability and try again.",
                    kind="error",
                )
        prediction = st.session_state.get("application_prediction")
        if prediction is not None:
            _render_prediction_result(prediction)


def _render_prediction_result(prediction: data_access.CreditRiskPrediction) -> None:
    with st.container(border=True):
        render_section_heading("Model prediction")
        if prediction.status != "available" or prediction.probability is None:
            if prediction.status in {"unavailable", "model_unavailable", "no_prediction"}:
                render_status(
                    "No prediction is currently available.",
                    "The application record may exist while its model output or required artifact is unavailable.",
                    kind="warning",
                )
            elif prediction.status == "not_found":
                render_status("No application was found for this prediction request.", kind="warning")
            else:
                render_status(
                    "The prediction request could not be completed.",
                    "No score is displayed because the API did not return a valid prediction.",
                    kind="error",
                )
            return
        render_metric_grid(
            [
                (
                    "Probability of repayment difficulty",
                    f"{prediction.probability:.1%}",
                    "Model probability returned by the API; not a lending decision.",
                ),
                (
                    "Provisional model risk band",
                    prediction.risk_level or "Unavailable",
                    "A model category, not an approval or rejection decision.",
                ),
            ],
            columns=2,
        )
        try:
            thresholds = data_access.get_model_info().get("thresholds", {})
            bands = thresholds.get("risk_bands", {})
            lower = bands.get("low_upper")
            upper = bands.get("medium_upper")
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                st.caption(
                    f"Provisional model bands: LOW < {lower:.0%}, MEDIUM < "
                    f"{upper:.0%}, HIGH otherwise. These are not lending decisions."
                )
            else:
                st.warning("The API did not provide model risk-band thresholds.")
        except data_access.DataAccessError:
            render_status(
                "Risk-band definitions are unavailable.",
                "The prediction is shown with the model metadata returned by the prediction endpoint.",
                kind="warning",
            )
        st.caption(
            f"Model: {prediction.model_name or 'N/A'} "
            f"({prediction.model_version or 'version unavailable'}); "
            f"scored at {prediction.prediction_timestamp or 'timestamp unavailable'}."
        )

        explanation = prediction.explanation
        if not isinstance(explanation, dict):
            st.info("The API did not return a SHAP explanation for this prediction.")
            return
        render_section_heading("Model explanation", "Feature contributions describe model associations, not causes.")
        st.write(
            "These are the features that most influenced the model's estimated "
            "risk for this application. They describe model associations, not causes."
        )
        features = explanation.get("top_features")
        if isinstance(features, list) and features:
            rows = []
            for feature in features[:8]:
                if not isinstance(feature, dict):
                    continue
                direction = {
                    "increases_risk": "Raises estimated risk",
                    "decreases_risk": "Lowers estimated risk",
                    "neutral": "No material direction",
                }.get(feature.get("effect"), "Direction unavailable")
                rows.append(
                    {
                        "Applicant attribute": str(
                            feature.get("raw_feature") or feature.get("feature")
                        ).replace("_", " ").title(),
                        "Observed value": feature.get("raw_value"),
                        "Model contribution": direction,
                    }
                )
            if rows:
                st.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )
        summary = explanation.get("summary")
        if isinstance(summary, str) and summary:
            st.caption(summary)
