from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from agent import tools
from dashboard.app import _render_data_source_selector
from dashboard import data_access
from dashboard.pages.application_explorer import render_application_explorer
from dashboard.pages.model_comparison import render_model_comparison
from dashboard.pages.overview import render_overview
from dashboard.providers import mock_provider


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDIT_RISK_DATA_MODE", "mock")
    monkeypatch.setenv("CREDIT_RISK_MOCK_SAMPLE_SIZE", "32")
    monkeypatch.setenv("CREDIT_RISK_MOCK_SEED", "42")


def test_mode_defaults_to_api_and_rejects_invalid_values(monkeypatch) -> None:
    monkeypatch.delenv("CREDIT_RISK_DATA_MODE", raising=False)
    assert data_access.configured_data_mode() == "api"
    monkeypatch.setenv("CREDIT_RISK_DATA_MODE", "automatic")
    with pytest.raises(data_access.DataModeConfigurationError):
        data_access.configured_data_mode()
    with pytest.raises(data_access.DataModeConfigurationError):
        data_access.set_data_mode("automatic")


def test_provider_selection_switches_explicitly_and_missing_api_never_uses_mock(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CREDIT_RISK_DATA_MODE", raising=False)
    monkeypatch.delenv("CREDIT_RISK_API_URL", raising=False)
    previous_mode = data_access._DATA_MODE_OVERRIDE.get()
    try:
        assert data_access.configured_data_mode() == "api"
        data_access.set_data_mode("mock")
        assert data_access.get_provider_name() == "mock"
        assert data_access.get_applications_page(limit=2)[1] == 32

        data_access.set_data_mode("api")
        assert data_access.get_provider_name() == "fastapi"
        monkeypatch.setattr(
            data_access,
            "_client",
            lambda: (_ for _ in ()).throw(data_access.APIUnavailableError("offline")),
        )
        with pytest.raises(data_access.DataSourceUnavailableError):
            data_access.get_api_health()
    finally:
        data_access._DATA_MODE_OVERRIDE.set(previous_mode)


def test_sidebar_mode_selector_persists_and_switches_provider(monkeypatch) -> None:
    monkeypatch.setenv("CREDIT_RISK_DATA_MODE", "api")
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from dashboard.app import _render_data_source_selector\n"
        "from dashboard import data_access\n"
        "_render_data_source_selector()\n"
        "st.write(data_access.get_data_mode())\n"
    ).run()
    assert not app.exception
    assert app.session_state["credit_risk_data_mode"] == "api"
    app.selectbox(key="credit_risk_data_mode").select("mock").run()
    assert not app.exception
    assert app.session_state["credit_risk_data_mode"] == "mock"
    assert "mock" in [item.value for item in app.markdown]


def test_dashboard_entrypoint_launches_in_explicit_mock_mode() -> None:
    entrypoint = Path(__file__).parents[2] / "src" / "dashboard" / "app.py"
    app = AppTest.from_file(entrypoint, default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Credit Risk Overview"
    assert app.session_state["credit_risk_data_mode"] == "mock"
    assert any("Synthetic mock mode" in item.value for item in app.warning)


def test_mock_sample_is_reproducible_and_has_home_credit_fields() -> None:
    first = mock_provider.get_dataset()
    second = mock_provider.get_dataset()
    pd.testing.assert_frame_equal(first.applications, second.applications)
    assert len(first.applications) == 32
    assert first.applications["SK_ID_CURR"].is_unique
    assert set(first.applications["TARGET"].unique()).issubset({0, 1})
    assert {
        "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
        "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
        "REGION_POPULATION_RELATIVE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    }.issubset(first.applications.columns)
    assert first.applications["DAYS_BIRTH"].lt(0).all()


def test_mock_historical_tables_have_valid_foreign_keys_and_ordered_installments() -> None:
    data = mock_provider.get_dataset()
    application_ids = set(data.applications["SK_ID_CURR"])
    previous_ids = set(data.previous_application["SK_ID_PREV"])
    bureau_ids = set(data.bureau["SK_ID_BUREAU"])
    assert set(data.installments_payments["SK_ID_CURR"]).issubset(application_ids)
    assert set(data.previous_application["SK_ID_CURR"]).issubset(application_ids)
    assert set(data.bureau["SK_ID_CURR"]).issubset(application_ids)
    assert set(data.installments_payments["SK_ID_PREV"]).issubset(previous_ids)
    assert set(data.pos_cash_balance["SK_ID_PREV"]).issubset(previous_ids)
    assert set(data.credit_card_balance["SK_ID_PREV"]).issubset(previous_ids)
    assert set(data.bureau_balance["SK_ID_BUREAU"]).issubset(bureau_ids)
    for _, rows in data.installments_payments.groupby(["SK_ID_CURR", "SK_ID_PREV"]):
        assert rows["DAYS_INSTALMENT"].is_monotonic_increasing
        assert rows["NUM_INSTALMENT_NUMBER"].is_monotonic_increasing


def test_mock_mode_dispatches_every_dashboard_query_without_api_calls(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("mock mode called the API client")

    monkeypatch.setattr(data_access, "_client", forbidden)
    health = data_access.get_api_health()
    page, total = data_access.get_applications_page(limit=7, offset=0)
    application = data_access.get_application_by_id(100002)
    prediction = data_access.get_credit_risk_prediction(100002)
    summary = data_access.get_prediction_summary()
    metrics = data_access.get_overview_metrics()
    model_info = data_access.get_model_info()
    evaluation = data_access.get_model_evaluation_results()

    assert data_access.get_provider_name() == "mock"
    assert data_access.is_mock_mode()
    assert health["source_mode"] == "mock"
    assert total == 32 and len(page) == 7
    assert application.status == "found"
    assert application.application is not None
    assert "TARGET" not in application.application.index
    assert prediction.status == "available"
    assert 0 <= prediction.probability <= 1
    assert "Synthetic" in prediction.message
    assert summary["source_mode"] == "mock"
    assert metrics["total_applications"] == 32
    assert model_info["source_mode"] == "mock"
    assert evaluation.attrs["provenance"]["synthetic"] is True


def test_mock_predictions_use_contract_statuses_and_exact_risk_boundaries() -> None:
    assert mock_provider._risk_level(0.2999) == "LOW"
    assert mock_provider._risk_level(0.30) == "MEDIUM"
    assert mock_provider._risk_level(0.5999) == "MEDIUM"
    assert mock_provider._risk_level(0.60) == "HIGH"
    missing = data_access.get_credit_risk_prediction(100001)
    unknown = data_access.get_credit_risk_prediction(999999)
    assert missing.status == "no_prediction"
    assert missing.probability is None
    assert unknown.status == "not_found"
    assert unknown.probability is None


def test_mock_predictions_and_evaluations_are_explicitly_illustrative() -> None:
    prediction = data_access.get_credit_risk_prediction(100002, include_explanation=True)
    assert prediction.status == "available"
    assert prediction.model_name == "Synthetic demo predictor"
    assert prediction.explanation["synthetic"] is True
    assert "not produced by a trained" in prediction.message
    assert prediction.prediction_timestamp is None
    evaluation = data_access.get_model_evaluation_results()
    assert evaluation.empty
    assert list(evaluation.columns) == ["model", "metric", "value"]
    assert evaluation.attrs["provenance"]["synthetic"] is True
    assert evaluation.attrs["provenance"]["split_protocol"] is None
    assert evaluation.attrs["provenance"]["generated_at"] is None


def test_mock_mode_never_exposes_target_and_api_mode_never_falls_back(monkeypatch) -> None:
    with pytest.raises(data_access.MalformedProviderResponseError):
        data_access.get_applications_page(fields=["TARGET"])

    monkeypatch.setenv("CREDIT_RISK_DATA_MODE", "api")
    monkeypatch.setattr(
        data_access,
        "_client",
        lambda: (_ for _ in ()).throw(data_access.APIUnavailableError("offline")),
    )
    with pytest.raises(data_access.DataSourceUnavailableError):
        data_access.get_applications_page()


@pytest.mark.parametrize(
    ("page", "expected_label"),
    [
        (render_overview, "Synthetic mock mode"),
        (render_application_explorer, "Synthetic mock applications"),
        (render_model_comparison, "No trained-model evaluation metrics"),
    ],
)
def test_dashboard_pages_label_mock_data_and_do_not_call_api(
    monkeypatch, page, expected_label: str
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("mock dashboard page called the API client")

    monkeypatch.setattr(data_access, "_client", forbidden)
    page_module = page.__module__
    page_name = page.__name__
    app = AppTest.from_string(
        f"from {page_module} import {page_name}\n{page_name}()\n"
    ).run()
    assert not app.exception
    visible_text = " ".join(
        element.value
        for collection in (app.warning, app.info, app.caption, app.markdown)
        for element in collection
        if isinstance(element.value, str)
    )
    assert expected_label in visible_text


def test_all_agent_tools_use_mock_provider_and_preserve_synthetic_labels(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("mock agent tool called the API client")

    monkeypatch.setattr(data_access, "_client", forbidden)
    details = tools.get_application_details(100002)
    metrics = tools.get_dashboard_metrics()
    prediction = tools.get_prediction(100002)
    explanation = tools.explain_prediction(100002)
    comparison = tools.get_model_comparison()
    report = tools.generate_report("30d")

    for result in (details, metrics, prediction, explanation, comparison):
        assert result["source_mode"] == "mock"
    assert "TARGET" not in details["fields"]
    assert prediction["status"] == "available"
    assert "Synthetic" in prediction["message"]
    assert explanation["explanation_method"] == "synthetic_illustrative"
    assert comparison["comparison_available"] is False
    assert "Synthetic demo data" in comparison["comparison_note"]
    assert comparison["statuses"] == {
        "XGBoost": "Not available",
        "LSTM": "Not available",
    }
    assert all(
        score is None
        for model_metrics in comparison["models"].values()
        for score in model_metrics.values()
    )
    assert report["source_mode"] == "mock"
    assert "Synthetic mock report" in report["markdown"]

