from __future__ import annotations

import json

import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard.data_access import (
    get_application_by_id,
    get_credit_risk_prediction,
    normalize_prediction_response,
)
from dashboard.pages.application_explorer import _format_detail_value

_NO_PREDICTION_FIXTURE = object()


def test_prediction_response_normalizes_probability_and_metadata() -> None:
    result = normalize_prediction_response(
        {
            "probability": 0.1234,
            "model_name": "Provider model",
            "model_version": "v2",
            "prediction_timestamp": "2026-10-31T12:00:00Z",
            "explanation": {"factors": ["provider supplied"]},
        }
    )

    assert result.status == "available"
    assert result.probability == 0.1234
    assert result.model_name == "Provider model"
    assert result.model_version == "v2"
    assert result.prediction_timestamp == "2026-10-31T12:00:00Z"
    assert result.explanation == {"factors": ["provider supplied"]}


def test_prediction_response_handles_missing_unavailable_and_malformed_results() -> None:
    assert normalize_prediction_response(None).status == "no_prediction"
    assert (
        normalize_prediction_response({"status": "model_unavailable"}).status
        == "model_unavailable"
    )
    assert (
        normalize_prediction_response({"status": "unavailable"}).status
        == "unavailable"
    )
    assert normalize_prediction_response(["malformed"]).status == "invalid"
    assert normalize_prediction_response({"model_name": "missing score"}).status == "invalid"


def test_prediction_response_rejects_invalid_probability_values() -> None:
    invalid_values = (True, "not numeric", float("nan"), float("inf"), -0.01, 1.01)

    for probability in invalid_values:
        result = normalize_prediction_response({"probability": probability})
        assert result.status == "invalid"
        assert result.probability is None


def test_prediction_response_allows_missing_metadata_but_no_fake_explanation() -> None:
    result = normalize_prediction_response({"probability": 0.5})

    assert result.status == "available"
    assert result.model_name is None
    assert result.model_version is None
    assert result.prediction_timestamp is None
    assert result.explanation is None


def test_configured_prediction_boundary_reports_unavailable_without_provider() -> None:
    result = get_credit_risk_prediction(1001)

    assert result.status == "unavailable"
    assert result.probability is None
    assert result.model_name is None
    assert result.model_version is None
    assert result.prediction_timestamp is None


def test_lookup_finds_matching_application_after_trimming_whitespace() -> None:
    applications = pd.DataFrame(
        {"SK_ID_CURR": [1001, 1002], "AMT_CREDIT": [50000, 75000]}
    )

    result = get_application_by_id(" 1002 ", applications)

    assert result.status == "found"
    assert result.application is not None
    assert result.application["AMT_CREDIT"] == 75000


def test_lookup_distinguishes_invalid_and_well_formed_missing_ids() -> None:
    applications = pd.DataFrame({"SK_ID_CURR": [1001]})

    invalid_ids = ("", "1001.0", "-1001", 1001.0, True, "9" * 5000)
    assert all(
        get_application_by_id(invalid_id, applications).status == "invalid_id"
        for invalid_id in invalid_ids
    )
    assert get_application_by_id("9999", applications).status == "not_found"


def test_lookup_reports_duplicate_ids_without_selecting_a_record() -> None:
    applications = pd.DataFrame(
        {"SK_ID_CURR": [1001, 1001], "AMT_CREDIT": [50000, 75000]}
    )

    result = get_application_by_id("1001", applications)

    assert result.status == "duplicate"
    assert result.application is None


def test_lookup_reports_unavailable_for_empty_or_missing_id_source() -> None:
    assert get_application_by_id("1001", pd.DataFrame()).status == "unavailable"
    assert (
        get_application_by_id("1001", pd.DataFrame({"AMT_CREDIT": [50000]})).status
        == "unavailable"
    )


def test_lookup_uses_the_configured_data_access_source_when_no_frame_is_passed(
    monkeypatch,
) -> None:
    from dashboard import data_access

    configured_source = pd.DataFrame({"SK_ID_CURR": [1001]})
    monkeypatch.setattr(data_access, "get_applications", lambda: configured_source)

    result = get_application_by_id(1001)

    assert result.status == "found"
    assert result.application is not None
    assert result.application["SK_ID_CURR"] == 1001


def test_search_displays_source_details_and_target_meaning() -> None:
    app = _render_explorer(
        [
            {
                "SK_ID_CURR": 1001,
                "NAME_INCOME_TYPE": "Working",
                "AMT_CREDIT": 50000,
                "TARGET": 1,
                "RISK_PROBABILITY": 0.9,
                "RISK_CATEGORY": "High Risk",
            }
        ],
        search_id="1001",
    )

    assert not app.exception
    assert "Application 1001" in [element.value for element in app.subheader]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Application ID"] == "1001"
    assert metrics["Income type"] == "Working"
    assert metrics["Credit amount"] == "$50,000"
    assert metrics["TARGET"] == "1 — Repayment difficulty"
    assert "Risk probability" not in metrics
    assert "Risk category" not in metrics
    assert any("DEMO DATA — Synthetic applications" in element.value for element in app.warning)
    assert any(
        "TARGET = 1 indicates repayment difficulty" in element.value
        for element in app.caption
    )


def test_search_displays_real_provider_prediction_and_explanation_fixture() -> None:
    app = _render_explorer(
        [{"SK_ID_CURR": 1001, "TARGET": 0}],
        search_id="1001",
        prediction_response={
            "probability": 0.1234,
            "model_name": "Fixture model",
            "model_version": "fixture-v1",
            "prediction_timestamp": "2026-10-31T12:00:00Z",
            "explanation": "Fixture explanation supplied by the test provider.",
        },
    )

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Credit-risk probability"] == "12.3%"
    assert metrics["Model name"] == "Fixture model"
    assert metrics["Model version"] == "fixture-v1"
    assert metrics["Prediction timestamp"] == "2026-10-31T12:00:00Z"
    assert "TARGET" in metrics
    assert any(
        "Fixture explanation supplied by the test provider." in element.value
        for element in app.markdown
    )


def test_search_shows_unavailable_prediction_and_missing_metadata_as_na() -> None:
    unavailable_app = _render_explorer(
        [{"SK_ID_CURR": 1001}],
        search_id="1001",
    )
    partial_app = _render_explorer(
        [{"SK_ID_CURR": 1001}],
        search_id="1001",
        prediction_response={"probability": 0.75},
    )

    assert not unavailable_app.exception
    assert any(
        "Prediction functionality is not yet available" in element.value
        for element in unavailable_app.info
    )
    unavailable_metrics = {
        metric.label: metric.value for metric in unavailable_app.metric
    }
    assert unavailable_metrics["Credit-risk probability"] == "N/A"
    assert unavailable_metrics["Model name"] == "N/A"
    assert "Prediction explanations are not available yet." in [
        element.value for element in unavailable_app.info
    ]

    model_unavailable_app = _render_explorer(
        [{"SK_ID_CURR": 1001}],
        search_id="1001",
        prediction_response={
            "status": "model_unavailable",
            "message": "Model has not been loaded.",
        },
    )
    assert not model_unavailable_app.exception
    assert "Model has not been loaded." in [
        element.value for element in model_unavailable_app.info
    ]

    assert not partial_app.exception
    partial_metrics = {metric.label: metric.value for metric in partial_app.metric}
    assert partial_metrics["Credit-risk probability"] == "75.0%"
    assert partial_metrics["Model name"] == "N/A"
    assert partial_metrics["Model version"] == "N/A"
    assert partial_metrics["Prediction timestamp"] == "N/A"


def test_search_never_displays_invalid_or_missing_probability_as_prediction() -> None:
    for response in (
        {"probability": 1.5},
        {"probability": "invalid"},
        {"model_name": "missing probability"},
        ["malformed response"],
    ):
        app = _render_explorer(
            [{"SK_ID_CURR": 1001}],
            search_id="1001",
            prediction_response=response,
        )

        assert not app.exception
        metrics = {metric.label: metric.value for metric in app.metric}
        assert metrics["Credit-risk probability"] == "N/A"
        assert any(
            any(
                term in element.value.lower()
                for term in ("invalid", "probability", "malformed")
            )
            for element in list(app.warning) + list(app.info)
        )


def test_target_zero_is_described_as_no_defined_repayment_difficulty() -> None:
    assert (
        _format_detail_value("TARGET", 0)
        == "0 — No defined repayment difficulty"
    )
    assert _format_detail_value("TARGET", 1.5) == "1.5"


def test_search_distinguishes_invalid_and_not_found_ids() -> None:
    applications = [{"SK_ID_CURR": 1001}]

    invalid_app = _render_explorer(applications, search_id="not-an-id")
    missing_app = _render_explorer(applications, search_id="9999")

    assert not invalid_app.exception
    assert "valid positive integer" in " ".join(
        element.value for element in invalid_app.warning
    )
    assert not missing_app.exception
    assert "No application was found for ID 9999." in [
        element.value for element in missing_app.info
    ]


def test_search_handles_empty_input_and_duplicate_records() -> None:
    empty_search = _render_explorer([{"SK_ID_CURR": 1001}], search_id="   ")
    duplicate_search = _render_explorer(
        [{"SK_ID_CURR": 1001}, {"SK_ID_CURR": 1001}],
        search_id="1001",
    )

    assert not empty_search.exception
    assert "Enter an application ID to search." in [
        element.value for element in empty_search.warning
    ]
    assert not duplicate_search.exception
    assert any("duplicated in the source data" in element.value for element in duplicate_search.warning)


def test_search_handles_null_values_missing_optional_columns_and_empty_data() -> None:
    null_value_app = _render_explorer(
        [{"SK_ID_CURR": 1001, "AMT_CREDIT": None, "TARGET": None}],
        search_id="1001",
    )
    optional_columns_app = _render_explorer(
        [{"SK_ID_CURR": 1002}],
        search_id="1002",
    )
    empty_data_app = _render_explorer([], search_id=None)

    assert not null_value_app.exception
    null_metrics = {metric.label: metric.value for metric in null_value_app.metric}
    assert null_metrics["Credit amount"] == "N/A"
    assert null_metrics["TARGET"] == "N/A"

    assert not optional_columns_app.exception
    optional_metrics = {metric.label: metric.value for metric in optional_columns_app.metric}
    assert optional_metrics["Application ID"] == "1002"
    assert optional_metrics["Credit-risk probability"] == "N/A"

    assert not empty_data_app.exception
    assert any(
        "No application records are available" in element.value
        for element in empty_data_app.info
    )


def _render_explorer(
    rows: list[dict[str, object]],
    search_id: str | None,
    prediction_response: object = _NO_PREDICTION_FIXTURE,
) -> AppTest:
    serialized_rows = pd.DataFrame(rows).to_json(orient="records")
    app_source = (
        "import json\n"
        "import pandas as pd\n"
        "from dashboard.pages import application_explorer\n"
        "from dashboard import data_access\n"
        f"rows = json.loads({serialized_rows!r})\n"
        "application_explorer.data_access.get_applications = lambda: pd.DataFrame(rows)\n"
        "application_explorer.data_access.get_credit_risk_prediction = "
        "lambda application_id: data_access.CreditRiskPrediction("
        "status='unavailable', message='Prediction functionality is not yet available; "
        "no provider is configured.')\n"
    )
    if prediction_response is not _NO_PREDICTION_FIXTURE:
        serialized_prediction = json.dumps(prediction_response)
        app_source += (
            f"response = json.loads({serialized_prediction!r})\n"
            "application_explorer.data_access.get_credit_risk_prediction = "
            "lambda application_id: data_access.normalize_prediction_response(response)\n"
        )
    app_source += "application_explorer.render_application_explorer()\n"
    app = AppTest.from_string(app_source).run()
    if search_id is not None:
        app.text_input[0].set_value(search_id)
        app.button[0].click().run()
    return app
