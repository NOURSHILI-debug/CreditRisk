from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard import data_access
from dashboard.data_access import (
    ApplicationLookupResult,
    normalize_prediction_response,
)
from dashboard.pages.application_explorer import (
    _DETAIL_GROUPS,
    _format_detail_value,
    _render_application_table,
    _render_prediction_result,
    render_application_explorer,
)


def test_prediction_response_normalizes_probability_risk_and_explanation() -> None:
    prediction = normalize_prediction_response(
        {
            "status": "available",
            "probability": 0.1234,
            "risk_level": "LOW",
            "model_name": "XGBoost",
            "model_version": "test-version",
            "prediction_timestamp": "2026-11-04T12:00:00Z",
            "explanation": {"top_features": []},
        }
    )
    assert prediction.status == "available"
    assert prediction.probability == 0.1234
    assert prediction.risk_level == "LOW"
    assert prediction.explanation == {"top_features": []}


def test_prediction_response_rejects_invalid_probabilities() -> None:
    for probability in (True, "not numeric", float("nan"), float("inf"), -0.01, 1.01):
        prediction = normalize_prediction_response(
            {"status": "available", "probability": probability}
        )
        assert prediction.status == "invalid"
        assert prediction.probability is None


def test_detail_formatting_does_not_special_case_target() -> None:
    assert _format_detail_value("AMT_CREDIT", 50000) == "50,000"
    assert _format_detail_value("DAYS_BIRTH", -12000) == "32.9 years"
    assert _format_detail_value("DAYS_EMPLOYED", -365) == "-365 source days"
    assert _format_detail_value("NAME_INCOME_TYPE", "Working") == "Working"
    assert "TARGET" not in {
        field for _, fields in _DETAIL_GROUPS for field, _ in fields
    }


def test_application_table_never_displays_target() -> None:
    applications = pd.DataFrame(
        {"SK_ID_CURR": [1001], "AMT_CREDIT": [50000], "TARGET": [1]}
    )
    app = AppTest.from_string(
        "import pandas as pd\n"
        "from dashboard.pages.application_explorer import _render_application_table\n"
        "applications = pd.DataFrame({'SK_ID_CURR': [1001], "
        "'AMT_CREDIT': [50000], 'TARGET': [1]})\n"
        "_render_application_table(applications)\n"
    ).run()
    assert not app.exception
    assert "TARGET" not in app.dataframe[0].value.columns


def test_prediction_view_renders_real_shap_factors_and_api_thresholds(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        data_access,
        "get_model_info",
        lambda: {
            "thresholds": {
                "risk_bands": {"low_upper": 0.3, "medium_upper": 0.6}
            }
        },
    )
    app = AppTest.from_string(
        "from dashboard.data_access import CreditRiskPrediction\n"
        "from dashboard.pages.application_explorer import _render_prediction_result\n"
        "prediction = CreditRiskPrediction(status='available', probability=0.75, "
        "risk_level='HIGH', model_name='XGBoost', model_version='v2', "
        "prediction_timestamp='2026-11-04T12:00:00Z', explanation={"
        "'top_features': [{'raw_feature': 'EXT_SOURCE_2', 'raw_value': 0.2, "
        "'effect': 'increases_risk'}], 'summary': 'Tree SHAP.'})\n"
        "_render_prediction_result(prediction)\n"
    ).run()
    assert not app.exception
    assert "Probability of repayment difficulty" in [
        metric.label for metric in app.metric
    ]
    assert any("LOW < 30%" in caption.value for caption in app.caption)
    assert "Ext Source 2" in app.dataframe[0].value.iloc[0, 0]


def test_unavailable_api_is_reported_without_sample_applicants(monkeypatch) -> None:
    def unavailable(**kwargs):
        raise data_access.DataSourceUnavailableError("API connection refused")

    monkeypatch.setattr(data_access, "get_applications_page", unavailable)
    app = AppTest.from_string(
        "from dashboard.pages.application_explorer import render_application_explorer\n"
        "render_application_explorer()\n"
    ).run()
    assert not app.exception
    assert any("Applicant data is unavailable" in item.value for item in app.warning)
    assert any("No sample records are shown" in item.value for item in app.warning)
    assert "API connection refused" not in str(app)


def test_nonfirst_applicant_selection_survives_details_rerun(monkeypatch) -> None:
    applications = pd.DataFrame(
        [
            {"SK_ID_CURR": 1001, "AMT_CREDIT": 50000},
            {"SK_ID_CURR": 1002, "AMT_CREDIT": 75000},
        ]
    )
    monkeypatch.setattr(
        data_access,
        "get_applications_page",
        lambda **kwargs: (applications, len(applications)),
    )
    monkeypatch.setattr(
        data_access,
        "get_application_by_id",
        lambda application_id: ApplicationLookupResult(
            status="found",
            application=pd.Series(
                {"SK_ID_CURR": application_id, "AMT_CREDIT": 75000}
            ),
        ),
    )

    app = AppTest.from_string(
        "from dashboard.pages.application_explorer import render_application_explorer\n"
        "render_application_explorer()\n"
    ).run()
    app.selectbox[2].select(1002).run()
    next(
        button for button in app.button if button.label == "Load applicant details"
    ).click().run()

    assert not app.exception
    assert app.session_state["selected_application"] == 1002
    assert app.selectbox[2].value == 1002
