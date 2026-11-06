from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest

from dashboard import data_access
from dashboard.api_client import APIClient


def _client(
    handler,
) -> tuple[APIClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return APIClient("http://api.test", client=http_client), http_client


def test_dashboard_defaults_to_fastapi_and_never_uses_mock_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CREDIT_RISK_DATA_PROVIDER", raising=False)
    assert data_access.get_provider_name() == "fastapi"
    assert not data_access.is_mock_mode()


def test_application_page_uses_real_api_pagination_and_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/applications"
        assert request.url.params["limit"] == "25"
        assert request.url.params["offset"] == "50"
        assert request.url.params["filter_field"] == "NAME_INCOME_TYPE"
        assert request.url.params["filter_value"] == "Working"
        return httpx.Response(
            200,
            json={
                "total": 26,
                "limit": 25,
                "offset": 50,
                "fields": ["SK_ID_CURR", "NAME_INCOME_TYPE"],
                "items": [
                    {"SK_ID_CURR": 1001, "NAME_INCOME_TYPE": "Working"}
                ],
            },
        )

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        frame, total = data_access.get_applications_page(
            limit=25,
            offset=50,
            fields=["SK_ID_CURR", "NAME_INCOME_TYPE"],
            filter_field="NAME_INCOME_TYPE",
            filter_value="Working",
        )
    finally:
        http_client.close()
    assert total == 26
    assert frame.to_dict("records") == [
        {"SK_ID_CURR": 1001, "NAME_INCOME_TYPE": "Working"}
    ]


def test_application_lookup_and_prediction_are_backed_by_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/applications/1001":
            return httpx.Response(
                200,
                json={
                    "application_id": 1001,
                    "application": {"SK_ID_CURR": 1001, "AMT_CREDIT": 50000},
                },
            )
        if request.url.path == "/applications/1001/prediction":
            assert request.url.params["include_explanation"] == "true"
            return httpx.Response(
                200,
                json={
                    "status": "available",
                    "application_id": 1001,
                    "probability": 0.25,
                    "risk_level": "LOW",
                    "model_name": "XGBoost",
                    "model_version": "test-version",
                    "prediction_timestamp": "2026-11-04T12:00:00Z",
                    "explanation": {"top_features": []},
                    "message": "Prediction generated successfully.",
                },
            )
        return httpx.Response(404, json={"message": "not found"})

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        application = data_access.get_application_by_id(1001)
        prediction = data_access.get_credit_risk_prediction(1001)
    finally:
        http_client.close()
    assert application.status == "found"
    assert application.application is not None
    assert application.application["AMT_CREDIT"] == 50000
    assert prediction.status == "available"
    assert prediction.probability == 0.25
    assert prediction.risk_level == "LOW"
    assert prediction.explanation == {"top_features": []}


def test_prediction_without_explanation_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/applications/1001/prediction"
        assert request.url.params["include_explanation"] == "false"
        return httpx.Response(
            200,
            json={
                "status": "available",
                "application_id": 1001,
                "probability": 0.25,
                "model_name": "XGBoost",
                "model_version": "v2",
                "prediction_timestamp": "2026-11-04T12:00:00Z",
            },
        )

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        result = data_access.get_credit_risk_prediction(
            1001, include_explanation=False
        )
    finally:
        http_client.close()

    assert result.status == "available"
    assert result.probability == 0.25
    assert result.explanation is None


@pytest.mark.parametrize(
    ("response_status", "payload", "expected_status"),
    [
        (404, {"status": "no_prediction", "message": "Application missing."}, "not_found"),
        (
            503,
            {
                "status": "model_unavailable",
                "message": "Model artifacts are unavailable.",
                "model_name": "XGBoost",
                "model_version": "v2",
                "prediction_timestamp": "2026-11-04T12:00:00Z",
            },
            "model_unavailable",
        ),
        (
            503,
            {"status": "unavailable", "message": "Application data unavailable."},
            "unavailable",
        ),
    ],
)
def test_prediction_http_errors_keep_distinct_statuses(
    monkeypatch: pytest.MonkeyPatch,
    response_status: int,
    payload: dict[str, object],
    expected_status: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/applications/1001/prediction"
        assert request.url.params["include_explanation"] == "false"
        return httpx.Response(response_status, json=payload)

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        result = data_access.get_credit_risk_prediction(
            1001, include_explanation=False
        )
    finally:
        http_client.close()

    assert result.status == expected_status
    assert result.probability is None
    if expected_status == "model_unavailable":
        assert result.model_name == "XGBoost"
        assert result.model_version == "v2"
        assert result.prediction_timestamp == "2026-11-04T12:00:00Z"


def test_valid_application_without_prediction_is_not_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["include_explanation"] == "false"
        return httpx.Response(
            200,
            json={
                "status": "no_prediction",
                "application_id": 1001,
                "probability": None,
                "message": "No prediction is available.",
            },
        )

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        result = data_access.get_credit_risk_prediction(
            1001, include_explanation=False
        )
    finally:
        http_client.close()

    assert result.status == "no_prediction"
    assert result.probability is None


def test_unexpected_prediction_404_is_not_reported_as_missing_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/applications/1001/prediction"
        return httpx.Response(404, json={"status": "invalid", "message": "missing"})

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        result = data_access.get_credit_risk_prediction(
            1001, include_explanation=False
        )
    finally:
        http_client.close()

    assert result.status == "invalid"
    assert result.probability is None


def test_prediction_connection_and_malformed_response_errors_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def connection_failure(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, http_client = _client(connection_failure)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        with pytest.raises(data_access.DataSourceUnavailableError):
            data_access.get_credit_risk_prediction(
                1001, include_explanation=False
            )
    finally:
        http_client.close()

    def malformed_success(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/applications/1001/prediction"
        return httpx.Response(
            200,
            json={"status": "available", "probability": "not-a-number"},
        )

    client, http_client = _client(malformed_success)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        malformed = data_access.get_credit_risk_prediction(
            1001, include_explanation=False
        )
    finally:
        http_client.close()
    assert malformed.status == "invalid"
    assert malformed.probability is None


def test_prediction_execution_failure_has_error_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/applications/1001/prediction"
        return httpx.Response(
            500, json={"status": "invalid", "message": "Internal server error."}
        )

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        result = data_access.get_credit_risk_prediction(
            1001, include_explanation=False
        )
    finally:
        http_client.close()

    assert result.status == "error"
    assert result.probability is None


def test_missing_application_is_not_misreported_as_api_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"status": "invalid", "message": "missing"})

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        result = data_access.get_application_by_id(999)
    finally:
        http_client.close()
    assert result.status == "not_found"


def test_overview_and_evaluation_use_backend_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/applications":
            return httpx.Response(
                200,
                json={
                    "total": 307511,
                    "limit": 1,
                    "offset": 0,
                    "fields": ["SK_ID_CURR"],
                    "items": [{"SK_ID_CURR": 1001}],
                },
            )
        if request.url.path == "/model/prediction-summary":
            return httpx.Response(
                200,
                json={
                    "split": "test",
                    "prediction_count": 3,
                    "predicted_outcomes": {
                        "no_defined_repayment_difficulty": 2,
                        "repayment_difficulty": 1,
                    },
                    "risk_distribution": {"LOW": 1, "MEDIUM": 1, "HIGH": 1},
                    "risk_bands": {"low_upper": 0.3, "medium_upper": 0.6},
                    "decision_threshold": 0.5,
                    "source": "test_predictions.csv",
                },
            )
        if request.url.path == "/model/evaluation":
            return httpx.Response(
                200,
                json={
                    "split": "test",
                    "rows": [
                        {"model": "XGBoost", "metric": "ROC-AUC", "value": 0.8}
                    ],
                    "provenance": {
                        "n_samples": 3,
                        "target": {
                            "name": "TARGET",
                            "positive_class": 1,
                            "probability_semantics": "P(TARGET=1)",
                        },
                    },
                    "confusion_matrix": {"tn": 1, "fp": 0, "fn": 1, "tp": 1},
                },
            )
        return httpx.Response(
            200,
            json={
                "model_name": "XGBoost",
                "model_version": "v1",
                "model_hash": "hash",
                "thresholds": {},
            },
        )

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        summary = data_access.get_overview_metrics()
        evaluation = data_access.get_model_evaluation_results("test")
    finally:
        http_client.close()
    assert summary["total_applications"] == 307511
    assert summary["prediction_count"] == 3
    assert evaluation.iloc[0]["value"] == 0.8
    assert evaluation.attrs["split"] == "test"
    assert evaluation.attrs["provenance"]["target"]["positive_class"] == 1
    assert evaluation.attrs["confusion_matrix"]["tp"] == 1
    assert data_access.normalize_model_evaluation_results(evaluation)[0][
        "XGBoost"
    ]["ROC-AUC"] == 0.8


def test_unreachable_api_raises_an_explicit_data_source_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        with pytest.raises(data_access.DataSourceUnavailableError, match="unavailable"):
            data_access.get_applications_page(limit=10, offset=0)
    finally:
        http_client.close()


def test_malformed_application_page_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"items": "bad"}))

    client, http_client = _client(handler)
    monkeypatch.setattr(data_access, "_client", lambda: client)
    try:
        with pytest.raises(data_access.MalformedProviderResponseError):
            data_access.get_applications_page(limit=10, offset=0)
    finally:
        http_client.close()
