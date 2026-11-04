from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Literal

import pandas as pd

from dashboard.api_client import (
    APIClient,
    APIClientError,
    APIResponseError,
    APIUnavailableError,
)

APPLICATION_ID_COLUMN = "SK_ID_CURR"
MODEL_METRICS = ("ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-score")
_MODEL_IDS = {"xgboost": "XGBoost", "lstm": "LSTM"}
_METRIC_IDS = {
    "rocauc": "ROC-AUC",
    "prauc": "PR-AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1score": "F1-score",
}


def _identifier(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


class DataAccessError(RuntimeError):
    """Base error for dashboard data-provider failures."""


class DataSourceUnavailableError(DataAccessError):
    """The FastAPI source cannot supply dashboard data."""


class MalformedProviderResponseError(DataAccessError):
    """The API returned data outside its documented response contract."""


def _client() -> APIClient:
    return APIClient()


def get_provider_name() -> str:
    return "fastapi"


def is_mock_mode() -> bool:
    return False


ApplicationLookupStatus = Literal[
    "found",
    "invalid_id",
    "not_found",
    "unavailable",
    "duplicate",
]


@dataclass(frozen=True)
class ApplicationLookupResult:
    status: ApplicationLookupStatus
    application: pd.Series | None = None


PredictionStatus = Literal[
    "available",
    "no_prediction",
    "unavailable",
    "model_unavailable",
    "invalid",
]


@dataclass(frozen=True)
class CreditRiskPrediction:
    status: PredictionStatus
    probability: float | None = None
    risk_level: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    prediction_timestamp: str | None = None
    explanation: Any = None
    message: str | None = None


def _prediction_metadata(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Real, datetime, date)):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    return value.isoformat() if isinstance(value, (datetime, date)) else str(value)


def normalize_prediction_response(response: Any) -> CreditRiskPrediction:
    if response is None:
        return CreditRiskPrediction(
            status="no_prediction",
            message="No prediction was returned for this application.",
        )
    if not isinstance(response, Mapping):
        return CreditRiskPrediction(
            status="invalid",
            message="The prediction API returned a malformed response.",
        )

    status = response.get("status")
    if status in ("unavailable", "model_unavailable", "no_prediction"):
        return CreditRiskPrediction(
            status=status,
            message=_prediction_metadata(response.get("message")),
        )
    if status != "available":
        return CreditRiskPrediction(
            status="invalid",
            message="The prediction API returned an unsupported status.",
        )

    raw_probability = response.get("probability")
    if isinstance(raw_probability, bool) or not isinstance(raw_probability, Real):
        return CreditRiskPrediction(
            status="invalid",
            message="The prediction response does not contain a valid probability.",
        )
    probability = float(raw_probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        return CreditRiskPrediction(
            status="invalid",
            message="The prediction probability must be between 0 and 1.",
        )

    risk_level = response.get("risk_level")
    if risk_level not in (None, "LOW", "MEDIUM", "HIGH"):
        return CreditRiskPrediction(
            status="invalid",
            message="The prediction response contains an invalid risk category.",
        )
    return CreditRiskPrediction(
        status="available",
        probability=probability,
        risk_level=risk_level,
        model_name=_prediction_metadata(response.get("model_name")),
        model_version=_prediction_metadata(response.get("model_version")),
        prediction_timestamp=_prediction_metadata(
            response.get("prediction_timestamp")
        ),
        explanation=response.get("explanation"),
        message=_prediction_metadata(response.get("message")),
    )


def parse_application_id(application_id: Any) -> int | None:
    if isinstance(application_id, bool):
        return None
    if isinstance(application_id, Integral):
        parsed_id = int(application_id)
    elif isinstance(application_id, str):
        value = application_id.strip()
        if not re.fullmatch(r"[0-9]+", value):
            return None
        parsed_id = int(value)
    else:
        return None
    return parsed_id if parsed_id > 0 else None


def _raise_data_error(error: APIClientError) -> None:
    if isinstance(error, APIUnavailableError):
        raise DataSourceUnavailableError(str(error)) from error
    if isinstance(error, APIResponseError) and error.status_code == 503:
        raise DataSourceUnavailableError(str(error)) from error
    if isinstance(error, APIResponseError):
        raise MalformedProviderResponseError(str(error)) from error
    raise DataAccessError(str(error)) from error


def get_api_health() -> dict[str, Any]:
    try:
        payload = _client().health()
    except APIClientError as error:
        _raise_data_error(error)
    if not isinstance(payload, dict) or "status" not in payload:
        raise MalformedProviderResponseError("The API returned malformed health data.")
    return payload


def get_applications_page(
    *,
    limit: int = 100,
    offset: int = 0,
    fields: list[str] | None = None,
    application_id: int | None = None,
    filter_field: str | None = None,
    filter_value: str | None = None,
) -> tuple[pd.DataFrame, int]:
    try:
        payload = _client().list_applications(
            limit=limit,
            offset=offset,
            fields=fields,
            application_id=application_id,
            filter_field=filter_field,
            filter_value=filter_value,
        )
    except APIClientError as error:
        _raise_data_error(error)
    items = payload.get("items") if isinstance(payload, dict) else None
    total = payload.get("total") if isinstance(payload, dict) else None
    if (
        not isinstance(items, list)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or any(not isinstance(item, dict) for item in items)
    ):
        raise MalformedProviderResponseError(
            "The application API returned a malformed page."
        )
    return pd.DataFrame(items), total


def get_applications() -> pd.DataFrame:
    """Fetch one real, bounded page for legacy callers."""
    applications, _ = get_applications_page(limit=500, offset=0)
    return applications


def get_application_by_id(
    application_id: Any,
    applications: pd.DataFrame | None = None,
) -> ApplicationLookupResult:
    app_id = parse_application_id(application_id)
    if app_id is None:
        return ApplicationLookupResult(status="invalid_id")
    if applications is not None:
        if not isinstance(applications, pd.DataFrame):
            raise MalformedProviderResponseError(
                "The supplied application records must be a pandas DataFrame."
            )
        if APPLICATION_ID_COLUMN not in applications.columns:
            return ApplicationLookupResult(status="unavailable")
        matches = applications.loc[
            pd.to_numeric(applications[APPLICATION_ID_COLUMN], errors="coerce")
            .eq(app_id)
            .fillna(False)
        ]
        if matches.empty:
            return ApplicationLookupResult(status="not_found")
        if len(matches) > 1:
            return ApplicationLookupResult(status="duplicate")
        return ApplicationLookupResult("found", matches.iloc[0].copy())

    try:
        payload = _client().get_application(app_id)
    except APIResponseError as error:
        if error.status_code == 404:
            return ApplicationLookupResult(status="not_found")
        if error.status_code == 409:
            return ApplicationLookupResult(status="duplicate")
        if error.status_code == 503:
            return ApplicationLookupResult(status="unavailable")
        raise MalformedProviderResponseError(str(error)) from error
    except APIClientError as error:
        _raise_data_error(error)

    application = payload.get("application") if isinstance(payload, dict) else None
    if not isinstance(application, dict):
        raise MalformedProviderResponseError(
            "The application API returned a malformed application."
        )
    return ApplicationLookupResult(
        status="found", application=pd.Series(application)
    )


def get_credit_risk_prediction(
    application_id: Any,
    *,
    include_explanation: bool = True,
) -> CreditRiskPrediction:
    parsed_id = parse_application_id(application_id)
    if parsed_id is None:
        return CreditRiskPrediction(
            status="invalid",
            message="A valid application ID is required for prediction lookup.",
        )
    try:
        response = _client().predict(
            parsed_id, include_explanation=include_explanation
        )
    except APIResponseError as error:
        if error.status_code == 404:
            return CreditRiskPrediction(
                status="no_prediction", message=str(error)
            )
        if error.status_code == 503 and isinstance(error.payload, dict):
            return normalize_prediction_response(error.payload)
        if error.status_code == 503:
            return CreditRiskPrediction(
                status="unavailable", message=str(error)
            )
        if error.status_code == 422:
            return CreditRiskPrediction(status="invalid", message=str(error))
        raise MalformedProviderResponseError(str(error)) from error
    except APIClientError as error:
        _raise_data_error(error)
    return normalize_prediction_response(response)


def get_model_info() -> dict[str, Any]:
    try:
        payload = _client().model_info()
    except APIClientError as error:
        _raise_data_error(error)
    if not isinstance(payload, dict):
        raise MalformedProviderResponseError("The model API returned malformed metadata.")
    return payload


def get_prediction_summary(split: str = "test") -> dict[str, Any]:
    try:
        payload = _client().prediction_summary(split)
    except APIClientError as error:
        _raise_data_error(error)
    if not isinstance(payload, dict):
        raise MalformedProviderResponseError(
            "The model API returned a malformed prediction summary."
        )
    return payload


def get_overview_metrics(applications: pd.DataFrame | None = None) -> dict[str, Any]:
    del applications
    _, total = get_applications_page(
        limit=1, offset=0, fields=[APPLICATION_ID_COLUMN]
    )
    summary = get_prediction_summary()
    return {
        "total_applications": total,
        "prediction_count": summary["prediction_count"],
        "prediction_split": summary["split"],
        "predicted_outcomes": summary["predicted_outcomes"],
        "risk_distribution": summary["risk_distribution"],
        "risk_bands": summary["risk_bands"],
        "decision_threshold": summary["decision_threshold"],
        "source": summary["source"],
    }


def get_model_evaluation_results(split: str = "test") -> pd.DataFrame:
    try:
        payload = _client().evaluation(split)
    except APIClientError as error:
        _raise_data_error(error)
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise MalformedProviderResponseError(
            "The evaluation API returned malformed metric rows."
        )
    results = pd.DataFrame(rows, columns=["model", "metric", "value"])
    results.attrs["split"] = payload.get("split")
    results.attrs["provenance"] = payload.get("provenance", {})
    results.attrs["confusion_matrix"] = payload.get("confusion_matrix")
    return results


def normalize_model_evaluation_results(
    results: pd.DataFrame | None,
) -> tuple[dict[str, dict[str, float | None]], dict[str, str], list[str]]:
    models = ("XGBoost", "LSTM")
    values = {model: {metric: None for metric in MODEL_METRICS} for model in models}
    supplied_models: set[str] = set()
    issues: list[str] = []
    if results is not None and not isinstance(results, pd.DataFrame):
        raise MalformedProviderResponseError(
            "Model-evaluation results must be a pandas DataFrame or None."
        )
    if results is None or results.empty:
        return values, {model: "Not available" for model in models}, issues
    required = {"model", "metric", "value"}
    missing = required.difference(results.columns)
    if missing:
        issues.append(
            "Evaluation results are missing required columns: "
            + ", ".join(sorted(missing))
            + ". Expected model, metric, and value."
        )
        return values, {model: "Invalid results" for model in models}, issues

    occurrences: Counter[tuple[str, str]] = Counter()
    parsed: dict[tuple[str, str], float | None] = {}
    for row in results.loc[:, ["model", "metric", "value"]].itertuples(index=False):
        model = _MODEL_IDS.get(_identifier(row.model))
        metric = _METRIC_IDS.get(_identifier(row.metric))
        if model is None or metric is None:
            issues.append(f"Unrecognized model or metric: {row.model!r}/{row.metric!r}.")
            continue
        supplied_models.add(model)
        key = (model, metric)
        occurrences[key] += 1
        try:
            value = float(row.value)
        except (TypeError, ValueError, OverflowError):
            value = math.nan
        parsed[key] = value if math.isfinite(value) and 0 <= value <= 1 else None

    for key, count in occurrences.items():
        if count > 1:
            parsed[key] = None
            issues.append(
                f"Duplicate evaluation rows for {key[0]} / {key[1]}; value marked N/A."
            )
        elif parsed[key] is None:
            issues.append(
                f"Invalid or missing value for {key[0]} / {key[1]}; "
                "expected a finite score from 0 to 1."
            )
        else:
            values[key[0]][key[1]] = parsed[key]

    statuses: dict[str, str] = {}
    for model in models:
        count = sum(value is not None for value in values[model].values())
        if model not in supplied_models:
            statuses[model] = "Not available"
        elif count == len(MODEL_METRICS):
            statuses[model] = "Available"
        elif count:
            statuses[model] = f"Partially available ({count}/{len(MODEL_METRICS)} metrics)"
        else:
            statuses[model] = "Invalid or incomplete results"
    return values, statuses, issues
