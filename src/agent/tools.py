from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from dashboard import data_access

_NON_APPLICATION_OUTPUT_FIELDS = {"RISK_PROBABILITY", "RISK_CATEGORY"}


def _source_mode() -> str:
    return "mock" if data_access.is_mock_mode() else "configured"


def _json_value(value: Any) -> Any:
    """Convert provider scalars and containers to strict JSON-compatible values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]

    if pd.api.types.is_scalar(value):
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Unsupported provider value type: {type(value).__name__}.")


def _application_id_for_output(application_id: Any) -> int | str | None:
    if isinstance(application_id, bool):
        return None
    if isinstance(application_id, Integral):
        return int(application_id)
    if isinstance(application_id, str):
        return application_id.strip()
    return None


def get_application_details(application_id: int | str) -> dict[str, Any]:
    """Retrieve a source application as JSON-compatible fields and lookup status."""
    source_mode = _source_mode()
    response: dict[str, Any] = {
        "status": "unavailable",
        "application_id": _application_id_for_output(application_id),
        "fields": {},
        "null_fields": [],
        "source_mode": source_mode,
    }

    try:
        result = data_access.get_application_by_id(application_id)
    except data_access.DataSourceUnavailableError as error:
        response["message"] = str(error)
        return response
    except data_access.MalformedProviderResponseError as error:
        response["status"] = "invalid"
        response["message"] = str(error)
        return response

    response["status"] = result.status
    if result.application is None:
        messages = {
            "invalid_id": "A valid positive integer application ID is required.",
            "not_found": "No application was found for this ID.",
            "unavailable": "Application lookup is unavailable for the configured source.",
            "duplicate": "The application ID is duplicated; no record was selected.",
        }
        response["message"] = messages.get(result.status, "Application details are unavailable.")
        return response

    fields: dict[str, Any] = {}
    null_fields: list[str] = []
    try:
        for field, value in result.application.items():
            if field in _NON_APPLICATION_OUTPUT_FIELDS:
                continue
            if value is None or value is pd.NA or value is pd.NaT:
                null_fields.append(str(field))
                continue
            try:
                is_missing = pd.api.types.is_scalar(value) and bool(pd.isna(value))
            except (TypeError, ValueError):
                is_missing = False
            if is_missing:
                null_fields.append(str(field))
            else:
                fields[str(field)] = _json_value(value)
    except TypeError as error:
        response["status"] = "invalid"
        response["message"] = str(error)
        return response

    response["fields"] = fields
    response["null_fields"] = null_fields
    return response


def get_dashboard_metrics() -> dict[str, Any]:
    """Retrieve shared overview metrics for the configured application snapshot."""
    source_mode = _source_mode()
    try:
        metrics = data_access.get_overview_metrics()
    except data_access.DataSourceUnavailableError as error:
        return {
            "status": "unavailable",
            "metrics": None,
            "source_mode": source_mode,
            "message": str(error),
        }
    except data_access.MalformedProviderResponseError as error:
        return {
            "status": "invalid",
            "metrics": None,
            "source_mode": source_mode,
            "message": str(error),
        }

    status = (
        "available"
        if metrics["total_applications"] > 0
        else "unavailable"
    )
    response = {
        "status": status,
        "metrics": _json_value(metrics),
        "source_mode": source_mode,
    }
    if metrics["total_applications"] == 0:
        response["message"] = "No application records are available."
    return response


def get_model_comparison() -> dict[str, Any]:
    """Retrieve and normalize actual evaluation rows without filling missing values."""
    source_mode = _source_mode()
    try:
        results = data_access.get_model_evaluation_results()
        values, statuses, issues = data_access.normalize_model_evaluation_results(results)
    except data_access.DataSourceUnavailableError as error:
        return {
            "status": "unavailable",
            "models": {},
            "statuses": {},
            "issues": [],
            "source_mode": source_mode,
            "message": str(error),
        }
    except data_access.MalformedProviderResponseError as error:
        return {
            "status": "invalid",
            "models": {},
            "statuses": {},
            "issues": [],
            "source_mode": source_mode,
            "message": str(error),
        }

    model_metrics = {
        model: _json_value(metrics)
        for model, metrics in values.items()
    }
    serialized_issues = _json_value(issues)
    if results is None or results.empty:
        status = "unavailable"
        message = "Model evaluation results are not available."
    elif not any(
        score is not None
        for metrics in values.values()
        for score in metrics.values()
    ):
        status = "invalid" if serialized_issues else "unavailable"
        message = "No valid model evaluation metrics are available."
    elif any(
        score is None
        for metrics in values.values()
        for score in metrics.values()
    ):
        status = "partial"
        message = "Some model evaluation metrics are unavailable or invalid."
    else:
        status = "available"
        message = None

    response = {
        "status": status,
        "models": model_metrics,
        "statuses": _json_value(statuses),
        "issues": serialized_issues,
        "source_mode": source_mode,
    }
    if message is not None:
        response["message"] = message
    return response


def get_prediction(application_id: int | str) -> dict[str, Any]:
    """Retrieve a prediction using the Task 9 data-access response contract."""
    source_mode = _source_mode()
    try:
        prediction = data_access.get_credit_risk_prediction(application_id)
    except data_access.DataSourceUnavailableError as error:
        return {
            "status": "unavailable",
            "probability": None,
            "model_name": None,
            "model_version": None,
            "prediction_timestamp": None,
            "explanation": None,
            "message": str(error),
            "source_mode": source_mode,
        }
    except data_access.MalformedProviderResponseError as error:
        return {
            "status": "invalid",
            "probability": None,
            "model_name": None,
            "model_version": None,
            "prediction_timestamp": None,
            "explanation": None,
            "message": str(error),
            "source_mode": source_mode,
        }

    try:
        response = _json_value(
            {
                "status": prediction.status,
                "probability": prediction.probability,
                "model_name": prediction.model_name,
                "model_version": prediction.model_version,
                "prediction_timestamp": prediction.prediction_timestamp,
                "explanation": prediction.explanation,
                "message": prediction.message,
                "source_mode": source_mode,
            }
        )
    except TypeError as error:
        return {
            "status": "invalid",
            "probability": None,
            "model_name": None,
            "model_version": None,
            "prediction_timestamp": None,
            "explanation": None,
            "message": str(error),
            "source_mode": source_mode,
        }
    return response