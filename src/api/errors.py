"""Uniform error body construction for every API error response."""

from __future__ import annotations

from typing import Any

from api.status import PredictionStatus


def error_payload(
    status: PredictionStatus,
    message: str,
    *,
    application_id: int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    prediction_timestamp: str | None = None,
) -> dict[str, Any]:
    """Return the single error shape shared by all endpoints."""
    return {
        "status": status,
        "message": message,
        "application_id": application_id,
        "probability": None,
        "risk_level": None,
        "model_name": model_name,
        "model_version": model_version,
        "prediction_timestamp": prediction_timestamp,
        "explanation": None,
    }
