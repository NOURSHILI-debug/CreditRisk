"""Canonical prediction status mapping for the API contract.

This module is the single place that maps probabilities and failures to the
dashboard-compatible ``PredictionStatus`` values. The legacy predictor helper
``predict_credit_risk()`` intentionally keeps its historical ``success`` /
``error`` values; only the HTTP API uses these canonical statuses.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Literal

PredictionStatus = Literal[
    "available",
    "no_prediction",
    "unavailable",
    "model_unavailable",
    "invalid",
]

STATUS_HTTP_CODES: dict[PredictionStatus, int] = {
    "available": 200,
    "no_prediction": 200,
    "invalid": 422,
    "model_unavailable": 503,
    "unavailable": 503,
}

LEGACY_SUCCESS_STATUS = "success"
LEGACY_ERROR_STATUS = "error"
LEGACY_UNAVAILABLE_STATUS = "unavailable"


def status_for_probability(value: Any) -> PredictionStatus:
    """Map a candidate probability to its canonical status.

    Booleans, non-numeric values, NaN, infinities and values outside ``[0, 1]``
    are invalid. ``None`` means the provider had no prediction.
    """
    if value is None:
        return "no_prediction"
    if isinstance(value, bool) or not isinstance(value, Real):
        return "invalid"
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        return "invalid"
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return "invalid"
    return "available"


def probability_from(value: Any) -> float | None:
    """Return a validated probability, ``None`` for no prediction, or raise."""
    status = status_for_probability(value)
    if status == "available":
        return float(value)
    if status == "no_prediction":
        return None
    raise ValueError(
        "Probability must be a finite number between 0 and 1, or null."
    )


def http_status_for(status: PredictionStatus) -> int:
    """Return the default HTTP status code for a canonical status."""
    try:
        return STATUS_HTTP_CODES[status]
    except KeyError as exc:
        raise ValueError(f"Unsupported prediction status: {status!r}") from exc
