from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Literal, Optional, Protocol

import numpy as np
import pandas as pd

APPLICATION_ID_COLUMN = "SK_ID_CURR"
MODEL_NAMES = ("XGBoost", "LSTM")
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
    """Normalize external model and metric IDs for matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


class DataAccessError(RuntimeError):
    """Base error for dashboard data-provider failures."""


class DataSourceUnavailableError(DataAccessError):
    """Raised when the selected provider cannot supply dashboard data."""


class MalformedProviderResponseError(DataAccessError):
    """Raised when a provider returns a response outside its documented contract."""


class DashboardDataProvider(Protocol):
    """Provider contract shared by mock and future application-data sources."""

    def get_applications(self) -> pd.DataFrame: ...

    def get_prediction(self, application_id: int) -> Any: ...

    def get_model_evaluation_results(self) -> Optional[pd.DataFrame]: ...


def get_provider_name() -> str:
    """Return the centrally configured dashboard data-provider name."""
    return os.environ.get("CREDIT_RISK_DATA_PROVIDER", "mock").strip().lower()


def is_mock_mode() -> bool:
    """Whether the configured dashboard provider is the explicitly labeled mock source."""
    return get_provider_name() == "mock"


MOCK_MODE = is_mock_mode()

ApplicationLookupStatus = Literal[
    "found",
    "invalid_id",
    "not_found",
    "unavailable",
    "duplicate",
]


@dataclass(frozen=True)
class ApplicationLookupResult:
    """Outcome of looking up an application in the configured data source."""

    status: ApplicationLookupStatus
    application: Optional[pd.Series] = None


PredictionStatus = Literal[
    "available",
    "no_prediction",
    "unavailable",
    "model_unavailable",
    "invalid",
]


@dataclass(frozen=True)
class CreditRiskPrediction:
    """Validated prediction payload returned to the dashboard."""

    status: PredictionStatus
    probability: Optional[float] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    prediction_timestamp: Optional[str] = None
    explanation: Any = None
    message: Optional[str] = None


def _prediction_metadata(value: Any) -> Optional[str]:
    """Convert optional scalar metadata to displayable text."""
    if value is None:
        return None
    if not isinstance(value, (str, Real, datetime, date)):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def normalize_prediction_response(response: Any) -> CreditRiskPrediction:
    """Validate the prediction provider's documented response mapping.

    The probability contract is a finite numeric value in [0, 1]. Missing or
    invalid probabilities never produce an available prediction.
    """
    if response is None:
        return CreditRiskPrediction(
            status="no_prediction",
            message="No prediction was returned for this application.",
        )
    if not isinstance(response, Mapping):
        return CreditRiskPrediction(
            status="invalid",
            message="The prediction provider returned a malformed response.",
        )

    status = response.get("status")
    if status in ("unavailable", "model_unavailable", "no_prediction"):
        return CreditRiskPrediction(
            status=status,
            message=_prediction_metadata(response.get("message")),
        )
    if status not in (None, "available"):
        return CreditRiskPrediction(
            status="invalid",
            message="The prediction provider returned an unsupported status.",
        )

    raw_probability = response.get("probability")
    if isinstance(raw_probability, bool) or not isinstance(raw_probability, Real):
        return CreditRiskPrediction(
            status="invalid",
            message="The prediction response does not contain a valid probability.",
        )
    try:
        probability = float(raw_probability)
    except (TypeError, ValueError, OverflowError):
        return CreditRiskPrediction(
            status="invalid",
            message="The prediction response does not contain a valid probability.",
        )
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        return CreditRiskPrediction(
            status="invalid",
            message="The prediction probability must be between 0 and 1.",
        )

    timestamp = response.get("prediction_timestamp")
    return CreditRiskPrediction(
        status="available",
        probability=probability,
        model_name=_prediction_metadata(response.get("model_name")),
        model_version=_prediction_metadata(response.get("model_version")),
        prediction_timestamp=_prediction_metadata(timestamp),
        explanation=response.get("explanation"),
    )


def get_credit_risk_prediction(application_id: Any) -> CreditRiskPrediction:
    """Return a prediction for an application from the configured provider.

    Providers must return the documented response mapping handled by
    :func:`normalize_prediction_response`. Provider-selection failures become
    an explicit unavailable result; malformed responses become invalid results.
    """
    parsed_id = _parse_application_id(application_id)
    if parsed_id is None:
        return CreditRiskPrediction(
            status="invalid",
            message="A valid application ID is required for prediction lookup.",
        )
    try:
        response = get_data_provider().get_prediction(parsed_id)
    except DataSourceUnavailableError as error:
        return CreditRiskPrediction(status="unavailable", message=str(error))
    return normalize_prediction_response(response)


def _build_mock_applications() -> pd.DataFrame:
    """Construct a small synthetic application dataset for local dashboard development."""
    rng = np.random.default_rng(42)
    row_count = 120

    sk_ids = np.arange(1001, 1001 + row_count)
    income = np.clip(rng.lognormal(mean=10.3, sigma=0.55, size=row_count) * 1000, 15000, 500000)
    credit = np.clip(rng.lognormal(mean=11.1, sigma=0.5, size=row_count) * 1500, 40000, 2500000)
    annuity = np.clip(credit * rng.uniform(0.12, 0.32, size=row_count), 4500, 180000)
    target = rng.binomial(1, 0.27, size=row_count)
    risk_probability = np.clip(0.08 + 0.72 * target + rng.normal(0.0, 0.12, size=row_count), 0.01, 0.99)
    risk_category = np.where(
        risk_probability >= 0.60,
        "High Risk",
        np.where(risk_probability >= 0.35, "Medium Risk", "Low Risk"),
    )

    income_types = [
        "Working",
        "Commercial associate",
        "Pensioner",
        "State servant",
        "Student",
    ]
    education_types = [
        "Higher education",
        "Secondary / secondary special",
        "Incomplete higher",
        "Lower secondary",
    ]

    data = {
        "SK_ID_CURR": sk_ids,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "NAME_INCOME_TYPE": rng.choice(income_types, size=row_count),
        "NAME_EDUCATION_TYPE": rng.choice(education_types, size=row_count),
        "TARGET": target,
        "RISK_PROBABILITY": np.round(risk_probability, 3),
        "RISK_CATEGORY": risk_category,
    }

    df = pd.DataFrame(data)
    df["RISK_PROBABILITY"] = df["RISK_PROBABILITY"].astype(float)
    df["TARGET"] = df["TARGET"].astype(int)
    return df


_MOCK_APPLICATIONS = _build_mock_applications()


class MockDataProvider:
    """Synthetic application provider used for local dashboard development."""

    def get_applications(self) -> pd.DataFrame:
        return _MOCK_APPLICATIONS.copy()

    def get_prediction(self, application_id: int) -> Any:
        del application_id
        return {
            "status": "unavailable",
            "message": "Prediction functionality is not yet available; no provider is configured.",
        }

    def get_model_evaluation_results(self) -> Optional[pd.DataFrame]:
        return None


_MOCK_PROVIDER = MockDataProvider()


def get_data_provider() -> DashboardDataProvider:
    """Select the configured provider in one place.

    ``CREDIT_RISK_DATA_PROVIDER=mock`` is the supported provider today. Other
    provider adapters can implement :class:`DashboardDataProvider` without
    changing dashboard pages.
    """
    provider_name = get_provider_name()
    if provider_name == "mock":
        return _MOCK_PROVIDER
    if not provider_name:
        raise DataSourceUnavailableError(
            "CREDIT_RISK_DATA_PROVIDER must not be empty."
        )
    raise DataSourceUnavailableError(
        f"Dashboard data provider {provider_name!r} is not available."
    )


def _validate_application_frame(applications: Any) -> pd.DataFrame:
    if not isinstance(applications, pd.DataFrame):
        raise MalformedProviderResponseError(
            "The application provider must return a pandas DataFrame."
        )
    return applications


def get_applications() -> pd.DataFrame:
    """Return a copy of application records from the configured provider.

    Empty frames are valid. A provider-selection failure raises
    :class:`DataSourceUnavailableError`; a non-DataFrame response raises
    :class:`MalformedProviderResponseError`.
    """
    applications = get_data_provider().get_applications()
    return _validate_application_frame(applications).copy()


def _parse_application_id(application_id: Any) -> Optional[int]:
    """Parse the positive integer application IDs used by the configured source."""
    if isinstance(application_id, bool):
        return None
    if isinstance(application_id, Integral):
        parsed_id = int(application_id)
    elif isinstance(application_id, str):
        value = application_id.strip()
        if not re.fullmatch(r"[0-9]+", value):
            return None
        try:
            parsed_id = int(value)
        except ValueError:
            return None
    else:
        return None

    return parsed_id if parsed_id > 0 else None


def get_application_by_id(
    application_id: Any,
    applications: Optional[pd.DataFrame] = None,
) -> ApplicationLookupResult:
    """Look up one application in the configured source or a supplied source frame.

    ``applications`` lets callers reuse an already-loaded source snapshot. When
    omitted, the function queries :func:`get_applications`.
    """
    app_id = _parse_application_id(application_id)
    if app_id is None:
        return ApplicationLookupResult(status="invalid_id")

    source = get_applications() if applications is None else _validate_application_frame(applications)
    if source.empty or APPLICATION_ID_COLUMN not in source.columns:
        return ApplicationLookupResult(status="unavailable")

    source_ids = pd.to_numeric(source[APPLICATION_ID_COLUMN], errors="coerce")
    matches = source.loc[source_ids.eq(app_id)]
    if matches.empty:
        return ApplicationLookupResult(status="not_found")
    if len(matches) > 1:
        return ApplicationLookupResult(status="duplicate")

    return ApplicationLookupResult(
        status="found",
        application=matches.iloc[0].copy(),
    )


def get_valid_target_values(applications: pd.DataFrame) -> Optional[pd.Series]:
    """Return recognized binary target values, or ``None`` if none are present."""
    applications = _validate_application_frame(applications)
    if "TARGET" not in applications.columns:
        return None

    target = pd.to_numeric(applications["TARGET"], errors="coerce")
    target = target.where(target.isin([0, 1])).dropna()
    return target if not target.empty else None


def get_overview_metrics(
    applications: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Calculate overview metrics from one configured application snapshot.

    The existing keys ``total_applications``, ``target_0``, ``target_1``, and
    ``risk_distribution`` are preserved. Missing/invalid target labels remain
    unavailable (``None``), while absent risk categories produce an empty
    distribution. ``applications`` allows a page to reuse its loaded snapshot.
    """
    frame = get_applications() if applications is None else _validate_application_frame(applications)
    target = get_valid_target_values(frame)
    if target is None:
        target_0 = None
        target_1 = None
        difficulty_rate = None
        unlabeled = len(frame)
    else:
        target_0 = int(target.eq(0).sum())
        target_1 = int(target.eq(1).sum())
        labeled_count = target_0 + target_1
        difficulty_rate = target_1 / labeled_count if labeled_count else None
        unlabeled = len(frame) - labeled_count

    risk_distribution = (
        frame["RISK_CATEGORY"].value_counts().to_dict()
        if "RISK_CATEGORY" in frame.columns
        else {}
    )
    return {
        "total_applications": int(len(frame)),
        "target_0": target_0,
        "target_1": target_1,
        "difficulty_rate": difficulty_rate,
        "unlabeled": unlabeled,
        "risk_distribution": risk_distribution,
    }


def normalize_model_evaluation_results(
    results: Optional[pd.DataFrame],
) -> tuple[dict[str, dict[str, Optional[float]]], dict[str, str], list[str]]:
    """Validate model/metric rows using the dashboard's established contract."""
    values = {model: {metric: None for metric in MODEL_METRICS} for model in MODEL_NAMES}
    supplied_models: set[str] = set()
    issues: list[str] = []

    if results is not None and not isinstance(results, pd.DataFrame):
        raise MalformedProviderResponseError(
            "Model-evaluation results must be a pandas DataFrame or None."
        )
    if results is None or results.empty:
        return values, {model: "Not available" for model in MODEL_NAMES}, issues

    required_columns = {"model", "metric", "value"}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        issues.append(
            "Evaluation results are missing required columns: "
            + ", ".join(sorted(missing_columns))
            + ". Expected model, metric, and value."
        )
        return values, {model: "Invalid results" for model in MODEL_NAMES}, issues

    occurrences: Counter[tuple[str, str]] = Counter()
    parsed_values: dict[tuple[str, str], Optional[float]] = {}
    for row in results.loc[:, ["model", "metric", "value"]].itertuples(index=False):
        model = _MODEL_IDS.get(_identifier(row.model))
        if model is None:
            issues.append(f"Unrecognized model identifier: {row.model!r}.")
            continue

        supplied_models.add(model)
        metric = _METRIC_IDS.get(_identifier(row.metric))
        if metric is None:
            issues.append(f"Unrecognized metric identifier for {model}: {row.metric!r}.")
            continue

        key = (model, metric)
        occurrences[key] += 1
        try:
            numeric_value = float(row.value)
        except (TypeError, ValueError):
            numeric_value = math.nan

        parsed_values[key] = (
            numeric_value
            if math.isfinite(numeric_value) and 0.0 <= numeric_value <= 1.0
            else None
        )

    for (model, metric), count in occurrences.items():
        if count > 1:
            parsed_values[(model, metric)] = None
            issues.append(f"Duplicate evaluation rows for {model} / {metric}; value marked N/A.")
        elif parsed_values[(model, metric)] is None:
            issues.append(
                f"Invalid or missing value for {model} / {metric}; expected a finite score from 0 to 1."
            )
        else:
            values[model][metric] = parsed_values[(model, metric)]

    statuses: dict[str, str] = {}
    for model in MODEL_NAMES:
        available_count = sum(value is not None for value in values[model].values())
        if model not in supplied_models:
            statuses[model] = "Not available"
        elif available_count == len(MODEL_METRICS):
            statuses[model] = "Available"
        elif available_count:
            statuses[model] = f"Partially available ({available_count}/{len(MODEL_METRICS)} metrics)"
        else:
            statuses[model] = "Invalid or incomplete results"

    return values, statuses, issues


def get_model_evaluation_results() -> Optional[pd.DataFrame]:
    """Return canonical evaluation rows, or ``None`` when none are available.

    Available results use one row per model/metric pair and ``model``,
    ``metric``, and ``value`` columns. Unsupported response types raise
    :class:`MalformedProviderResponseError`; missing results are not fabricated.
    """
    results = get_data_provider().get_model_evaluation_results()
    if results is not None and not isinstance(results, pd.DataFrame):
        raise MalformedProviderResponseError(
            "The model-evaluation provider must return a pandas DataFrame or None."
        )
    return results.copy() if results is not None else None
