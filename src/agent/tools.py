from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from dashboard import data_access
from agent.report_generator import generate_report as _generate_report

LOGGER = logging.getLogger(__name__)
_NON_APPLICATION_OUTPUT_FIELDS = {"RISK_PROBABILITY", "RISK_CATEGORY"}
_PREDICTION_FAILURE_MESSAGES = {
    "invalid": "The prediction request or response was invalid.",
    "not_found": "No application was found for this ID.",
    "no_prediction": "No prediction is available for this application.",
    "model_unavailable": "The prediction model is unavailable.",
    "unavailable": "The prediction service is unavailable.",
    "error": "Prediction execution failed.",
}


def generate_report(period: str) -> dict[str, Any]:
    """Build an aggregate report through the shared dashboard data layer."""
    try:
        return _generate_report(period)
    except ValueError as error:
        return {"status": "invalid", "message": str(error)}
    except Exception as error:
        LOGGER.warning("agent_report_generation_failed error_type=%s", type(error).__name__)
        return {"status": "error", "message": "The report could not be generated safely."}


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


def _empty_model_comparison_result(
    status: str, message: str, source_mode: str
) -> dict[str, Any]:
    evaluation_status = "invalid" if status == "invalid" else "unavailable"
    model_status = "Invalid results" if status == "invalid" else "Not available"
    metrics = {
        model: {metric: None for metric in data_access.MODEL_METRICS}
        for model in ("XGBoost", "LSTM")
    }
    return {
        "status": status,
        "models": metrics,
        "statuses": {model: model_status for model in metrics},
        "issues": [],
        "evaluations": [
            {
                "model_name": model,
                "evaluation_status": evaluation_status,
                "dataset_split": None,
                "metrics": None,
                "model_version": None,
                "unavailable_reason": message,
            }
            for model in metrics
        ],
        "evaluation_metadata": None,
        "comparison_available": False,
        "comparison_note": message,
        "source_mode": source_mode,
        "message": message,
    }


def get_model_comparison() -> dict[str, Any]:
    """Retrieve actual evaluation rows and report whether models are comparable."""
    source_mode = _source_mode()
    try:
        results = data_access.get_model_evaluation_results()
        values, statuses, issues = data_access.normalize_model_evaluation_results(results)
    except data_access.DataSourceUnavailableError as error:
        return _empty_model_comparison_result(
            "unavailable",
            "Evaluation results are unavailable from the API.",
            source_mode,
        )
    except data_access.MalformedProviderResponseError as error:
        return _empty_model_comparison_result(
            "invalid",
            "The API returned malformed evaluation results.",
            source_mode,
        )
    except data_access.DataAccessError as error:
        LOGGER.warning(
            "agent_model_comparison_failed error_type=%s",
            type(error).__name__,
        )
        return _empty_model_comparison_result(
            "error",
            "Evaluation results could not be retrieved.",
            source_mode,
        )
    except Exception as error:
        LOGGER.warning(
            "agent_model_comparison_failed error_type=%s",
            type(error).__name__,
        )
        return _empty_model_comparison_result(
            "error",
            "Evaluation results could not be retrieved.",
            source_mode,
        )

    model_metrics = {
        model: _json_value(metrics)
        for model, metrics in values.items()
    }
    serialized_issues = _json_value(issues)
    attributes = results.attrs if isinstance(results, pd.DataFrame) else {}
    provenance = attributes.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    split = attributes.get("split")
    if not isinstance(split, str) or not split.strip():
        split = None

    raw_target = provenance.get("target")
    target: dict[str, Any] | None = None
    if isinstance(raw_target, Mapping):
        target = {
            key: raw_target.get(key)
            for key in ("name", "positive_class", "positive_label", "probability_semantics")
        }
    raw_definitions = provenance.get("metric_definitions")
    metric_definitions = (
        {
            key: raw_definitions.get(key)
            for key in ("roc_auc", "pr_auc", "precision", "recall", "f1")
        }
        if isinstance(raw_definitions, Mapping)
        else None
    )
    threshold = provenance.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        threshold = None
    else:
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            threshold = None
    raw_sample_count = provenance.get("n_samples")
    sample_count = (
        int(raw_sample_count)
        if isinstance(raw_sample_count, Integral)
        and not isinstance(raw_sample_count, bool)
        and raw_sample_count > 0
        else None
    )
    raw_class_counts = provenance.get("class_counts")
    class_counts = None
    if isinstance(raw_class_counts, Mapping):
        valid_counts = {
            str(label): int(count)
            for label, count in raw_class_counts.items()
            if isinstance(count, Integral)
            and not isinstance(count, bool)
            and count >= 0
        }
        if len(valid_counts) == len(raw_class_counts):
            class_counts = valid_counts
    split_protocol = provenance.get("split_protocol")
    if not isinstance(split_protocol, Mapping):
        split_protocol = None
    evaluation_metadata = {
        "dataset_split": split,
        "source": provenance.get("source")
        if isinstance(provenance.get("source"), str)
        else None,
        "generated_at": provenance.get("generated_at")
        if isinstance(provenance.get("generated_at"), str)
        else None,
        "threshold": threshold,
        "n_samples": sample_count,
        "class_counts": class_counts,
        "target": target,
        "metric_definitions": metric_definitions,
        "split_protocol": _json_value(split_protocol)
        if split_protocol is not None
        else None,
    }

    evaluations: list[dict[str, Any]] = []
    for model in ("XGBoost", "LSTM"):
        normalized_status = statuses[model]
        if normalized_status == "Available":
            evaluation_status = "available"
        elif normalized_status.startswith("Partially available"):
            evaluation_status = "partial"
        elif normalized_status == "Not available":
            evaluation_status = "unavailable"
        else:
            evaluation_status = "invalid"
        has_values = any(value is not None for value in values[model].values())
        model_provenance = (
            provenance.get("model_name") == model
            and isinstance(provenance.get("model_version"), str)
        )
        evaluations.append(
            {
                "model_name": model,
                "evaluation_status": evaluation_status,
                "dataset_split": split,
                "metrics": model_metrics[model] if has_values else None,
                "model_version": provenance.get("model_version")
                if model_provenance
                else None,
                "unavailable_reason": (
                    (
                        "The API reports that LSTM evaluation results are unavailable."
                        if model == "LSTM"
                        and provenance.get("lstm_available") is False
                        else "Evaluation results are not available in the API."
                    )
                    if evaluation_status == "unavailable"
                    else None
                ),
            }
        )

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

    comparison_requirements = (
        statuses["XGBoost"] == "Available"
        and statuses["LSTM"] == "Available"
        and split is not None
        and isinstance(target, dict)
        and target.get("name") == "TARGET"
        and type(target.get("positive_class")) is int
        and target.get("positive_class") == 1
        and target.get("positive_label") == "repayment difficulty"
        and target.get("probability_semantics") == "P(TARGET=1)"
        and isinstance(metric_definitions, dict)
        and all(
            isinstance(metric_definitions.get(key), str)
            and metric_definitions[key].strip()
            for key in ("roc_auc", "pr_auc", "precision", "recall", "f1")
        )
        and threshold is not None
        and sample_count is not None
        and isinstance(class_counts, dict)
        and {"0", "1"}.issubset(class_counts)
        and sum(class_counts.values()) == sample_count
        and isinstance(split_protocol, Mapping)
        and bool(split_protocol)
    )
    comparison_available = bool(comparison_requirements)
    if comparison_available:
        comparison_note = (
            "Both models have complete evaluation results under the same "
            "reported split, population, target convention, metric definitions, "
            "and threshold."
        )
    elif statuses["XGBoost"] == "Not available" and statuses["LSTM"] == "Not available":
        comparison_note = "No model evaluation results are available."
    elif statuses["LSTM"] == "Not available":
        comparison_note = (
            "Only XGBoost evaluation results are available; LSTM evaluation "
            "results are not provided by the API."
        )
    elif statuses["XGBoost"] == "Not available":
        comparison_note = (
            "Only LSTM evaluation results are available; XGBoost evaluation "
            "results are not provided by the API."
        )
    elif statuses["XGBoost"] != "Available" or statuses["LSTM"] != "Available":
        comparison_note = (
            "A direct comparison requires complete metrics for both models."
        )
    else:
        comparison_note = (
            "Evaluation protocol metadata is incomplete, so direct comparability "
            "cannot be confirmed."
        )

    response = {
        "status": status,
        "models": model_metrics,
        "statuses": _json_value(statuses),
        "issues": serialized_issues,
        "evaluations": evaluations,
        "evaluation_metadata": evaluation_metadata,
        "comparison_available": comparison_available,
        "comparison_note": comparison_note,
        "source_mode": source_mode,
    }
    if message is not None:
        response["message"] = message
    return response


def get_prediction(application_id: int | str) -> dict[str, Any]:
    """Request a prediction through the configured API without an explanation."""
    source_mode = _source_mode()
    parsed_id = data_access.parse_application_id(application_id)
    response: dict[str, Any] = {
        "status": "invalid",
        "application_id": _application_id_for_output(application_id),
        "probability": None,
        "risk_level": None,
        "model_name": None,
        "model_version": None,
        "prediction_timestamp": None,
        "message": "A valid positive integer application ID is required.",
        "source_mode": source_mode,
    }
    if parsed_id is None:
        return response

    response["application_id"] = parsed_id
    response["message"] = None
    try:
        prediction = data_access.get_credit_risk_prediction(
            parsed_id, include_explanation=False
        )
    except data_access.DataSourceUnavailableError:
        response.update(
            status="unavailable",
            message="Prediction service is unavailable.",
        )
        return response
    except data_access.MalformedProviderResponseError:
        response.update(
            status="invalid",
            message="The prediction service returned an invalid response.",
        )
        return response
    except data_access.DataAccessError:
        response.update(
            status="error",
            message="Prediction retrieval failed.",
        )
        return response

    try:
        response.update(
            _json_value(
                {
                    "status": prediction.status,
                    "probability": prediction.probability,
                    "risk_level": prediction.risk_level,
                    "model_name": prediction.model_name,
                    "model_version": prediction.model_version,
                    "prediction_timestamp": prediction.prediction_timestamp,
                    "message": prediction.message,
                }
            )
        )
    except (TypeError, ValueError):
        response.update(
            status="invalid",
            probability=None,
            risk_level=None,
            model_name=None,
            model_version=None,
            prediction_timestamp=None,
            message="The prediction service returned an invalid response.",
        )
    return response


def explain_prediction(application_id: int | str) -> dict[str, Any]:
    """Request and minimize the actual SHAP explanation for one application."""
    parsed_id = data_access.parse_application_id(application_id)
    source_mode = _source_mode()
    result: dict[str, Any] = {
        "status": "invalid",
        "application_id": _application_id_for_output(application_id),
        "prediction_probability": None,
        "risk_level": None,
        "model_name": None,
        "model_version": None,
        "explanation_method": None,
        "output_space": None,
        "base_value": None,
        "base_probability": None,
        "prediction_margin": None,
        "contribution_sum": None,
        "top_features": [],
        "message": "A valid positive integer application ID is required.",
        "source_mode": source_mode,
    }
    if parsed_id is None:
        return result
    result["application_id"] = parsed_id

    def invalid_response() -> dict[str, Any]:
        result.update(
            status="invalid",
            prediction_probability=None,
            risk_level=None,
            model_name=None,
            model_version=None,
            explanation_method=None,
            output_space=None,
            base_value=None,
            base_probability=None,
            prediction_margin=None,
            contribution_sum=None,
            top_features=[],
            message="The prediction explanation response was invalid.",
        )
        return result

    try:
        prediction = data_access.get_credit_risk_prediction(
            parsed_id, include_explanation=True
        )
    except data_access.DataSourceUnavailableError:
        result.update(
            status="unavailable",
            message="The prediction service is unavailable.",
        )
        return result
    except data_access.MalformedProviderResponseError:
        result.update(
            status="invalid",
            message="The prediction explanation response was invalid.",
        )
        return result
    except data_access.DataAccessError:
        result.update(status="error", message="Prediction explanation failed.")
        return result
    except Exception as error:
        LOGGER.warning(
            "agent_prediction_explanation_failed error_type=%s",
            type(error).__name__,
        )
        result.update(status="error", message="Prediction explanation failed.")
        return result

    if prediction.status != "available":
        if prediction.status not in _PREDICTION_FAILURE_MESSAGES:
            return invalid_response()
        result.update(
            status=prediction.status,
            message=_PREDICTION_FAILURE_MESSAGES[prediction.status],
        )
        if prediction.model_name is not None and not isinstance(
            prediction.model_name, str
        ):
            return invalid_response()
        if prediction.model_version is not None and not isinstance(
            prediction.model_version, str
        ):
            return invalid_response()
        result["model_name"] = prediction.model_name
        result["model_version"] = prediction.model_version
        return result

    probability = prediction.probability
    if (
        isinstance(probability, bool)
        or not isinstance(probability, Real)
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        return invalid_response()
    if prediction.risk_level not in (None, "LOW", "MEDIUM", "HIGH"):
        return invalid_response()
    if prediction.model_name is not None and not isinstance(
        prediction.model_name, str
    ):
        return invalid_response()
    if prediction.model_version is not None and not isinstance(
        prediction.model_version, str
    ):
        return invalid_response()

    result.update(
        prediction_probability=float(probability),
        risk_level=prediction.risk_level,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
    )
    explanation = prediction.explanation
    if explanation is None:
        result.update(
            status="explanation_unavailable",
            message="The prediction is available, but its SHAP explanation is unavailable.",
        )
        return result
    if not isinstance(explanation, Mapping):
        return invalid_response()

    method = explanation.get("method")
    output_space = explanation.get("output_space")
    top_features = explanation.get("top_features")
    if (
        method != "tree_shap"
        or output_space != "log_odds"
        or not isinstance(top_features, list)
    ):
        return invalid_response()

    numeric_fields = (
        "base_value",
        "base_probability",
        "prediction_margin",
        "contribution_sum",
    )
    numeric_values: dict[str, float | None] = {}
    for field in numeric_fields:
        value = explanation.get(field)
        if value is None:
            numeric_values[field] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            return invalid_response()
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return invalid_response()
        if field == "base_probability" and not 0.0 <= numeric_value <= 1.0:
            return invalid_response()
        numeric_values[field] = numeric_value

    explanation_probability = explanation.get("probability")
    if explanation_probability is not None:
        if (
            isinstance(explanation_probability, bool)
            or not isinstance(explanation_probability, Real)
        ):
            return invalid_response()
        explanation_probability = float(explanation_probability)
        if (
            not math.isfinite(explanation_probability)
            or not 0.0 <= explanation_probability <= 1.0
            or not math.isclose(
                explanation_probability, float(probability), rel_tol=0.0, abs_tol=1e-12
            )
        ):
            return invalid_response()

    base_value = numeric_values["base_value"]
    prediction_margin = numeric_values["prediction_margin"]
    contribution_sum = numeric_values["contribution_sum"]
    if (
        base_value is not None
        and prediction_margin is not None
        and contribution_sum is not None
        and not math.isclose(
            base_value + contribution_sum,
            prediction_margin,
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
    ):
        return invalid_response()

    sanitized_features: list[dict[str, Any]] = []
    for feature in top_features:
        if not isinstance(feature, Mapping):
            return invalid_response()
        name = feature.get("feature")
        shap_value = feature.get("shap_value")
        effect = feature.get("effect")
        if (
            not isinstance(name, str)
            or not name.strip()
            or name.strip().upper() in {"TARGET", "SK_ID_CURR"}
            or isinstance(shap_value, bool)
            or not isinstance(shap_value, Real)
        ):
            return invalid_response()
        contribution = float(shap_value)
        if not math.isfinite(contribution):
            return invalid_response()
        expected_effect = (
            "increases_risk"
            if contribution > 0
            else "decreases_risk"
            if contribution < 0
            else "neutral"
        )
        if effect != expected_effect:
            return invalid_response()
        sanitized_features.append(
            {
                "feature": name,
                "shap_value": contribution,
                "effect": effect,
            }
        )

    result.update(
        status="available",
        explanation_method=method,
        output_space=output_space,
        **numeric_values,
        top_features=sanitized_features,
        message="Explanation retrieved successfully.",
    )
    return result
