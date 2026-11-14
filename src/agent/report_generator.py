"""Deterministic, source-grounded aggregate credit-risk reports."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from dashboard import data_access

SUPPORTED_PERIODS = ("7d", "30d", "90d", "all")
_EVALUATION_METRICS = ("ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-score")


def _unavailable_evaluation() -> dict[str, Any]:
    return {
        "status": "unavailable",
        "models": {
            model: {metric: None for metric in _EVALUATION_METRICS}
            for model in ("XGBoost", "LSTM")
        },
        "statuses": {"XGBoost": "Not available", "LSTM": "Not available"},
        "evaluations": [],
        "evaluation_metadata": None,
        "comparison_available": False,
        "comparison_note": "Evaluation results are unavailable.",
    }


def _retrieve(call: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = call()
    except Exception as error:
        return None, f"{type(error).__name__}: data source retrieval failed."
    if not isinstance(value, dict):
        return None, "Malformed source response: expected an object."
    return value, None


def _application_section(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        return {"status": "unavailable", "total_applications": None}
    raw = metrics.get("total_applications")
    count = raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else None
    return {
        "status": "available" if count is not None else "unavailable",
        "total_applications": count,
        "target_counts": None,
        "target_proportions": None,
        "risk_related_distributions": None,
        "note": "Application API exposes an allowlisted record count; TARGET is not exposed by applicant browsing.",
    }


def _prediction_section(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        return {"status": "unavailable", "prediction_count": None, "average_probability": None}
    count = metrics.get("prediction_count")
    count = count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None
    summary = {
        "status": "available" if count is not None else "unavailable",
        "prediction_count": count,
        "risk_distribution": metrics.get("risk_distribution") if isinstance(metrics.get("risk_distribution"), Mapping) else None,
        "predicted_outcomes": metrics.get("predicted_outcomes") if isinstance(metrics.get("predicted_outcomes"), Mapping) else None,
        "average_probability": None,
        "source": metrics.get("source") if isinstance(metrics.get("source"), str) else None,
        "dataset_split": metrics.get("prediction_split") if isinstance(metrics.get("prediction_split"), str) else None,
        "provenance": "saved_historical_split_predictions",
        "note": "The API provides aggregate saved split predictions, not persistent per-application prediction history or mean probability.",
    }
    return summary


def _evaluation_section(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return _unavailable_evaluation()
    evaluations = result.get("evaluations")
    metadata = result.get("evaluation_metadata")
    return {
        "status": result.get("status", "unavailable"),
        "models": result.get("models") if isinstance(result.get("models"), Mapping) else _unavailable_evaluation()["models"],
        "statuses": result.get("statuses") if isinstance(result.get("statuses"), Mapping) else _unavailable_evaluation()["statuses"],
        "evaluations": evaluations if isinstance(evaluations, list) else [],
        "evaluation_metadata": metadata if isinstance(metadata, Mapping) else None,
        "comparison_available": result.get("comparison_available") is True,
        "comparison_note": result.get("comparison_note") if isinstance(result.get("comparison_note"), str) else "Comparable evaluation results are unavailable.",
    }


def _get_model_comparison() -> dict[str, Any]:
    # Reuse the existing wrapper's strict metric and protocol comparability
    # checks while avoiding an import cycle during module initialization.
    from agent.tools import get_model_comparison

    return get_model_comparison()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _period_window(period: str, reference_time: datetime) -> tuple[datetime | None, datetime | None]:
    """Return an inclusive UTC window; None bounds represent the all-data scope."""
    end = (
        reference_time.replace(tzinfo=timezone.utc)
        if reference_time.tzinfo is None
        else reference_time.astimezone(timezone.utc)
    )
    if period == "all":
        return None, None
    return end - timedelta(days=int(period[:-1])), end


def _coverage_section(
    *,
    timestamp_field: str | None,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "records_included": None,
        "timestamp_field": timestamp_field,
        "recorded_at": recorded_at,
        "reason": (
            "The source provides an aggregate or snapshot without timestamped records "
            "and a period filter; the requested period was not applied."
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    metadata = report["metadata"]
    apps = report["application_statistics"]
    predictions = report["prediction_statistics"]
    evaluation = report["model_evaluation"]
    lines = [
        f"# {metadata['title']}",
        f"Requested period: **{metadata['requested_period']}**. Actual coverage: {metadata['actual_coverage']}.",
        f"Status: **{report['status']}**.",
        "",
        "## Executive summary",
        f"- Applications: {apps.get('total_applications') if apps.get('total_applications') is not None else 'Unavailable'}",
        f"- Saved predictions: {predictions.get('prediction_count') if predictions.get('prediction_count') is not None else 'Unavailable'}",
        f"- Model evaluation: {evaluation.get('status')}",
        "",
        "## Application statistics",
        f"- Total applications: {apps.get('total_applications') if apps.get('total_applications') is not None else 'Unavailable'}",
        "- TARGET class counts and proportions: unavailable through the current applicant API.",
        "",
        "## Prediction statistics",
        f"- Saved prediction count: {predictions.get('prediction_count') if predictions.get('prediction_count') is not None else 'Unavailable'}",
        f"- Dataset split/source: {predictions.get('dataset_split') or 'Unavailable'} / {predictions.get('source') or 'Unavailable'}",
        f"- Risk distribution: {predictions.get('risk_distribution') or 'Unavailable'}",
        "- Average predicted probability: unavailable from aggregate source.",
        "",
        "## Model evaluation",
    ]
    if metadata.get("source_mode") == "mock":
        lines.insert(
            3,
            "**Synthetic mock report:** application records and predictions are generated examples. No trained-model evaluation results are available.",
        )
    for model in ("XGBoost", "LSTM"):
        status = evaluation.get("statuses", {}).get(model, "Not available")
        lines.append(f"- {model} ({status}): {evaluation.get('models', {}).get(model)}")
    lines.extend([
        f"- Direct comparison available: {evaluation.get('comparison_available')}",
        f"- {evaluation.get('comparison_note')}",
        "",
        "## Data quality and limitations",
    ])
    coverage = metadata.get("period_coverage", {})
    lines.extend(
        [
            "",
            "## Period coverage",
            f"- Coverage status: {coverage.get('status', 'unavailable')}",
            f"- UTC window: {coverage.get('window_start') or 'not bounded'} to {coverage.get('window_end') or 'not bounded'}",
        ]
    )
    for section, details in coverage.get("sections", {}).items():
        if isinstance(details, Mapping):
            lines.append(
                f"- {section}: {details.get('status')}; records included: "
                f"{details.get('records_included') if details.get('records_included') is not None else 'unknown'}; "
                f"{details.get('reason', 'No coverage detail available.')}"
            )
    quality = report["data_quality"]
    lines.extend(f"- {item}" for item in quality["limitations"])
    lines.extend(f"- Missing source: {item}" for item in quality["missing_data"])
    return "\n".join(lines)


def generate_report(period: str) -> dict[str, Any]:
    """Generate a report for an accepted scope, explicitly recording actual coverage."""
    if not isinstance(period, str) or period not in SUPPORTED_PERIODS:
        raise ValueError(f"period must be one of: {', '.join(SUPPORTED_PERIODS)}.")

    generated_at = _utc_now()
    window_start, window_end = _period_window(period, generated_at)
    application_result, application_error = _retrieve(_application_count_only)
    prediction_result, prediction_error = _retrieve(data_access.get_prediction_summary)
    comparison_result, evaluation_error = _retrieve(_get_model_comparison)

    application_payload = application_result
    prediction_payload = prediction_result
    evaluation_payload = _evaluation_section(comparison_result)
    application_section = _application_section(application_payload)
    prediction_section = _prediction_section(prediction_payload)
    missing = []
    if application_payload is None:
        missing.append("application statistics")
    if prediction_payload is None:
        missing.append("saved prediction summary")
    if comparison_result is None or evaluation_payload["status"] in {"unavailable", "error", "invalid"}:
        missing.append("model evaluation")
    evaluation_metadata = evaluation_payload.get("evaluation_metadata")
    evaluation_recorded_at = (
        evaluation_metadata.get("generated_at")
        if isinstance(evaluation_metadata, Mapping)
        else None
    )
    if isinstance(evaluation_recorded_at, str):
        try:
            parsed_timestamp = datetime.fromisoformat(
                evaluation_recorded_at.replace("Z", "+00:00")
            )
            if parsed_timestamp.tzinfo is None:
                parsed_timestamp = parsed_timestamp.replace(tzinfo=timezone.utc)
            evaluation_recorded_at = parsed_timestamp.astimezone(timezone.utc).isoformat()
        except ValueError:
            evaluation_recorded_at = None
    else:
        evaluation_recorded_at = None
    period_coverage = {
        "status": "partial",
        "timezone": "UTC",
        "window_start": window_start.isoformat() if window_start else None,
        "window_end": window_end.isoformat() if window_end else None,
        "sections": {
            "application_statistics": _coverage_section(timestamp_field=None),
            "prediction_statistics": _coverage_section(timestamp_field=None),
            "model_evaluation": _coverage_section(
                timestamp_field="generated_at",
                recorded_at=evaluation_recorded_at,
            ),
        },
    }
    limitations = [
        "The requested period was not applied: the application snapshot and saved prediction summary contain no timestamped records or period filters.",
        "An evaluation generated_at timestamp describes artifact time only; it does not date-filter aggregate metrics by prediction or application date.",
        "Actual coverage is limited to the available application snapshot and named historical split; complete lifetime coverage is not established.",
        "TARGET counts and proportions are unavailable because the application API field allowlist excludes TARGET.",
        "Average predicted probability is unavailable because the aggregate prediction API does not return probabilities.",
        "Fresh per-application inference is not a saved historical prediction and is not included in these statistics.",
    ]
    if application_error:
        limitations.append(f"Application source issue: {application_error}")
    if prediction_error:
        limitations.append(f"Prediction source issue: {prediction_error}")
    if evaluation_error:
        limitations.append(f"Evaluation source issue: {evaluation_error}")
    evaluation_available = (
        comparison_result is not None
        and evaluation_payload["status"] not in {"unavailable", "error", "invalid"}
    )
    sources = [
        name
        for name, available in (
            ("application API", application_payload is not None),
            ("saved prediction summary", prediction_payload is not None),
            ("model evaluation API", evaluation_available),
        )
        if available
    ]
    available_count = sum(
        (application_payload is not None, prediction_payload is not None, evaluation_available)
    )
    # Source retrieval can be successful while temporal coverage is unsupported.
    status = "partial" if available_count else "unavailable"
    report: dict[str, Any] = {
        "status": status,
        "source_mode": data_access.get_provider_name(),
        "metadata": {
            "report_type": "credit_risk",
            "title": "Credit Risk Intelligence Report",
            "requested_period": period,
            "generated_at": generated_at.isoformat(),
            "actual_coverage": (
                "Deterministic synthetic applicant sample and illustrative predictions; no trained evaluation results or real historical coverage are available."
                if data_access.is_mock_mode()
                else "Current API application snapshot plus named historical prediction/evaluation split(s); no date filtering was applied because record timestamps and period filters are unavailable."
            ),
            "period_filtering_supported": False,
            "period_coverage": period_coverage,
            "sources": sources,
            "source_mode": data_access.get_provider_name(),
        },
        "executive_summary": {
            "application_count": application_section["total_applications"],
            "saved_prediction_count": prediction_section["prediction_count"],
            "evaluation_status": evaluation_payload["status"],
            "limitations": limitations,
        },
        "application_statistics": application_section,
        "prediction_statistics": prediction_section,
        "model_evaluation": evaluation_payload,
        "model_comparison": {
            "available": evaluation_payload["comparison_available"],
            "note": evaluation_payload["comparison_note"],
        },
        "data_quality": {"missing_data": missing, "limitations": limitations, "warnings": []},
    }
    report["markdown"] = _markdown(report)
    return report


def _application_count_only() -> dict[str, Any]:
    _, total = data_access.get_applications_page(
        limit=1, offset=0, fields=[data_access.APPLICATION_ID_COLUMN]
    )
    return {"total_applications": total}
