from __future__ import annotations

import math
from numbers import Real
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from api.dependencies import get_state
from api.errors import error_payload
from api.schemas import (
    ErrorResponse,
    EvaluationResponse,
    ModelInfoResponse,
    PredictionSummaryResponse,
)
from api.state import ApplicationState, threshold_payload
from inference.predictor import RISK_BANDS

router = APIRouter(tags=["models"])

EVALUATION_METRICS: tuple[tuple[str, str], ...] = (
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1-score"),
)


@router.get(
    "/models",
    response_model=ModelInfoResponse,
    responses={503: {"model": ErrorResponse, "description": "Model unavailable."}},
)
def model_info(
    state: ApplicationState = Depends(get_state),
) -> ModelInfoResponse:
    if not state.model_loaded:
        raise HTTPException(
            status_code=503, detail="Model artifacts are unavailable."
        )
    return ModelInfoResponse(
        model_name=state.model_name or "XGBoost",
        model_version=state.model_version or "unknown",
        model_hash=state.model_hash or "",
        training_date=state.training_date,
        library_versions=state.library_versions,
        thresholds=threshold_payload(state),
    )


def _evaluation_error(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=error_payload("model_unavailable", message),
    )


@router.get(
    "/model/evaluation",
    response_model=EvaluationResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid evaluation split."},
        503: {"model": ErrorResponse, "description": "Evaluation unavailable."},
    },
)
def model_evaluation(
    split: Literal["validation", "test"] = Query(default="test"),
    state: ApplicationState = Depends(get_state),
) -> EvaluationResponse | JSONResponse:
    if not state.model_loaded:
        return _evaluation_error("Model artifacts are unavailable.")
    if state.evaluation_missing:
        return _evaluation_error("Evaluation artifact is not available.")
    if state.evaluation_error or state.evaluation_results is None:
        return _evaluation_error("Evaluation artifact is unavailable.")

    metrics_by_split = state.evaluation_results.get("metrics")
    if not isinstance(metrics_by_split, dict):
        return _evaluation_error("Evaluation artifact has no metrics.")
    metrics = metrics_by_split.get(split)
    if not isinstance(metrics, dict):
        return _evaluation_error(
            f"Evaluation results for split {split!r} are not available."
        )

    rows = []
    for key, label in EVALUATION_METRICS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            continue
        rows.append(
            {"model": "XGBoost", "metric": label, "value": numeric}
        )

    artifact = state.evaluation_results
    provenance: dict[str, Any] = {
        "artifact_version": artifact.get("artifact_version"),
        "model_name": artifact.get("model_name") or state.model_name,
        "model_version": artifact.get("model_version") or state.model_version,
        "model_hash": artifact.get("model_hash") or state.model_hash,
        "generated_at": artifact.get("generated_at"),
        "training_date": artifact.get("training_date") or state.training_date,
        "threshold": artifact.get("threshold"),
        "split": split,
        "split_protocol": artifact.get("split_protocol"),
        "n_samples": metrics.get("n_samples"),
        "class_counts": metrics.get("class_counts"),
        "source": "evaluation_results.json",
        "lstm_available": False,
        "target": artifact.get("target"),
        "metric_definitions": artifact.get("metric_definitions"),
    }
    return EvaluationResponse(
        split=split,
        rows=rows,
        provenance=provenance,
        confusion_matrix=metrics.get("confusion_matrix"),
    )


@router.get(
    "/model/prediction-summary",
    response_model=PredictionSummaryResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid prediction split."},
        503: {"model": ErrorResponse, "description": "Prediction records unavailable."},
    },
)
def prediction_summary(
    split: Literal["validation", "test"] = Query(default="test"),
    state: ApplicationState = Depends(get_state),
) -> PredictionSummaryResponse | JSONResponse:
    if not state.model_loaded:
        return _evaluation_error("Model artifacts are unavailable.")
    records = state.prediction_records.get(split)
    if records is None:
        return _evaluation_error(
            state.prediction_record_errors.get(
                split, "Prediction records are unavailable."
            )
        )

    probabilities = records["probability"]
    risk_distribution = {
        "LOW": int(probabilities.lt(RISK_BANDS["low_upper"]).sum()),
        "MEDIUM": int(
            (
                probabilities.ge(RISK_BANDS["low_upper"])
                & probabilities.lt(RISK_BANDS["medium_upper"])
            ).sum()
        ),
        "HIGH": int(probabilities.ge(RISK_BANDS["medium_upper"]).sum()),
    }
    return PredictionSummaryResponse(
        split=split,
        prediction_count=int(len(records)),
        predicted_outcomes={
            "no_defined_repayment_difficulty": int(
                records["prediction"].eq(0).sum()
            ),
            "repayment_difficulty": int(records["prediction"].eq(1).sum()),
        },
        risk_distribution=risk_distribution,
        risk_bands={
            "low_upper": float(RISK_BANDS["low_upper"]),
            "medium_upper": float(RISK_BANDS["medium_upper"]),
        },
        decision_threshold=state.decision_threshold,
        source=f"{split}_predictions.csv",
    )


@router.get(
    "/model/global-importance",
    responses={
        404: {"model": ErrorResponse, "description": "Importance artifact not found."},
        503: {"model": ErrorResponse, "description": "Importance artifact unavailable."},
    },
)
def global_importance(
    state: ApplicationState = Depends(get_state),
) -> dict[str, Any]:
    if state.global_importance_missing:
        raise HTTPException(
            status_code=404,
            detail="Global importance artifact is not available.",
        )
    if state.global_importance_error or state.global_importance is None:
        raise HTTPException(
            status_code=503,
            detail="Global importance artifact is unavailable.",
        )
    return state.global_importance
