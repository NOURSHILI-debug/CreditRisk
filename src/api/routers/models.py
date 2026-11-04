from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_state
from api.schemas import ModelInfoResponse
from api.state import ApplicationState, threshold_payload

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelInfoResponse)
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


@router.get("/model/evaluation")
def model_evaluation(
    state: ApplicationState = Depends(get_state),
) -> dict[str, Any]:
    if not state.model_loaded:
        raise HTTPException(
            status_code=503, detail="Model artifacts are unavailable."
        )
    if state.evaluation_missing:
        raise HTTPException(
            status_code=404, detail="Evaluation artifact is not available."
        )
    if state.evaluation_error or state.evaluation_results is None:
        raise HTTPException(
            status_code=503, detail="Evaluation artifact is unavailable."
        )
    return state.evaluation_results


@router.get("/model/global-importance")
def global_importance(
    state: ApplicationState = Depends(get_state),
) -> dict[str, Any]:
    if state.global_importance_missing:
        raise HTTPException(
            status_code=404, detail="Global importance artifact is not available."
        )
    if state.global_importance_error or state.global_importance is None:
        raise HTTPException(
            status_code=503, detail="Global importance artifact is unavailable."
        )
    return state.global_importance
