from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_state
from api.schemas import HealthResponse
from api.state import ApplicationState

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: ApplicationState = Depends(get_state)) -> HealthResponse:
    ready = state.model_loaded and state.data_available
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=state.model_loaded,
        data_available=state.data_available,
        model_name=state.model_name,
        model_version=state.model_version,
        model_hash=state.model_hash,
    )
