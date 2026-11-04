from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import JSONResponse

from api.dependencies import get_state
from api.schemas import PredictionResponse, PredictRequest
from api.service import (
    ApplicationNotFoundError,
    InvalidPredictionInputError,
    PredictionService,
    ServiceUnavailableError,
)
from api.state import ApplicationState

router = APIRouter(tags=["predictions"])


def _error(
    state: ApplicationState,
    status_code: int,
    status: str,
    message: str,
    application_id: int | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=state.prediction_error(status, message, application_id),
    )


def _run(
    state: ApplicationState,
    operation: str,
    application_id: int | None,
    include_explanation: bool,
    *,
    features: dict | None = None,
) -> PredictionResponse | JSONResponse:
    service = PredictionService(state)
    try:
        if operation == "id":
            assert application_id is not None
            result = service.predict_by_application_id(
                application_id, include_explanation
            )
        else:
            assert features is not None
            result = service.predict_by_features(
                features, include_explanation
            )
        return PredictionResponse.model_validate(result)
    except ApplicationNotFoundError as exc:
        return _error(state, 404, "error", str(exc), application_id)
    except InvalidPredictionInputError as exc:
        return _error(state, 422, "error", str(exc), application_id)
    except ServiceUnavailableError as exc:
        return _error(
            state, 503, "unavailable", str(exc), application_id
        )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictRequest,
    include_explanation: bool = Query(default=False),
    state: ApplicationState = Depends(get_state),
) -> PredictionResponse | JSONResponse:
    if payload.application_id is not None:
        return _run(
            state,
            "id",
            payload.application_id,
            include_explanation,
        )
    return _run(
        state,
        "features",
        None,
        include_explanation,
        features=payload.features,
    )


@router.get(
    "/applications/{application_id}/prediction",
    response_model=PredictionResponse,
)
def application_prediction(
    application_id: int = Path(gt=0),
    include_explanation: bool = Query(default=False),
    state: ApplicationState = Depends(get_state),
) -> PredictionResponse | JSONResponse:
    return _run(
        state,
        "id",
        application_id,
        include_explanation,
    )
