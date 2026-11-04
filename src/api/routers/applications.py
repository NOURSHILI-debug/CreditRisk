from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import JSONResponse

from api.dependencies import get_state
from api.errors import error_payload
from api.fields import (
    DEFAULT_APPLICATION_LIMIT,
    FILTERABLE_APPLICATION_FIELDS,
    MAX_APPLICATION_LIMIT,
    parse_fields,
)
from api.repository import DuplicateApplicationIdError
from api.schemas import (
    ApplicationDetailResponse,
    ApplicationListResponse,
    ErrorResponse,
)
from api.serialization import json_safe
from api.state import ApplicationState

router = APIRouter(tags=["applications"])


@router.get(
    "/applications",
    response_model=ApplicationListResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid filter or pagination."},
        503: {"model": ErrorResponse, "description": "Application data unavailable."},
    },
)
def list_applications(
    fields: str | None = Query(default=None),
    limit: int = Query(
        default=DEFAULT_APPLICATION_LIMIT,
        ge=1,
        le=MAX_APPLICATION_LIMIT,
    ),
    offset: int = Query(default=0, ge=0),
    application_id: int | None = Query(default=None, gt=0),
    filter_field: str | None = Query(default=None),
    filter_value: str | None = Query(default=None),
    state: ApplicationState = Depends(get_state),
) -> ApplicationListResponse | JSONResponse:
    repository = state.repository
    if repository is None or not repository.available:
        return JSONResponse(
            status_code=503,
            content=error_payload(
                "unavailable", "Application data is unavailable."
            ),
        )
    try:
        selected_fields = parse_fields(fields)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content=error_payload("invalid", str(exc)),
        )

    if (filter_field is None) != (filter_value is None):
        return JSONResponse(
            status_code=422,
            content=error_payload(
                "invalid",
                "filter_field and filter_value must be provided together.",
            ),
        )
    filters: dict[str, str] = {}
    if filter_field is not None and filter_value is not None:
        if filter_field not in FILTERABLE_APPLICATION_FIELDS:
            return JSONResponse(
                status_code=422,
                content=error_payload(
                    "invalid",
                    "Unsupported application filter field. Allowed fields: "
                    + ", ".join(FILTERABLE_APPLICATION_FIELDS)
                    + ".",
                ),
            )
        filters[filter_field] = filter_value

    rows, total = repository.list_rows(
        selected_fields,
        limit,
        offset,
        application_id=application_id,
        filters=filters,
    )
    return ApplicationListResponse(
        total=total,
        limit=limit,
        offset=offset,
        fields=selected_fields,
        items=[json_safe(row) for row in rows],
    )


@router.get(
    "/applications/{application_id}",
    response_model=ApplicationDetailResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Application not found."},
        409: {"model": ErrorResponse, "description": "Duplicate application ID."},
        422: {"model": ErrorResponse, "description": "Invalid request."},
        503: {"model": ErrorResponse, "description": "Application data unavailable."},
    },
)
def get_application(
    application_id: int = Path(gt=0),
    fields: str | None = Query(default=None),
    state: ApplicationState = Depends(get_state),
) -> ApplicationDetailResponse | JSONResponse:
    repository = state.repository
    if repository is None or not repository.available:
        return JSONResponse(
            status_code=503,
            content=error_payload(
                "unavailable", "Application data is unavailable."
            ),
        )
    selected_fields: list[str] | None = None
    if fields is not None:
        try:
            selected_fields = parse_fields(fields)
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content=error_payload("invalid", str(exc)),
            )

    try:
        row = repository.get_full_by_id(application_id)
    except DuplicateApplicationIdError as exc:
        return JSONResponse(
            status_code=409,
            content=error_payload("invalid", str(exc)),
        )
    if row is None:
        return JSONResponse(
            status_code=404,
            content=error_payload(
                "invalid", f"Application {application_id} was not found."
            ),
        )
    columns = (
        selected_fields
        if selected_fields is not None
        else [field for field in parse_fields(None) if field in row.index]
    )
    return ApplicationDetailResponse(
        application_id=application_id,
        application={
            str(column): json_safe(row[column]) for column in columns
        },
    )
