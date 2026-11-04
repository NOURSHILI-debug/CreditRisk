from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.config import Settings
from api.logging_config import REQUEST_ID, configure_logging
from api.routers import health, models, predictions
from api.state import ApplicationState, build_state, utc_timestamp
from explainability.shap_explainer import clear_explainer_cache

LOGGER = logging.getLogger("credit_risk_api")


def _request_id(value: str | None) -> str:
    if value and value.isprintable() and len(value) <= 64:
        return value
    return uuid.uuid4().hex


def _error_payload(
    state: ApplicationState | None,
    message: str,
    status: str = "error",
) -> dict:
    if state is not None:
        return state.prediction_error(status, message)
    return {
        "status": status,
        "application_id": None,
        "probability": None,
        "risk_level": None,
        "model_name": "XGBoost",
        "model_version": "unknown",
        "prediction_timestamp": utc_timestamp(),
        "explanation": None,
        "message": message,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = build_state(resolved_settings)
        try:
            yield
        finally:
            clear_explainer_cache()

    app = FastAPI(
        title="Credit Risk Intelligence API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = _request_id(request.headers.get("X-Request-ID"))
        token = REQUEST_ID.set(request_id)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            LOGGER.exception(
                "request_failed",
                extra={
                    "event": "request_failed",
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            LOGGER.info(
                "request_completed",
                extra={
                    "event": "request_completed",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": elapsed_ms,
                },
            )
            REQUEST_ID.reset(token)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del exc
        state = getattr(request.app.state, "container", None)
        return JSONResponse(
            status_code=422,
            content=_error_payload(state, "Invalid request."),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        state = getattr(request.app.state, "container", None)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc.detail)},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        LOGGER.exception(
            "unhandled_exception",
            extra={
                "event": "unhandled_exception",
                "method": request.method,
                "path": request.url.path,
                "error_type": type(exc).__name__,
            },
        )
        state = getattr(request.app.state, "container", None)
        return JSONResponse(
            status_code=500,
            content=_error_payload(state, "Internal server error."),
        )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(predictions.router)
    return app


app = create_app()
