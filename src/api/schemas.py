from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.status import PredictionStatus

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: int | None = Field(default=None, gt=0)
    features: dict[str, Any] | None = None

    @model_validator(mode="after")
    def exactly_one_input(self) -> "PredictRequest":
        has_id = self.application_id is not None
        has_features = self.features is not None
        if has_id == has_features:
            raise ValueError(
                "Provide exactly one of application_id or features."
            )
        if has_features and not self.features:
            raise ValueError("features must not be empty.")
        return self


class PredictionResponse(BaseModel):
    status: PredictionStatus
    application_id: int | None = None
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_level: RiskLevel | None = None
    model_name: str
    model_version: str
    prediction_timestamp: str
    explanation: dict[str, Any] | None = None
    message: str


class ErrorResponse(BaseModel):
    """Uniform error body used by every non-2xx response."""

    status: PredictionStatus
    message: str
    application_id: int | None = None
    probability: float | None = None
    risk_level: RiskLevel | None = None
    model_name: str | None = None
    model_version: str | None = None
    prediction_timestamp: str | None = None
    explanation: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    data_available: bool
    model_name: str | None
    model_version: str | None
    model_hash: str | None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_hash: str
    training_date: str | None
    library_versions: dict[str, Any]
    thresholds: dict[str, Any]


class ApplicationDetailResponse(BaseModel):
    application_id: int
    application: dict[str, Any]


class ApplicationListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    fields: list[str]
    items: list[dict[str, Any]]


class EvaluationRow(BaseModel):
    model: str
    metric: str
    value: float


class EvaluationResponse(BaseModel):
    split: str
    rows: list[EvaluationRow]
    provenance: dict[str, Any]
    confusion_matrix: dict[str, Any] | None = None


class PredictionSummaryResponse(BaseModel):
    split: Literal["validation", "test"]
    prediction_count: int
    predicted_outcomes: dict[str, int]
    risk_distribution: dict[str, int]
    risk_bands: dict[str, float]
    decision_threshold: float | None
    source: str


class RootResponse(BaseModel):
    service: str
    version: str
    docs: str
