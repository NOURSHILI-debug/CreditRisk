from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PredictionStatus = Literal["success", "error", "unavailable"]
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
