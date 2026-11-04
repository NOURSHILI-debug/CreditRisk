from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.config import Settings
from api.features import FeatureValidator
from api.repository import ApplicationRepository
from explainability.shap_explainer import ShapExplainer
from inference.predictor import (
    MODEL_NAME,
    RISK_BANDS,
    RISK_BAND_NOTE,
    CreditRiskPredictor,
)

LOGGER = logging.getLogger("credit_risk_api")
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


LOCAL_PATH_KEYS = {
    "model_dir",
    "model_path",
    "preprocessor_path",
    "source_data",
    "source_predictions",
}


def _without_local_paths(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    for key in LOCAL_PATH_KEYS:
        sanitized.pop(key, None)
    return sanitized


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None, bool]:
    if not path.exists():
        return None, "Artifact is not available.", True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Artifact could not be parsed: {exc}", False
    if not isinstance(payload, dict):
        return None, "Artifact must contain a JSON object.", False
    return payload, None, False


@dataclass
class ApplicationState:
    settings: Settings
    predictor: CreditRiskPredictor | None
    explainer: ShapExplainer | None
    repository: ApplicationRepository | None
    feature_validator: FeatureValidator | None
    model_metadata: dict[str, Any] = field(default_factory=dict)
    training_metadata: dict[str, Any] = field(default_factory=dict)
    preprocessor_metadata: dict[str, Any] = field(default_factory=dict)
    evaluation_results: dict[str, Any] | None = None
    global_importance: dict[str, Any] | None = None
    model_error: str | None = None
    data_error: str | None = None
    explainer_error: str | None = None
    evaluation_error: str | None = None
    evaluation_missing: bool = False
    global_importance_error: str | None = None
    global_importance_missing: bool = False

    @property
    def model_loaded(self) -> bool:
        return self.predictor is not None

    @property
    def data_available(self) -> bool:
        return bool(self.repository is not None and self.repository.available)

    @property
    def prediction_ready(self) -> bool:
        return self.model_loaded and self.data_available

    @property
    def model_name(self) -> str | None:
        if self.predictor is not None:
            return self.predictor.model_name
        value = self.model_metadata.get("model_name")
        return str(value) if value else None

    @property
    def model_version(self) -> str | None:
        if self.predictor is not None:
            return self.predictor.model_version
        value = self.model_metadata.get("model_version")
        return str(value) if value else None

    @property
    def model_hash(self) -> str | None:
        if self.predictor is not None:
            return self.predictor.model_hash
        value = self.model_metadata.get("model_hash")
        return str(value) if value else None

    @property
    def training_date(self) -> str | None:
        value = self.model_metadata.get(
            "training_date", self.training_metadata.get("training_date")
        )
        return str(value) if value else None

    @property
    def library_versions(self) -> dict[str, Any]:
        value = self.training_metadata.get("library_versions", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def decision_threshold(self) -> float | None:
        if not isinstance(self.evaluation_results, dict):
            return None
        value = self.evaluation_results.get("threshold")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def prediction_error(
        self,
        status: str,
        message: str,
        application_id: int | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "application_id": application_id,
            "probability": None,
            "risk_level": None,
            "model_name": self.model_name or MODEL_NAME,
            "model_version": self.model_version or "unknown",
            "prediction_timestamp": utc_timestamp(),
            "explanation": None,
            "message": message,
        }


def _build_feature_validator(
    predictor: CreditRiskPredictor,
    repository: ApplicationRepository,
    preprocessor_metadata: dict[str, Any],
) -> FeatureValidator:
    bounds = repository.numeric_bounds
    configured_numeric = [
        str(value) for value in preprocessor_metadata.get("numeric_features", [])
    ]
    numeric_features = [
        feature for feature in configured_numeric if feature in bounds
    ]
    if not numeric_features:
        numeric_features = [
            feature for feature in predictor.feature_columns if feature in bounds
        ]
    categorical_features = [
        str(value)
        for value in preprocessor_metadata.get("categorical_features", [])
    ]
    if not categorical_features:
        numeric_set = set(numeric_features)
        categorical_features = [
            feature
            for feature in predictor.feature_columns
            if feature not in numeric_set
        ]
    return FeatureValidator(
        feature_columns=predictor.feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_bounds=bounds,
    )


def build_state(settings: Settings) -> ApplicationState:
    model_metadata, model_metadata_error, _ = _read_json(
        settings.model_dir / "model_metadata.json"
    )
    training_metadata, training_metadata_error, _ = _read_json(
        settings.model_dir / "training_metadata.json"
    )
    preprocessor_metadata, preprocessor_metadata_error, _ = _read_json(
        settings.model_dir / "preprocessor_metadata.json"
    )
    evaluation, evaluation_error, evaluation_missing = _read_json(
        settings.model_dir / "evaluation_results.json"
    )
    global_importance, importance_error, importance_missing = _read_json(
        settings.model_dir / "global_importance.json"
    )
    if global_importance is not None:
        global_importance = _without_local_paths(global_importance)
    if evaluation is not None:
        evaluation = _without_local_paths(evaluation)

    predictor: CreditRiskPredictor | None = None
    model_error: str | None = None
    try:
        predictor = CreditRiskPredictor(settings.model_dir)
    except Exception as exc:
        model_error = str(exc)
        LOGGER.warning("model_load_failed error=%s", exc)

    explainer: ShapExplainer | None = None
    explainer_error: str | None = None
    if predictor is not None:
        try:
            explainer = ShapExplainer(settings.model_dir)
        except Exception as exc:
            explainer_error = str(exc)
            LOGGER.warning("explainer_load_failed error=%s", exc)

    repository: ApplicationRepository | None = None
    data_error: str | None = None
    try:
        repository = ApplicationRepository(settings.application_data_path)
        if not repository.available:
            data_error = repository.error
            LOGGER.warning("application_data_unavailable error=%s", data_error)
    except Exception as exc:
        data_error = str(exc)
        LOGGER.warning("application_data_load_failed error=%s", exc)

    for name, error in (
        ("model_metadata", model_metadata_error),
        ("training_metadata", training_metadata_error),
        ("preprocessor_metadata", preprocessor_metadata_error),
    ):
        if error:
            LOGGER.warning("%s_unavailable error=%s", name, error)

    validator: FeatureValidator | None = None
    if predictor is not None and repository is not None and repository.available:
        try:
            validator = _build_feature_validator(
                predictor,
                repository,
                preprocessor_metadata or {},
            )
        except Exception as exc:
            LOGGER.warning("feature_validator_build_failed error=%s", exc)

    return ApplicationState(
        settings=settings,
        predictor=predictor,
        explainer=explainer,
        repository=repository,
        feature_validator=validator,
        model_metadata=model_metadata or {},
        training_metadata=training_metadata or {},
        preprocessor_metadata=preprocessor_metadata or {},
        evaluation_results=evaluation,
        global_importance=global_importance,
        model_error=model_error,
        data_error=data_error,
        explainer_error=explainer_error,
        evaluation_error=evaluation_error,
        evaluation_missing=evaluation_missing,
        global_importance_error=importance_error,
        global_importance_missing=importance_missing,
    )


def threshold_payload(state: ApplicationState) -> dict[str, Any]:
    return {
        "decision_threshold": state.decision_threshold,
        "risk_bands": dict(RISK_BANDS),
        "risk_band_note": RISK_BAND_NOTE,
        "provisional": True,
        "note": (
            "Thresholds and risk bands are provisional, not validated lending "
            "thresholds, and must not drive automatic approval or rejection."
        ),
    }
