from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd

from api.repository import ID_COLUMN
from api.state import ApplicationState, utc_timestamp
from api.status import status_for_probability
from inference.predictor import PredictionError

LOGGER = logging.getLogger("credit_risk_api")


class ApplicationNotFoundError(LookupError):
    """Raised when an application ID is not present in the source data."""


class ModelUnavailableError(RuntimeError):
    """Raised when model artifacts or preprocessing metadata are unavailable."""


class DataUnavailableError(RuntimeError):
    """Raised when the application data source is unavailable."""


class InvalidPredictionInputError(ValueError):
    """Raised when incoming prediction input is outside the contract."""


class InvalidModelOutputError(RuntimeError):
    """Raised when the model returns a probability outside the contract."""


class PredictionService:
    def __init__(self, state: ApplicationState) -> None:
        self.state = state

    def _require_model(self) -> None:
        if self.state.predictor is None:
            raise ModelUnavailableError("Model artifacts are unavailable.")

    def _require_data(self) -> None:
        if self.state.repository is None or not self.state.repository.available:
            raise DataUnavailableError("Application data is unavailable.")

    def _require_feature_validation(self) -> None:
        if self.state.feature_validator is None:
            raise ModelUnavailableError(
                "Feature preprocessing metadata is unavailable."
            )

    def predict_by_application_id(
        self, application_id: int, include_explanation: bool = False
    ) -> dict[str, Any]:
        self._require_model()
        self._require_data()
        assert self.state.repository is not None
        row = self.state.repository.get_by_id(application_id)
        if row is None:
            raise ApplicationNotFoundError(
                f"Application {application_id} was not found."
            )
        frame = pd.DataFrame([row])
        return self._predict(frame, application_id, include_explanation)

    def predict_by_features(
        self, features: Mapping[str, Any], include_explanation: bool = False
    ) -> dict[str, Any]:
        self._require_model()
        self._require_data()
        self._require_feature_validation()
        assert self.state.feature_validator is not None
        try:
            frame = self.state.feature_validator.validate(features)
        except ValueError as exc:
            raise InvalidPredictionInputError(str(exc)) from exc

        application_id = None
        if ID_COLUMN in frame.columns and not pd.isna(frame[ID_COLUMN].iloc[0]):
            application_id = int(frame[ID_COLUMN].iloc[0])
        return self._predict(frame, application_id, include_explanation)

    def _predict(
        self,
        application: pd.DataFrame,
        application_id: int | None,
        include_explanation: bool,
    ) -> dict[str, Any]:
        assert self.state.predictor is not None
        try:
            result = self.state.predictor.predict(
                application, include_explanation=False
            )
        except PredictionError as exc:
            raise InvalidPredictionInputError(str(exc)) from exc

        status = result.get("status")
        if status == "no_prediction":
            return self._no_prediction(application_id, result)
        if status == "unavailable":
            raise ModelUnavailableError(
                str(result.get("message", "Model artifacts are unavailable."))
            )
        if status != "success":
            raise InvalidPredictionInputError(
                str(result.get("message", "Prediction input is invalid."))
            )

        probability_status = status_for_probability(result.get("probability"))
        if probability_status == "no_prediction":
            return self._no_prediction(application_id, result)
        if probability_status == "invalid":
            raise InvalidModelOutputError(
                "The model returned an invalid probability."
            )
        probability = float(result["probability"])

        explanation = None
        message = str(result.get("message", "Prediction generated successfully."))
        if include_explanation:
            explanation, explanation_message = self._explain(
                application, application_id, probability
            )
            if explanation_message:
                message = explanation_message

        return {
            "status": "available",
            "application_id": result.get("application_id", application_id),
            "probability": probability,
            "risk_level": result.get("risk_level"),
            "model_name": result.get("model_name") or self.state.model_name,
            "model_version": result.get("model_version")
            or self.state.model_version,
            "prediction_timestamp": result.get("prediction_timestamp")
            or utc_timestamp(),
            "explanation": explanation,
            "message": message,
        }

    def _no_prediction(
        self,
        application_id: int | None,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "no_prediction",
            "application_id": result.get("application_id", application_id),
            "probability": None,
            "risk_level": None,
            "model_name": result.get("model_name") or self.state.model_name,
            "model_version": result.get("model_version")
            or self.state.model_version,
            "prediction_timestamp": result.get("prediction_timestamp")
            or utc_timestamp(),
            "explanation": None,
            "message": str(
                result.get(
                    "message",
                    "No prediction is available for this application.",
                )
            ),
        }

    def _explain(
        self,
        application: pd.DataFrame,
        application_id: int | None,
        probability: float,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if self.state.explainer is None:
            LOGGER.warning(
                "explanation_unavailable application_id=%s reason=explainer_not_loaded",
                application_id,
            )
            return None, "Prediction generated successfully. Explanation unavailable."
        try:
            transformed = self.state.explainer.transform(application)
            batch = self.state.explainer.compute_shap_batch(transformed)
            explanation = self.state.explainer.explanation_from_batch(
                batch,
                row_index=0,
                transformed=transformed,
                application_id=application_id,
                raw_application=application,
            )
            if abs(float(explanation["probability"]) - probability) > 1e-12:
                raise RuntimeError(
                    "Explanation probability does not match prediction probability."
                )
            return explanation, None
        except Exception as exc:
            LOGGER.warning(
                "explanation_failed application_id=%s error_type=%s",
                application_id,
                type(exc).__name__,
            )
            return None, "Prediction generated successfully. Explanation unavailable."
