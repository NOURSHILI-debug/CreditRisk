from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from xgboost import XGBClassifier

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_paths import resolve_model_dir  # noqa: E402

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
MODEL_NAME = "XGBoost"
MODEL_VERSION = "baseline-v2-local"
FALLBACK_MODEL_VERSION = "baseline-v1"
RISK_BANDS = {
    "low_upper": 0.30,
    "medium_upper": 0.60,
}
RISK_BAND_NOTE = (
    "Provisional risk bands: LOW < 0.30, MEDIUM 0.30-0.60, HIGH >= 0.60. "
    "These are not validated lending thresholds and are decision support only."
)


class PredictionError(ValueError):
    """Raised when a prediction cannot be produced from the supplied input."""


def _default_model_dir() -> Path:
    return resolve_model_dir()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_application_id(application: Any) -> int | None:
    if isinstance(application, Mapping):
        frame = pd.DataFrame([dict(application)])
    elif isinstance(application, pd.DataFrame):
        frame = application
    else:
        return None
    if ID_COLUMN not in frame.columns or frame.empty:
        return None
    value = frame[ID_COLUMN].iloc[0]
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _risk_level(probability: float) -> str:
    if probability < RISK_BANDS["low_upper"]:
        return "LOW"
    if probability < RISK_BANDS["medium_upper"]:
        return "MEDIUM"
    return "HIGH"


def _error_response(
    status: str,
    message: str,
    application_id: int | None,
    predictor: CreditRiskPredictor | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "application_id": application_id,
        "probability": None,
        "risk_level": None,
        "model_name": predictor.model_name if predictor is not None else MODEL_NAME,
        "model_version": (
            predictor.model_version if predictor is not None else MODEL_VERSION
        ),
        "model_hash": predictor.model_hash if predictor is not None else None,
        "prediction_timestamp": _utc_timestamp(),
        "explanation": None,
        "message": message,
    }


class CreditRiskPredictor:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = resolve_model_dir(model_dir)
        self.model_path = self.model_dir / "xgboost_baseline.json"
        self.preprocessor_path = self.model_dir / "preprocessor.pkl"
        self.features_path = self.model_dir / "feature_columns.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {self.preprocessor_path}"
            )

        self.model = XGBClassifier()
        self.model.load_model(str(self.model_path))
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model_hash = _sha256_file(self.model_path)
        self.model_metadata = self._load_model_metadata()
        self.model_name = str(self.model_metadata.get("model_name", MODEL_NAME))
        self.model_version = str(
            self.model_metadata.get("model_version", FALLBACK_MODEL_VERSION)
        )
        metadata_hash = self.model_metadata.get("model_hash")
        if metadata_hash and metadata_hash != self.model_hash:
            raise ValueError(
                "model_metadata.json hash does not match the loaded model artifact."
            )
        self.feature_columns = self._load_feature_columns()

        forbidden = {ID_COLUMN, TARGET_COLUMN}.intersection(self.feature_columns)
        if forbidden:
            raise ValueError(
                "Unexpected identifier/target columns in the preprocessor: "
                f"{sorted(forbidden)}. Check the training pipeline."
            )

        self.model_feature_count = int(getattr(self.model, "n_features_in_", -1))
        self.post_transform_feature_count = self._post_transform_feature_count()
        if self.model_feature_count != self.post_transform_feature_count:
            raise ValueError(
                "Model/preprocessor feature-count mismatch: "
                f"model={self.model_feature_count}, "
                f"preprocessor={self.post_transform_feature_count}."
            )

    def _load_model_metadata(self) -> dict[str, Any]:
        path = self.model_dir / "model_metadata.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError("model_metadata.json must contain a JSON object.")
        return metadata

    def _load_feature_columns(self) -> list[str]:
        if self.features_path.exists():
            with self.features_path.open("r", encoding="utf-8") as handle:
                columns = json.load(handle)
        elif hasattr(self.preprocessor, "feature_names_in_"):
            columns = list(self.preprocessor.feature_names_in_)
        else:
            raise ValueError(
                "Cannot determine the training feature columns. "
                f"Expected {self.features_path} or a fitted preprocessor."
            )
        if not isinstance(columns, list) or not all(
            isinstance(column, str) for column in columns
        ):
            raise ValueError("feature_columns.json must contain a list of strings.")
        return columns

    def _post_transform_feature_count(self) -> int:
        if hasattr(self.preprocessor, "get_feature_names_out"):
            try:
                return len(self.preprocessor.get_feature_names_out())
            except Exception:
                pass
        return int(getattr(self.model, "n_features_in_", -1))

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_hash": self.model_hash,
            "model_dir": str(self.model_dir),
            "model_path": str(self.model_path),
            "preprocessor_path": str(self.preprocessor_path),
            "feature_columns_count": len(self.feature_columns),
            "post_transform_feature_count": self.post_transform_feature_count,
            "model_feature_count": self.model_feature_count,
            "risk_bands": dict(RISK_BANDS),
            "risk_band_note": RISK_BAND_NOTE,
        }

    def predict(
        self, application: Any, include_explanation: bool = False
    ) -> dict[str, Any]:
        if isinstance(application, Mapping):
            application = pd.DataFrame([dict(application)])
        if not isinstance(application, pd.DataFrame):
            raise PredictionError(
                "Input must be a pandas DataFrame or a dictionary."
            )
        if len(application) != 1:
            raise PredictionError("This function expects exactly one applicant.")

        frame = application.copy()
        if ID_COLUMN in frame.columns:
            try:
                application_id = int(frame[ID_COLUMN].iloc[0])
            except (TypeError, ValueError) as exc:
                raise PredictionError(
                    f"{ID_COLUMN} must be an integer."
                ) from exc
        else:
            application_id = None

        frame = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")
        missing_columns = [
            column for column in self.feature_columns if column not in frame.columns
        ]
        if missing_columns:
            raise PredictionError(f"Missing required features: {missing_columns}")

        frame = frame[self.feature_columns]
        processed_application = self.preprocessor.transform(frame)
        processed_feature_count = int(getattr(processed_application, "shape", (0, 0))[1])
        if processed_feature_count != self.model_feature_count:
            raise PredictionError(
                "Transformed feature-count mismatch: "
                f"model={self.model_feature_count}, "
                f"preprocessor={processed_feature_count}."
            )

        classes = [int(value) for value in self.model.classes_]
        if 1 not in classes:
            raise PredictionError(
                "The model does not contain class 1 (TARGET = 1)."
            )
        positive_class_index = classes.index(1)
        probabilities = self.model.predict_proba(processed_application)
        probability = float(probabilities[0][positive_class_index])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise PredictionError(
                f"Model returned an invalid probability: {probability!r}."
            )

        explanation = None
        if include_explanation:
            try:
                from explainability.shap_explainer import explain_application

                explanation = explain_application(
                    application,
                    model_dir=self.model_dir,
                )
            except Exception as exc:
                raise PredictionError(
                    f"Explanation generation failed: {exc}"
                ) from exc

        return {
            "status": "success",
            "application_id": application_id,
            "probability": probability,
            "risk_level": _risk_level(probability),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_hash": self.model_hash,
            "prediction_timestamp": _utc_timestamp(),
            "explanation": explanation,
            "message": "Prediction generated successfully.",
        }


_PREDICTORS: dict[str, CreditRiskPredictor] = {}


def _resolve_model_dir(model_dir: str | Path | None) -> Path:
    return resolve_model_dir(model_dir)


def _get_predictor(model_dir: str | Path | None) -> CreditRiskPredictor:
    resolved = _resolve_model_dir(model_dir).resolve()
    key = str(resolved)
    if key not in _PREDICTORS:
        _PREDICTORS[key] = CreditRiskPredictor(resolved)
    return _PREDICTORS[key]


def predict_credit_risk(
    application: Any,
    model_dir: str | Path | None = None,
    include_explanation: bool = False,
) -> dict[str, Any]:
    application_id = _safe_application_id(application)
    try:
        predictor = _get_predictor(model_dir)
    except FileNotFoundError as exc:
        return _error_response("unavailable", str(exc), application_id)
    except Exception as exc:
        return _error_response(
            "unavailable", f"Model unavailable: {exc}", application_id
        )

    try:
        return predictor.predict(
            application,
            include_explanation=include_explanation,
        )
    except (PredictionError, TypeError, ValueError, KeyError) as exc:
        return _error_response("error", str(exc), application_id, predictor)
    except Exception as exc:
        return _error_response(
            "error", f"Prediction failed: {exc}", application_id, predictor
        )
