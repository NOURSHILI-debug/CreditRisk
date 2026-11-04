from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from xgboost import DMatrix, XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "local_v2"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
MISSING_INDICATOR_PREFIX = "num__missingindicator_"
NUMERIC_PREFIX = "num__"
CATEGORICAL_PREFIX = "cat__"
DEFAULT_TOP_K = 10

_EXPLAINER_CACHE: dict[tuple[str, int, int], "ShapExplainer"] = {}
_EXPLAINER_CACHE_LOCK = threading.Lock()


class ExplanationError(ValueError):
    """Raised when a SHAP explanation cannot be produced."""


@dataclass
class ShapBatch:
    shap_values: np.ndarray
    base_values: np.ndarray
    margins: np.ndarray
    probabilities: np.ndarray

    @property
    def base_value(self) -> float:
        return float(self.base_values[0])

    @property
    def base_probability(self) -> float:
        return float(1.0 / (1.0 + math.exp(-self.base_value)))

    def additivity_errors(self) -> np.ndarray:
        reconstructed = self.base_values + self.shap_values.sum(axis=1)
        return np.abs(reconstructed - self.margins)

    def probability_reconstruction_errors(self) -> np.ndarray:
        reconstructed = 1.0 / (1.0 + np.exp(-self.margins))
        return np.abs(reconstructed - self.probabilities)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_model_dir() -> Path:
    configured = os.getenv("CREDIT_RISK_MODEL_DIR")
    return Path(configured) if configured else DEFAULT_MODEL_DIR


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _raw_feature_name(
    transformed_name: str,
    categorical_features: Sequence[str],
    all_features: Sequence[str],
) -> str:
    if transformed_name.startswith(MISSING_INDICATOR_PREFIX):
        return transformed_name[len(MISSING_INDICATOR_PREFIX) :]
    if transformed_name.startswith(NUMERIC_PREFIX):
        return transformed_name[len(NUMERIC_PREFIX) :]
    if transformed_name.startswith(CATEGORICAL_PREFIX):
        body = transformed_name[len(CATEGORICAL_PREFIX) :]
        for feature in sorted(categorical_features, key=len, reverse=True):
            if body.startswith(f"{feature}_"):
                return feature
        for feature in sorted(all_features, key=len, reverse=True):
            if body.startswith(f"{feature}_"):
                return feature
        return body
    return transformed_name


def _display_name(
    transformed_name: str,
    raw_name: str,
    categorical_features: Sequence[str],
    all_features: Sequence[str],
) -> str:
    if transformed_name.startswith(MISSING_INDICATOR_PREFIX):
        return f"{raw_name} [missing]"
    if transformed_name.startswith(CATEGORICAL_PREFIX):
        body = transformed_name[len(CATEGORICAL_PREFIX) :]
        prefix = f"{raw_name}_"
        value = body[len(prefix) :] if body.startswith(prefix) else body
        return f"{raw_name}={value}"
    return raw_name


def _row_values(matrix: Any, row_index: int) -> np.ndarray:
    if sparse.issparse(matrix):
        row = matrix.getrow(row_index).toarray().ravel()
    else:
        row = np.asarray(matrix)[row_index].ravel()
    return row


class ShapExplainer:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = (
            Path(model_dir).resolve() if model_dir is not None else _default_model_dir()
        )
        self.model_path = self.model_dir / "xgboost_baseline.json"
        self.preprocessor_path = self.model_dir / "preprocessor.pkl"
        self.features_path = self.model_dir / "feature_columns.json"
        self.model_metadata_path = self.model_dir / "model_metadata.json"
        self.preprocessor_metadata_path = (
            self.model_dir / "preprocessor_metadata.json"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {self.preprocessor_path}"
            )

        self.model = XGBClassifier()
        self.model.load_model(str(self.model_path))
        self.booster = self.model.get_booster()
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model_hash = _sha256_file(self.model_path)
        self.model_metadata = _load_json(self.model_metadata_path)
        self.model_name = str(self.model_metadata.get("model_name", "XGBoost"))
        self.model_version = str(
            self.model_metadata.get("model_version", "unknown")
        )
        metadata_hash = self.model_metadata.get("model_hash")
        if metadata_hash and metadata_hash != self.model_hash:
            raise ValueError(
                "model_metadata.json hash does not match the loaded model artifact."
            )

        self.feature_columns = self._load_feature_columns()
        self.post_transform_features = [
            str(name) for name in self.preprocessor.get_feature_names_out()
        ]
        self.model_feature_count = int(getattr(self.model, "n_features_in_", -1))
        if len(self.post_transform_features) != self.model_feature_count:
            raise ValueError(
                "Model/preprocessor feature-count mismatch: "
                f"model={self.model_feature_count}, "
                f"preprocessor={len(self.post_transform_features)}."
            )

        self.classes = [int(value) for value in self.model.classes_]
        if 1 not in self.classes:
            raise ValueError("The model does not contain class 1 (TARGET = 1).")
        self.positive_class_index = self.classes.index(1)

        preprocessor_metadata = _load_json(self.preprocessor_metadata_path)
        self.categorical_features = list(
            preprocessor_metadata.get("categorical_features", [])
        )
        self.raw_feature_by_transformed = [
            _raw_feature_name(
                name, self.categorical_features, self.feature_columns
            )
            for name in self.post_transform_features
        ]
        self.display_names = [
            _display_name(
                name,
                raw_name,
                self.categorical_features,
                self.feature_columns,
            )
            for name, raw_name in zip(
                self.post_transform_features, self.raw_feature_by_transformed
            )
        ]
        self.raw_features = self._unique_raw_features()

    def _load_feature_columns(self) -> list[str]:
        if self.features_path.exists():
            with self.features_path.open("r", encoding="utf-8") as handle:
                columns = json.load(handle)
        elif hasattr(self.preprocessor, "feature_names_in_"):
            columns = list(self.preprocessor.feature_names_in_)
        else:
            raise ValueError(
                "Cannot determine training feature columns. "
                f"Expected {self.features_path} or a fitted preprocessor."
            )
        if not isinstance(columns, list) or not all(
            isinstance(column, str) for column in columns
        ):
            raise ValueError("feature_columns.json must contain a list of strings.")
        return columns

    def _unique_raw_features(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for raw_name in self.raw_feature_by_transformed:
            if raw_name not in seen:
                seen.add(raw_name)
                ordered.append(raw_name)
        return ordered

    def transform(self, application: Any) -> Any:
        if isinstance(application, Mapping):
            application = pd.DataFrame([dict(application)])
        if not isinstance(application, pd.DataFrame):
            raise ExplanationError(
                "Input must be a pandas DataFrame or a dictionary."
            )
        if application.empty:
            raise ExplanationError("Input contains no rows.")

        frame = application.copy()
        frame = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")
        missing_columns = [
            column for column in self.feature_columns if column not in frame.columns
        ]
        if missing_columns:
            raise ExplanationError(f"Missing required features: {missing_columns}")
        return self.preprocessor.transform(frame[self.feature_columns])

    def predict_probability(self, transformed: Any) -> np.ndarray:
        probabilities = self.model.predict_proba(transformed)
        return np.asarray(probabilities)[:, self.positive_class_index]

    def compute_shap_batch(self, transformed: Any) -> ShapBatch:
        dmatrix = DMatrix(transformed)
        contributions = self.booster.predict(dmatrix, pred_contribs=True)
        contributions = np.asarray(contributions, dtype=np.float64)
        if contributions.ndim == 1:
            contributions = contributions.reshape(1, -1)
        margins = np.asarray(
            self.booster.predict(dmatrix, output_margin=True), dtype=np.float64
        )
        return ShapBatch(
            shap_values=contributions[:, :-1],
            base_values=contributions[:, -1],
            margins=margins,
            probabilities=np.asarray(
                self.predict_probability(transformed), dtype=np.float64
            ),
        )

    def aggregate_row_by_raw_feature(
        self, shap_row: np.ndarray
    ) -> dict[str, float]:
        totals: dict[str, float] = {}
        for raw_name, value in zip(self.raw_feature_by_transformed, shap_row):
            totals[raw_name] = totals.get(raw_name, 0.0) + float(value)
        return totals

    def aggregate_shap_by_raw_feature(
        self, shap_values: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        values = np.asarray(shap_values, dtype=np.float64)
        unique_raw, inverse = np.unique(
            np.asarray(self.raw_feature_by_transformed, dtype=object),
            return_inverse=True,
        )
        aggregated = np.zeros((values.shape[0], unique_raw.size), dtype=np.float64)
        for column_index, raw_index in enumerate(inverse):
            aggregated[:, raw_index] += values[:, column_index]
        return unique_raw.astype(str).tolist(), aggregated

    def global_importance(
        self, shap_values: np.ndarray
    ) -> pd.DataFrame:
        raw_names, aggregated = self.aggregate_shap_by_raw_feature(shap_values)
        mean_abs = np.abs(aggregated).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        return pd.DataFrame(
            {
                "rank": np.arange(1, len(raw_names) + 1, dtype=int),
                "feature": np.asarray(raw_names, dtype=object)[order],
                "mean_abs_shap": mean_abs[order],
            }
        )

    def build_summary(self, raw_totals: Mapping[str, float]) -> str:
        positive = sorted(
            ((name, value) for name, value in raw_totals.items() if value > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        negative = sorted(
            ((name, value) for name, value in raw_totals.items() if value < 0),
            key=lambda item: item[1],
        )[:3]

        parts: list[str] = []
        if positive:
            parts.append(
                "Largest risk-increasing features: "
                + ", ".join(name for name, _ in positive)
                + "."
            )
        if negative:
            parts.append(
                "Largest risk-decreasing features: "
                + ", ".join(name for name, _ in negative)
                + "."
            )
        if not parts:
            parts.append("No material feature contributions were found.")
        return " ".join(parts)

    def explanation_from_batch(
        self,
        batch: ShapBatch,
        row_index: int,
        transformed: Any,
        top_k: int = DEFAULT_TOP_K,
        application_id: int | None = None,
    ) -> dict[str, Any]:
        shap_row = batch.shap_values[row_index]
        row_values = _row_values(transformed, row_index)
        raw_totals = self.aggregate_row_by_raw_feature(shap_row)
        top_indices = np.argsort(np.abs(shap_row))[::-1][:top_k]
        top_features = []
        for index in top_indices:
            value = float(row_values[index])
            shap_value = float(shap_row[index])
            top_features.append(
                {
                    "feature": self.display_names[index],
                    "raw_feature": self.raw_feature_by_transformed[index],
                    "transformed_feature": self.post_transform_features[index],
                    "value": value,
                    "shap_value": shap_value,
                    "effect": (
                        "increases_risk"
                        if shap_value > 0
                        else "decreases_risk"
                        if shap_value < 0
                        else "neutral"
                    ),
                }
            )

        probability = float(batch.probabilities[row_index])
        base_value = float(batch.base_values[row_index])
        return {
            "method": "tree_shap",
            "output_space": "log_odds",
            "application_id": application_id,
            "base_value": base_value,
            "base_probability": float(1.0 / (1.0 + math.exp(-base_value))),
            "prediction_margin": float(batch.margins[row_index]),
            "contribution_sum": float(shap_row.sum()),
            "probability": probability,
            "top_features": top_features,
            "summary": self.build_summary(raw_totals),
            "note": (
                "Positive SHAP values increase P(TARGET=1); values are "
                "additive in log-odds."
            ),
        }

    def explain_transformed_row(
        self,
        transformed: Any,
        top_k: int = DEFAULT_TOP_K,
        application_id: int | None = None,
    ) -> dict[str, Any]:
        if sparse.issparse(transformed):
            row_count = transformed.shape[0]
        else:
            row_count = np.asarray(transformed).shape[0]
        if row_count != 1:
            raise ExplanationError(
                "explain_transformed_row expects exactly one row."
            )
        batch = self.compute_shap_batch(transformed)
        return self.explanation_from_batch(
            batch,
            row_index=0,
            transformed=transformed,
            top_k=top_k,
            application_id=application_id,
        )


def _model_signature(model_path: Path) -> tuple[int, int]:
    stat = model_path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def get_explainer(model_dir: str | Path | None = None) -> ShapExplainer:
    resolved = (
        Path(model_dir).resolve() if model_dir is not None else _default_model_dir()
    )
    model_path = resolved / "xgboost_baseline.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    signature = _model_signature(model_path)
    key = (str(resolved), *signature)
    with _EXPLAINER_CACHE_LOCK:
        explainer = _EXPLAINER_CACHE.get(key)
        if explainer is None:
            for stale_key in [
                candidate
                for candidate in _EXPLAINER_CACHE
                if candidate[0] == str(resolved) and candidate != key
            ]:
                del _EXPLAINER_CACHE[stale_key]
            explainer = ShapExplainer(resolved)
            _EXPLAINER_CACHE[key] = explainer
        return explainer


def clear_explainer_cache() -> None:
    with _EXPLAINER_CACHE_LOCK:
        _EXPLAINER_CACHE.clear()


def explain_application(
    application: Any,
    model_dir: str | Path | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    explainer = get_explainer(model_dir)
    if isinstance(application, Mapping):
        application = pd.DataFrame([dict(application)])
    if not isinstance(application, pd.DataFrame):
        raise ExplanationError(
            "Input must be a pandas DataFrame or a dictionary."
        )
    if len(application) != 1:
        raise ExplanationError("This function expects exactly one applicant.")

    application_id = None
    if ID_COLUMN in application.columns:
        raw_id = application[ID_COLUMN].iloc[0]
        if not pd.isna(raw_id):
            try:
                application_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ExplanationError(
                    f"{ID_COLUMN} must be an integer."
                ) from exc

    transformed = explainer.transform(application)
    batch = explainer.compute_shap_batch(transformed)
    return explainer.explanation_from_batch(
        batch,
        row_index=0,
        transformed=transformed,
        top_k=top_k,
        application_id=application_id,
    )


def utc_now() -> str:
    return _utc_now()
