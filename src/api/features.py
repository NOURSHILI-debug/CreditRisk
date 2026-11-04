from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


def _is_missing(value: Any) -> bool:
    """Return True for None, NaN, pd.NA and other scalar missing sentinels."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class FeatureValidationError(ValueError):
    """Raised when an incoming feature mapping is outside the model contract."""


class FeatureValidator:
    def __init__(
        self,
        feature_columns: Sequence[str],
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        numeric_bounds: Mapping[str, tuple[float, float]],
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.numeric_features = set(numeric_features)
        self.categorical_features = set(categorical_features)
        self.numeric_bounds = dict(numeric_bounds)

    def validate(self, features: Mapping[str, Any]) -> pd.DataFrame:
        allowed = set(self.feature_columns) | {ID_COLUMN}
        unknown = sorted(set(features).difference(allowed))
        if unknown:
            raise FeatureValidationError(
                "Unknown feature columns: " + ", ".join(unknown)
            )
        if TARGET_COLUMN in features:
            raise FeatureValidationError(
                f"{TARGET_COLUMN} is not accepted as an input feature."
            )

        missing = [
            column for column in self.feature_columns if column not in features
        ]
        if missing:
            raise FeatureValidationError(
                "Missing required features: " + ", ".join(missing)
            )

        row: dict[str, Any] = {}
        for column in self.feature_columns:
            value = features[column]
            if _is_missing(value):
                # Normalize every missing sentinel to np.nan so the fitted
                # preprocessor imputes exactly as it does for the by-ID path.
                row[column] = np.nan
            elif column in self.numeric_features:
                row[column] = self._validate_numeric(column, value)
            elif column in self.categorical_features:
                row[column] = self._validate_categorical(column, value)
            else:
                row[column] = value

        identifier = features.get(ID_COLUMN)
        if not _is_missing(identifier):
            row[ID_COLUMN] = self._validate_identifier(identifier)
        return pd.DataFrame([row])

    def _validate_identifier(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise FeatureValidationError(f"{ID_COLUMN} must be a positive integer.")
        identifier = int(value)
        if identifier != value or identifier <= 0:
            raise FeatureValidationError(f"{ID_COLUMN} must be a positive integer.")
        return identifier

    def _validate_numeric(self, column: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise FeatureValidationError(
                f"{column} must be a numeric value or null."
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise FeatureValidationError(f"{column} must be finite.")
        bounds = self.numeric_bounds.get(column)
        if bounds is not None:
            lower, upper = bounds
            if numeric_value < lower or numeric_value > upper:
                raise FeatureValidationError(
                    f"{column} is outside the observed training range "
                    f"[{lower:g}, {upper:g}]."
                )
        return numeric_value

    def _validate_categorical(self, column: str, value: Any) -> str:
        if not isinstance(value, str):
            raise FeatureValidationError(
                f"{column} must be a string value or null."
            )
        return value
