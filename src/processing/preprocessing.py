from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
DAYS_EMPLOYED_COLUMN = "DAYS_EMPLOYED"
DAYS_EMPLOYED_SENTINEL = 365243


class DaysEmployedSentinelFixer(BaseEstimator, TransformerMixin):
    """Replace the documented DAYS_EMPLOYED sentinel with NaN before imputation."""

    def __init__(
        self,
        column: str = DAYS_EMPLOYED_COLUMN,
        sentinel: float = DAYS_EMPLOYED_SENTINEL,
    ) -> None:
        self.column = column
        self.sentinel = sentinel

    def fit(self, X: pd.DataFrame, y: Iterable | None = None) -> DaysEmployedSentinelFixer:
        self._validate_dataframe(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_dataframe(X)
        transformed = X.copy()
        if self.column in transformed.columns:
            values = pd.to_numeric(transformed[self.column], errors="coerce")
            transformed[self.column] = values.mask(values.eq(self.sentinel), np.nan)

        numeric_columns = transformed.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            transformed[numeric_columns] = transformed[numeric_columns].replace(
                [np.inf, -np.inf], np.nan
            )
        return transformed

    def get_feature_names_out(self, input_features: Iterable | None = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        raise AttributeError("This transformer has not been fitted yet.")

    @staticmethod
    def _validate_dataframe(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DaysEmployedSentinelFixer expects a pandas DataFrame.")


def split_feature_columns(data: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return model feature columns split into numeric and categorical groups."""
    feature_columns = [
        column for column in data.columns if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    categorical_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(data[column])
        or pd.api.types.is_string_dtype(data[column])
        or isinstance(data[column].dtype, pd.CategoricalDtype)
    ]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    unsupported = [
        column
        for column in numeric_columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if unsupported:
        raise TypeError(f"Unsupported feature dtypes for columns: {unsupported}")
    return feature_columns, numeric_columns, categorical_columns


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    add_numeric_indicators: bool = True,
) -> Pipeline:
    """Build the deterministic preprocessor fitted on training data only."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=add_numeric_indicators),
            )
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(
        steps=[
            ("sentinel", DaysEmployedSentinelFixer()),
            ("columns", column_transformer),
        ]
    )


def get_post_transform_feature_names(preprocessor: Pipeline) -> list[str]:
    """Return the post-transform feature names produced by a fitted preprocessor."""
    return [str(name) for name in preprocessor.get_feature_names_out()]
