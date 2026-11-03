from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inference.predictor import CreditRiskPredictor
from processing.preprocessing import (
    DAYS_EMPLOYED_COLUMN,
    DAYS_EMPLOYED_SENTINEL,
    DaysEmployedSentinelFixer,
)


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def test_days_employed_sentinel_is_replaced_with_nan() -> None:
    fixer = DaysEmployedSentinelFixer()
    frame = pd.DataFrame(
        {
            DAYS_EMPLOYED_COLUMN: [DAYS_EMPLOYED_SENTINEL, 100.0],
            "AMT_CREDIT": [1000.0, 2000.0],
        }
    )

    transformed = fixer.fit_transform(frame)

    assert pd.isna(transformed[DAYS_EMPLOYED_COLUMN].iloc[0])
    assert transformed[DAYS_EMPLOYED_COLUMN].iloc[1] == 100.0


def test_feature_columns_exclude_identifier_and_target(model_dir: Path) -> None:
    predictor = CreditRiskPredictor(model_dir)

    assert "SK_ID_CURR" not in predictor.feature_columns
    assert "TARGET" not in predictor.feature_columns
    assert len(predictor.feature_columns) == predictor.preprocessor.n_features_in_


def test_days_employed_missing_indicator_is_present(model_dir: Path) -> None:
    predictor = CreditRiskPredictor(model_dir)
    names = [str(name) for name in predictor.preprocessor.get_feature_names_out()]

    assert any("missingindicator_DAYS_EMPLOYED" in name for name in names)


def test_preprocessor_is_deterministic_and_has_no_missing_values(
    model_dir: Path, sample_application: dict
) -> None:
    predictor = CreditRiskPredictor(model_dir)
    frame = pd.DataFrame([sample_application])[predictor.feature_columns]

    first = _dense(predictor.preprocessor.transform(frame))
    second = _dense(predictor.preprocessor.transform(frame))

    assert first.shape == second.shape
    assert np.allclose(first, second)
    assert np.isfinite(first).all()


def test_unseen_category_is_ignored_by_encoder(
    model_dir: Path, sample_application: dict
) -> None:
    predictor = CreditRiskPredictor(model_dir)
    application = dict(sample_application)
    application["NAME_CONTRACT_TYPE"] = "__UNSEEN_CATEGORY__"
    frame = pd.DataFrame([application])[predictor.feature_columns]

    transformed = _dense(predictor.preprocessor.transform(frame))

    assert transformed.shape[1] == predictor.model.n_features_in_
    assert np.isfinite(transformed).all()
