from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

from inference.predictor import (
    MODEL_VERSION,
    CreditRiskPredictor,
    predict_credit_risk,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_and_preprocessor_load_with_matching_feature_counts(
    model_dir: Path, sample_application: dict
) -> None:
    predictor = CreditRiskPredictor(model_dir)
    frame = pd.DataFrame([sample_application])[predictor.feature_columns]
    transformed = predictor.preprocessor.transform(frame)

    assert len(predictor.feature_columns) == predictor.preprocessor.n_features_in_
    assert transformed.shape[1] == predictor.model.n_features_in_
    assert predictor.model.n_features_in_ == predictor.post_transform_feature_count
    assert predictor.model_hash == _sha256(model_dir / "xgboost_baseline.json")


def test_known_application_returns_valid_probability(
    model_dir: Path, sample_application: dict
) -> None:
    result = predict_credit_risk(sample_application, model_dir=model_dir)

    assert result["status"] == "success"
    assert result["application_id"] == int(sample_application["SK_ID_CURR"])
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["model_name"] == "XGBoost"
    assert result["model_version"] == MODEL_VERSION
    assert result["model_hash"] == _sha256(model_dir / "xgboost_baseline.json")
    assert result["prediction_timestamp"].endswith("+00:00")
    assert result["explanation"] is None
    assert result["message"]


def test_explanation_is_optional_and_consistent(
    model_dir: Path, sample_application: dict
) -> None:
    without_explanation = predict_credit_risk(
        sample_application, model_dir=model_dir
    )
    with_explanation = predict_credit_risk(
        sample_application,
        model_dir=model_dir,
        include_explanation=True,
    )

    assert without_explanation["explanation"] is None
    assert with_explanation["status"] == "success"
    assert with_explanation["explanation"] is not None
    assert with_explanation["probability"] == pytest.approx(
        without_explanation["probability"], rel=1e-12, abs=1e-12
    )
    explanation = with_explanation["explanation"]
    assert explanation["application_id"] == int(sample_application["SK_ID_CURR"])
    assert explanation["probability"] == pytest.approx(
        with_explanation["probability"], rel=1e-12, abs=1e-12
    )
    assert explanation["base_probability"] == pytest.approx(0.5282766402556108, rel=1e-9)
    assert explanation["top_features"]
    assert explanation["summary"]


def test_probability_is_p_target_1_from_predict_proba(
    model_dir: Path, sample_application: dict
) -> None:
    predictor = CreditRiskPredictor(model_dir)
    frame = pd.DataFrame([sample_application])[predictor.feature_columns]
    transformed = predictor.preprocessor.transform(frame)
    classes = [int(value) for value in predictor.model.classes_]
    expected = float(
        predictor.model.predict_proba(transformed)[0][classes.index(1)]
    )

    result = predict_credit_risk(sample_application, model_dir=model_dir)

    assert result["probability"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_missing_column_returns_handled_error(
    model_dir: Path, sample_application: dict
) -> None:
    application = dict(sample_application)
    application.pop("AMT_CREDIT")

    result = predict_credit_risk(application, model_dir=model_dir)

    assert result["status"] == "error"
    assert result["probability"] is None
    assert result["risk_level"] is None
    assert "AMT_CREDIT" in result["message"]


def test_unseen_category_is_handled_without_crashing(
    model_dir: Path, sample_application: dict
) -> None:
    application = dict(sample_application)
    application["NAME_CONTRACT_TYPE"] = "__UNSEEN_CATEGORY__"

    result = predict_credit_risk(application, model_dir=model_dir)

    assert result["status"] == "success"
    assert 0.0 <= result["probability"] <= 1.0


@pytest.mark.parametrize(
    "invalid_input",
    [
        [],
        "not-a-row",
        123,
        pd.DataFrame([{"SK_ID_CURR": 1}, {"SK_ID_CURR": 2}]),
    ],
)
def test_invalid_input_returns_handled_error(
    model_dir: Path, invalid_input: object
) -> None:
    result = predict_credit_risk(invalid_input, model_dir=model_dir)

    assert result["status"] == "error"
    assert result["probability"] is None
    assert result["message"]


def test_missing_artifacts_return_unavailable_without_default_probability(
    model_dir: Path, sample_application: dict, tmp_path: Path
) -> None:
    result = predict_credit_risk(sample_application, model_dir=tmp_path)

    assert result["status"] == "unavailable"
    assert result["probability"] is None
    assert result["risk_level"] is None
    assert result["message"]


def test_missing_preprocessor_returns_unavailable_without_exception(
    model_dir: Path, sample_application: dict, tmp_path: Path
) -> None:
    shutil.copy2(
        model_dir / "xgboost_baseline.json",
        tmp_path / "xgboost_baseline.json",
    )

    with pytest.raises(FileNotFoundError):
        CreditRiskPredictor(tmp_path)

    result = predict_credit_risk(sample_application, model_dir=tmp_path)
    assert result["status"] == "unavailable"
    assert result["probability"] is None
    assert result["message"]
