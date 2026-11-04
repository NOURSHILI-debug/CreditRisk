from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from explainability.shap_explainer import (
    ExplanationError,
    ShapExplainer,
    clear_explainer_cache,
    explain_application,
    get_explainer,
)


def _load_application(project_root: Path, application_id: int) -> pd.DataFrame:
    data_path = project_root / "data" / "raw" / "application_train.csv"
    if not data_path.exists():
        pytest.skip("Raw application_train.csv is missing.")
    frame = pd.read_csv(data_path)
    matches = frame.loc[frame["SK_ID_CURR"] == application_id]
    if matches.empty:
        pytest.fail(f"Application {application_id} was not found.")
    return matches.iloc[[0]].copy()


def test_tree_shap_reconstructs_model_probability(
    model_dir: Path, sample_application: dict
) -> None:
    explainer = ShapExplainer(model_dir)
    frame = pd.DataFrame([sample_application])
    transformed = explainer.transform(frame)
    batch = explainer.compute_shap_batch(transformed)

    assert transformed.shape[1] == 306
    assert batch.shap_values.shape == (1, 306)
    assert batch.base_values.shape == (1,)
    assert float(batch.additivity_errors().max()) <= 1e-4
    assert float(batch.probability_reconstruction_errors().max()) <= 1e-6

    classes = [int(value) for value in explainer.model.classes_]
    expected = float(
        explainer.model.predict_proba(transformed)[0][classes.index(1)]
    )
    assert float(batch.probabilities[0]) == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


def test_known_borderline_application_has_expected_explanation(
    project_root: Path, model_dir: Path
) -> None:
    application = _load_application(project_root, 120321)
    result = explain_application(application, model_dir=model_dir)

    assert result["method"] == "tree_shap"
    assert result["output_space"] == "log_odds"
    assert result["application_id"] == 120321
    assert result["probability"] == pytest.approx(
        0.4915030896663666, rel=1e-9, abs=1e-12
    )
    assert result["base_probability"] == pytest.approx(
        0.5282766402556108, rel=1e-9, abs=1e-12
    )
    assert result["contribution_sum"] == pytest.approx(
        result["prediction_margin"] - result["base_value"], rel=0.0, abs=1e-6
    )
    assert len(result["top_features"]) == 10
    assert result["summary"]
    assert "risk-increasing" in result["summary"] or "risk-decreasing" in result["summary"]
    for feature in result["top_features"]:
        assert feature["feature"]
        assert feature["raw_feature"]
        assert feature["transformed_feature"]
        assert np.isfinite(feature["shap_value"])
        assert feature["effect"] in {
            "increases_risk",
            "decreases_risk",
            "neutral",
        }
    assert "approve" not in result["summary"].lower()
    assert "reject" not in result["summary"].lower()


def test_global_importance_aggregates_all_raw_features(
    model_dir: Path, sample_application: dict
) -> None:
    explainer = ShapExplainer(model_dir)
    applications = pd.DataFrame(
        [
            sample_application,
            {
                **sample_application,
                "SK_ID_CURR": int(sample_application["SK_ID_CURR"]) + 1,
                "AMT_CREDIT": float(sample_application["AMT_CREDIT"]) * 1.25,
            },
        ]
    )
    batch = explainer.compute_shap_batch(explainer.transform(applications))
    raw_names, raw_values = explainer.aggregate_shap_by_raw_feature(
        batch.shap_values
    )
    importance = explainer.global_importance(batch.shap_values)

    assert raw_values.shape == (2, 120)
    assert len(raw_names) == 120
    assert len(importance) == 120
    assert list(importance["rank"]) == list(range(1, 121))
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert not any(name.startswith("missingindicator_") for name in raw_names)
    assert set(importance["feature"]) == set(raw_names)


def test_explainer_cache_reuses_and_clears(model_dir: Path) -> None:
    clear_explainer_cache()
    first = get_explainer(model_dir)
    second = get_explainer(model_dir)
    assert first is second

    clear_explainer_cache()
    third = get_explainer(model_dir)
    assert third is not first


def test_explain_application_rejects_multiple_rows(
    model_dir: Path, sample_application: dict
) -> None:
    frame = pd.DataFrame(
        [
            sample_application,
            {
                **sample_application,
                "SK_ID_CURR": int(sample_application["SK_ID_CURR"]) + 1,
            },
        ]
    )
    with pytest.raises(ExplanationError, match="exactly one"):
        explain_application(frame, model_dir=model_dir)
