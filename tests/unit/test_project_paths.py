from __future__ import annotations

from pathlib import Path

import project_paths
from api.config import Settings
from explainability import shap_explainer
from inference import predictor
from inference.predictor import predict_credit_risk
from training import train_xgboost


def test_all_defaults_share_local_v2_from_project_root() -> None:
    expected = project_paths.PROJECT_ROOT / "models" / "local_v2"

    assert project_paths.DEFAULT_MODEL_DIR == expected
    assert train_xgboost.DEFAULT_OUTPUT_DIR == expected
    assert predictor._default_model_dir() == expected
    assert shap_explainer._default_model_dir() == expected
    assert Settings.from_env(env={}).model_dir == expected


def test_env_override_relative_resolves_against_project_root(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CREDIT_RISK_MODEL_DIR", "models/from_env")

    expected = project_paths.PROJECT_ROOT / "models" / "from_env"
    assert project_paths.resolve_model_dir() == expected
    assert predictor._default_model_dir() == expected
    assert shap_explainer._default_model_dir() == expected


def test_env_override_absolute_is_used_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CREDIT_RISK_MODEL_DIR", str(tmp_path))

    assert project_paths.resolve_model_dir() == tmp_path
    assert predictor._default_model_dir() == tmp_path


def test_api_env_relative_override_resolves_against_project_root() -> None:
    settings = Settings.from_env(
        env={"CREDIT_RISK_MODEL_DIR": "models/legacy"}
    )

    assert settings.model_dir == project_paths.PROJECT_ROOT / "models" / "legacy"


def test_legacy_explicit_models_path_still_works() -> None:
    assert (
        project_paths.resolve_model_dir("models")
        == project_paths.PROJECT_ROOT / "models"
    )


def test_missing_model_returns_unavailable_without_exception(
    tmp_path: Path,
) -> None:
    result = predict_credit_risk({"AMT_CREDIT": 1000.0}, model_dir=tmp_path)

    assert result["status"] == "unavailable"
    assert result["probability"] is None
    assert result["message"]
