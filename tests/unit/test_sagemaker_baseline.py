from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from training.train_xgboost import compute_ks
from aws.sagemaker.run_sagemaker import FRAMEWORK_IMAGE_URI, job_config


def test_launcher_pins_sagemaker_xgboost_image() -> None:
    config = job_config()

    assert config["framework_version"] == "3.2-0"
    assert config["image_uri"] == FRAMEWORK_IMAGE_URI


def test_compute_ks_matches_roc_curve() -> None:
    truth = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7])
    fpr, tpr, _ = roc_curve(truth, probabilities)

    assert compute_ks(truth, probabilities) == float(np.max(tpr - fpr))


def test_sagemaker_entrypoint_runs_with_local_paths(tmp_path, monkeypatch) -> None:
    from training import train_xgboost

    from aws.sagemaker.train import run_training

    monkeypatch.setitem(train_xgboost.XGBOOST_PARAMS, "n_estimators", 5)
    row_count = 200
    data = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(1000, 1000 + row_count),
            "TARGET": np.tile([0, 1], row_count // 2),
            "AMT_INCOME_TOTAL": np.linspace(20000, 100000, row_count),
            "CODE_GENDER": np.tile(["F", "M"], row_count // 2),
        }
    )
    data_path = tmp_path / "application_train.csv"
    data.to_csv(data_path, index=False)
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    project_root = Path(__file__).resolve().parents[2]

    run_training(data_path, model_dir, output_dir, project_root)

    assert (model_dir / "xgboost_baseline.json").is_file()
    assert (model_dir / "preprocessor.pkl").is_file()
    result = pd.read_csv(output_dir / "test_predictions.csv")
    assert "SK_ID_CURR" in result.columns
    assert len(result) == 30
    extra = json.loads((output_dir / "extra_metrics.json").read_text())
    assert set(extra) == {"validation", "test"}
    assert 0 <= extra["test"]["ks"] <= 1
