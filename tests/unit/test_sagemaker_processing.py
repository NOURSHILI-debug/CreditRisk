from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aws.sagemaker.processing_entrypoint import run_processing_training
from aws.sagemaker.run_processing import (
    FRAMEWORK_IMAGE_URI,
    job_config,
)
from training import train_xgboost


def test_processing_config_defaults_and_cost_estimate() -> None:
    config = job_config()

    assert config["instance_type"] == "ml.t3.xlarge"
    assert config["instance_count"] == 1
    assert config["volume_size_in_gb"] == 10
    assert config["max_runtime_in_seconds"] == 1800
    assert config["image_uri"] == FRAMEWORK_IMAGE_URI
    assert config["spot"] is False
    assert config["endpoint"] is False
    assert config["max_estimated_cost_usd"] == 0.11


def test_processing_entrypoint_runs_shared_trainer_on_local_csv(
    tmp_path, monkeypatch
) -> None:
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
    project_root = Path(__file__).resolve().parents[2]
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"

    run_processing_training(data_path, model_dir, output_dir, project_root)

    assert (model_dir / "xgboost_baseline.json").is_file()
    assert (model_dir / "preprocessor.pkl").is_file()
    predictions = pd.read_csv(output_dir / "test_predictions.csv")
    assert len(predictions) == 30
    assert "SK_ID_CURR" in predictions.columns
    metrics = json.loads((output_dir / "extra_metrics.json").read_text())
    assert set(metrics) == {"validation", "test"}
    assert 0 <= metrics["test"]["ks"] <= 1
