from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def model_dir() -> Path:
    path = PROJECT_ROOT / "models" / "local_v2"
    if not (path / "xgboost_baseline.json").exists():
        pytest.skip(
            "models/local_v2 artifacts are missing; run "
            "python src/training/train_xgboost.py first."
        )
    return path


@pytest.fixture(scope="session")
def evaluation_results(model_dir: Path) -> dict:
    path = model_dir / "evaluation_results.json"
    if not path.exists():
        pytest.skip("evaluation_results.json is missing; run training first.")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sample_application(project_root: Path) -> dict:
    data_path = project_root / "data" / "raw" / "application_train.csv"
    if not data_path.exists():
        pytest.skip("Raw application_train.csv is missing.")
    frame = pd.read_csv(data_path, nrows=1)
    return frame.iloc[0].to_dict()
