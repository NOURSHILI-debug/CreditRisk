"""Local and SageMaker Processing wrapper around the shared XGBoost trainer."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _find_project_root(script_path: Path) -> Path:
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "src" / "training" / "train_xgboost.py").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find src/training/train_xgboost.py from {script_path}"
    )


def _load_shared_trainer(project_root: Path) -> ModuleType:
    trainer_path = project_root / "train.py"
    if not trainer_path.is_file():
        trainer_path = project_root / "aws" / "sagemaker" / "train.py"
    if not trainer_path.is_file():
        raise FileNotFoundError(f"Could not find the shared trainer at {trainer_path}")

    spec = importlib.util.spec_from_file_location(
        "credit_risk_sagemaker_shared_trainer", trainer_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the shared trainer from {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_processing_training(
    data_path: Path,
    model_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Run the existing trainer with caller-supplied local or Processing paths."""
    shared_trainer = _load_shared_trainer(project_root)
    return shared_trainer.run_training(
        data_path=data_path,
        model_dir=model_dir,
        output_dir=output_dir,
        project_root=project_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/opt/ml/processing/input/application_train.csv"),
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/opt/ml/processing/model")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/opt/ml/processing/output")
    )
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    project_root = args.project_root or _find_project_root(Path(__file__).resolve())
    run_processing_training(
        data_path=args.data_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
