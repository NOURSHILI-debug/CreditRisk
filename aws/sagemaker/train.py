"""SageMaker entry point for the shared local XGBoost baseline trainer."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _add_project_src(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def run_training(
    data_path: Path,
    model_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> dict[str, object]:
    """Train through the project trainer and stage artifacts in SageMaker paths."""
    _add_project_src(project_root)
    from training.train_xgboost import train

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = train(data_path=data_path, output_dir=model_dir)

    model_artifacts = (
        "xgboost_baseline.json",
        "preprocessor.pkl",
        "feature_columns.json",
        "preprocessor_metadata.json",
        "model_metadata.json",
        "training_metadata.json",
    )
    for filename in model_artifacts:
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(f"Trainer did not create {model_dir / filename}")

    # SageMaker uploads /opt/ml/output/data as output.tar.gz. Keep that upload
    # limited to the requested evaluation artifacts and predictions.
    output_files = (
        "evaluation_results.json",
        "validation_predictions.csv",
        "test_predictions.csv",
        "extra_metrics.json",
    )
    for filename in output_files:
        source = model_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Trainer did not create {source}")
        shutil.copy2(source, output_dir / filename)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/opt/ml/input/data/train/application_train.csv"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("/opt/ml/model"))
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/ml/output/data"))
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd()
    )
    args = parser.parse_args()
    run_training(args.data_path, args.model_dir, args.output_dir, args.project_root)


if __name__ == "__main__":
    main()
