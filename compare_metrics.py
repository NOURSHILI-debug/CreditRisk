"""Compare local_v2 metrics with an extracted SageMaker output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("roc_auc", "pr_auc", "ks")


def load_metrics(folder: Path) -> dict[str, float]:
    evaluation = json.loads((folder / "evaluation_results.json").read_text())
    extra = json.loads((folder / "extra_metrics.json").read_text())
    return {
        "roc_auc": float(evaluation["metrics"]["test"]["roc_auc"]),
        "pr_auc": float(evaluation["metrics"]["test"]["pr_auc"]),
        "ks": float(extra["test"]["ks"]),
    }


def compare(local_dir: Path, sagemaker_dir: Path, tolerance: float) -> bool:
    local = load_metrics(local_dir)
    sagemaker = load_metrics(sagemaker_dir)
    local_predictions = pd.read_csv(local_dir / "test_predictions.csv").set_index(
        "SK_ID_CURR"
    )
    sagemaker_predictions = pd.read_csv(
        sagemaker_dir / "test_predictions.csv"
    ).set_index("SK_ID_CURR")
    if local_predictions.index.has_duplicates or sagemaker_predictions.index.has_duplicates:
        raise ValueError("test_predictions.csv must have unique SK_ID_CURR values")
    if set(local_predictions.index) != set(sagemaker_predictions.index):
        raise ValueError("Local and SageMaker test predictions contain different rows")
    aligned = sagemaker_predictions.loc[local_predictions.index]
    if not local_predictions["y_true"].equals(aligned["y_true"]):
        raise ValueError("Targets differ for matched SK_ID_CURR rows")

    print(f"{'Metric':<12} {'Local':>12} {'SageMaker':>12} {'Difference':>12}")
    passed = True
    for metric in METRICS:
        difference = sagemaker[metric] - local[metric]
        print(
            f"{metric:<12} {local[metric]:>12.6f} "
            f"{sagemaker[metric]:>12.6f} {difference:>+12.6f}"
        )
        passed &= abs(difference) <= tolerance
    print(f"Aligned test rows: {len(local_predictions)}")
    print("PASS" if passed else "FAIL")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sagemaker_dir", type=Path)
    parser.add_argument(
        "--local-dir", type=Path, default=Path("models/local_v2")
    )
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")
    passed = compare(args.local_dir, args.sagemaker_dir, args.tolerance)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
