"""Compare local and SageMaker XGBoost test metrics and predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

METRICS = ("roc_auc", "pr_auc", "ks")
METRIC_LABELS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "ks": "KS"}
PREDICTION_COLUMNS = {"SK_ID_CURR", "y_true", "probability"}


def compute_ks(y_true: pd.Series, probability: pd.Series) -> float:
    truth = y_true.to_numpy(dtype=int)
    if np.unique(truth).size != 2:
        raise ValueError("Cannot calculate KS unless test labels contain both classes")
    false_positive_rate, true_positive_rate, _ = roc_curve(
        truth, probability.to_numpy(dtype=float)
    )
    return float(np.max(true_positive_rate - false_positive_rate))


def load_predictions(folder: Path) -> pd.DataFrame:
    path = folder / "test_predictions.csv"
    predictions = pd.read_csv(path)
    missing = PREDICTION_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if predictions["SK_ID_CURR"].isna().any():
        raise ValueError(f"{path} contains missing SK_ID_CURR values")
    if predictions["SK_ID_CURR"].duplicated().any():
        raise ValueError(f"{path} must have unique SK_ID_CURR values")
    return predictions.set_index("SK_ID_CURR")


def load_metrics(folder: Path, predictions: pd.DataFrame) -> dict[str, float]:
    evaluation_path = folder / "evaluation_results.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    test_metrics = evaluation["metrics"]["test"]
    metrics = {
        "roc_auc": float(test_metrics["roc_auc"]),
        "pr_auc": float(test_metrics["pr_auc"]),
    }

    extra_path = folder / "extra_metrics.json"
    ks: object | None = None
    if extra_path.is_file():
        extra_metrics = json.loads(extra_path.read_text(encoding="utf-8"))
        ks = extra_metrics.get("test", {}).get("ks")
    metrics["ks"] = (
        float(ks)
        if ks is not None
        else compute_ks(predictions["y_true"], predictions["probability"])
    )
    return metrics


def compare(local_dir: Path, sagemaker_dir: Path, tolerance: float) -> bool:
    local_predictions = load_predictions(local_dir)
    sagemaker_predictions = load_predictions(sagemaker_dir)
    local_metrics = load_metrics(local_dir, local_predictions)
    sagemaker_metrics = load_metrics(sagemaker_dir, sagemaker_predictions)

    print(f"{'Metric (test)':<16} {'Local':>12} {'SageMaker':>12} {'Diff':>12}")
    metrics_pass = True
    for metric in METRICS:
        difference = sagemaker_metrics[metric] - local_metrics[metric]
        print(
            f"{METRIC_LABELS[metric]:<16} {local_metrics[metric]:>12.6f} "
            f"{sagemaker_metrics[metric]:>12.6f} {difference:>+12.6f}"
        )
        metrics_pass &= abs(difference) <= tolerance

    local_ids = set(local_predictions.index)
    sagemaker_ids = set(sagemaker_predictions.index)
    common_ids = [applicant_id for applicant_id in local_predictions.index if applicant_id in sagemaker_ids]
    same_rows = local_ids == sagemaker_ids
    if common_ids:
        local_aligned = local_predictions.loc[common_ids]
        sagemaker_aligned = sagemaker_predictions.loc[common_ids]
        same_labels = np.array_equal(
            local_aligned["y_true"].to_numpy(),
            sagemaker_aligned["y_true"].to_numpy(),
        )
        max_probability_difference = float(
            np.max(
                np.abs(
                    local_aligned["probability"].to_numpy(dtype=float)
                    - sagemaker_aligned["probability"].to_numpy(dtype=float)
                )
            )
        )
    else:
        same_labels = False
        max_probability_difference = None

    row_status = "MATCH" if same_rows else "MISMATCH"
    print(
        f"Test applicant rows: {row_status} "
        f"(local={len(local_ids)}, SageMaker={len(sagemaker_ids)}, aligned={len(common_ids)})"
    )
    print(f"Targets match: {'yes' if same_labels else 'no'}")
    if max_probability_difference is None:
        print("Max absolute probability difference: unavailable (no aligned rows)")
    else:
        print(f"Max absolute probability difference: {max_probability_difference:.8f}")

    passed = metrics_pass and same_rows and same_labels
    print("PASS" if passed else "FAIL")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, default=Path("models/local_v2"))
    parser.add_argument("--sagemaker", type=Path, default=Path("sagemaker_run"))
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")
    passed = compare(args.local, args.sagemaker, args.tolerance)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
