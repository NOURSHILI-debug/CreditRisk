from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evaluation_artifact_has_required_sections(
    evaluation_results: dict,
) -> None:
    required = {
        "model_name",
        "model_version",
        "model_hash",
        "training_date",
        "target",
        "seed",
        "threshold",
        "split_protocol",
        "splits",
        "scale_pos_weight",
        "metrics",
        "metric_definitions",
        "artifacts",
        "library_versions",
        "baseline_reference",
        "baseline_comparison",
    }
    assert required.issubset(evaluation_results)
    assert evaluation_results["target"]["probability_semantics"] == "P(TARGET=1)"
    assert evaluation_results["seed"] == 42
    assert evaluation_results["threshold"] == 0.5
    assert set(evaluation_results["metrics"]) == {"validation", "test"}


@pytest.mark.parametrize("split_name", ["validation", "test"])
def test_evaluation_metrics_are_in_valid_ranges(
    evaluation_results: dict, split_name: str
) -> None:
    metrics = evaluation_results["metrics"][split_name]

    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["threshold"] == 0.5
    assert 0.0 <= metrics["predicted_positive_rate"] <= 1.0
    assert 0.0 <= metrics["observed_positive_rate"] <= 1.0

    matrix = metrics["confusion_matrix"]
    assert matrix["labels"] == [0, 1]
    assert matrix["tn"] + matrix["fp"] + matrix["fn"] + matrix["tp"] == metrics[
        "n_samples"
    ]
    assert len(matrix["matrix"]) == 2
    assert all(len(row) == 2 for row in matrix["matrix"])


def test_evaluation_split_sizes_and_class_counts_are_consistent(
    evaluation_results: dict,
) -> None:
    splits = evaluation_results["splits"]
    total = sum(splits[name]["n_samples"] for name in ("train", "validation", "test"))
    assert total == 307511

    for split in splits.values():
        counts = split["class_counts"]
        assert counts["0"] + counts["1"] == split["n_samples"]
        assert split["positive_rate"] == pytest.approx(
            counts["1"] / split["n_samples"]
        )


def test_scale_pos_weight_matches_train_counts(evaluation_results: dict) -> None:
    counts = evaluation_results["splits"]["train"]["class_counts"]
    expected = counts["0"] / counts["1"]
    assert evaluation_results["scale_pos_weight"] == pytest.approx(expected)


def test_evaluation_artifact_hashes_match_files(
    evaluation_results: dict, project_root: Path
) -> None:
    for artifact_name in ("model", "preprocessor", "feature_columns"):
        record = evaluation_results["artifacts"][artifact_name]
        path = project_root / record["path"]
        assert path.exists(), f"Missing artifact: {path}"
        assert _sha256(path) == record["sha256"]
        assert path.stat().st_size == record["size_bytes"]

    assert evaluation_results["model_hash"] == evaluation_results["artifacts"][
        "model"
    ]["sha256"]


def test_baseline_comparison_is_context_only(evaluation_results: dict) -> None:
    comparison = evaluation_results["baseline_comparison"]
    assert comparison["tolerance"] == 0.01
    for metric in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        record = comparison["metrics"][metric]
        assert record["delta"] == pytest.approx(
            record["actual"] - record["baseline"]
        )
        assert record["exceeds_tolerance"] is (
            abs(record["delta"]) > comparison["tolerance"]
        )


def test_dashboard_normalizer_contract_is_available() -> None:
    from dashboard.data_access import normalize_model_evaluation_results

    results = pd.DataFrame(
        [{"model": "XGBoost", "metric": "ROC-AUC", "value": 0.75}]
    )
    values, statuses, issues = normalize_model_evaluation_results(results)
    assert values["XGBoost"]["ROC-AUC"] == 0.75
    assert statuses["XGBoost"].startswith("Partially available")
    assert statuses["LSTM"] == "Not available"
    assert issues == []
