from __future__ import annotations

import json

import pandas as pd

from aws.sagemaker.compare_metrics import compare, load_metrics, load_predictions


def _write_side(
    folder,
    ids: list[int],
    labels: list[int],
    probabilities: list[float],
    *,
    include_ks: bool,
) -> None:
    folder.mkdir(parents=True)
    (folder / "evaluation_results.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "test": {"roc_auc": 0.75, "pr_auc": 0.75},
                }
            }
        ),
        encoding="utf-8",
    )
    if include_ks:
        (folder / "extra_metrics.json").write_text(
            json.dumps({"test": {"ks": 0.5}}), encoding="utf-8"
        )
    pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "y_true": labels,
            "probability": probabilities,
            "prediction": [int(value >= 0.5) for value in probabilities],
        }
    ).to_csv(folder / "test_predictions.csv", index=False)


def test_comparison_computes_missing_ks_and_aligns_predictions(tmp_path, capsys) -> None:
    local_dir = tmp_path / "local"
    sagemaker_dir = tmp_path / "sagemaker"
    _write_side(
        local_dir,
        [11, 22, 33, 44],
        [0, 0, 1, 1],
        [0.1, 0.4, 0.35, 0.8],
        include_ks=True,
    )
    _write_side(
        sagemaker_dir,
        [33, 11, 44, 22],
        [1, 0, 1, 0],
        [0.351, 0.101, 0.801, 0.401],
        include_ks=False,
    )

    predictions = load_predictions(sagemaker_dir)
    metrics = load_metrics(sagemaker_dir, predictions)
    passed = compare(local_dir, sagemaker_dir, tolerance=0.005)

    assert metrics["ks"] == 0.5
    assert passed is True
    output = capsys.readouterr().out
    assert "ROC-AUC" in output
    assert "PR-AUC" in output
    assert "Test applicant rows: MATCH" in output
    assert "Targets match: yes" in output
    assert "Max absolute probability difference: 0.00100000" in output
    assert output.rstrip().endswith("PASS")


def test_comparison_fails_when_test_applicants_differ(tmp_path, capsys) -> None:
    local_dir = tmp_path / "local"
    sagemaker_dir = tmp_path / "sagemaker"
    _write_side(local_dir, [1, 2], [0, 1], [0.1, 0.9], include_ks=True)
    _write_side(sagemaker_dir, [1, 3], [0, 1], [0.1, 0.9], include_ks=False)

    passed = compare(local_dir, sagemaker_dir, tolerance=0.005)

    assert passed is False
    output = capsys.readouterr().out
    assert "Test applicant rows: MISMATCH" in output
    assert output.rstrip().endswith("FAIL")
