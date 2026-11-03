from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from explainability.plots import (  # noqa: E402
    plot_beeswarm,
    plot_global_bar,
    plot_local_waterfall,
)
from explainability.shap_explainer import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    ID_COLUMN,
    TARGET_COLUMN,
    ExplanationError,
    ShapExplainer,
    get_explainer,
)

ADDITIVITY_TOLERANCE = 1e-4
PROBABILITY_TOLERANCE = 1e-6
DEFAULT_SAMPLE_SIZE = 2_000
DEFAULT_SEED = 42
DEFAULT_LOCAL_IDS = (172753, 339432, 120321)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures" / "shap"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _stratified_sample_indices(
    targets: np.ndarray, sample_size: int, seed: int
) -> np.ndarray:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if sample_size >= len(targets):
        return np.arange(len(targets), dtype=int)
    indices = np.arange(len(targets), dtype=int)
    sampled, _ = train_test_split(
        indices,
        train_size=sample_size,
        random_state=seed,
        stratify=targets,
        shuffle=True,
    )
    return np.sort(np.asarray(sampled, dtype=int))


def _load_test_frame(
    data_path: Path,
    predictions_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(predictions_path)
    required = {ID_COLUMN, "y_true", "probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Predictions file is missing columns: {sorted(missing)}"
        )
    if predictions[ID_COLUMN].duplicated().any():
        raise ValueError("Predictions file contains duplicate SK_ID_CURR values.")

    raw = pd.read_csv(data_path)
    if raw[ID_COLUMN].duplicated().any():
        raise ValueError("Raw application file contains duplicate SK_ID_CURR values.")
    raw = raw.set_index(ID_COLUMN)
    test_ids = predictions[ID_COLUMN].astype(int).tolist()
    missing_ids = [value for value in test_ids if value not in raw.index]
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} prediction IDs are absent from the raw data."
        )
    test_frame = raw.loc[test_ids].reset_index()
    observed = pd.to_numeric(
        test_frame[TARGET_COLUMN], errors="raise"
    ).to_numpy(dtype=int)
    expected = predictions["y_true"].to_numpy(dtype=int)
    if not np.array_equal(observed, expected):
        raise ValueError("TARGET values do not match y_true in predictions.")
    return test_frame, predictions


def _top_feature_rows(importance: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(row.rank),
            "feature": str(row.feature),
            "mean_abs_shap": float(row.mean_abs_shap),
        }
        for row in importance.itertuples(index=False)
    ]


def _render_index(
    output_path: Path,
    importance: pd.DataFrame,
    metadata: dict[str, Any],
    local_examples: list[dict[str, Any]],
    top_n: int = 20,
) -> None:
    table_rows = "\n".join(
        "<tr>"
        f"<td>{int(row.rank)}</td>"
        f"<td>{html.escape(str(row.feature))}</td>"
        f"<td>{float(row.mean_abs_shap):.6f}</td>"
        "</tr>"
        for row in importance.head(top_n).itertuples(index=False)
    )
    local_blocks = "\n".join(
        "<section>"
        f"<h2>Application {html.escape(str(example['application_id']))}</h2>"
        f"<p>Probability: {float(example['probability']):.6f}; "
        f"base probability: {float(example['base_probability']):.6f}</p>"
        f"<p>{html.escape(str(example['summary']))}</p>"
        f"<img src=\"{html.escape(example['figure'])}\" "
        f"alt=\"Waterfall for {html.escape(str(example['application_id']))}\">"
        "</section>"
        for example in local_examples
    )
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Credit-risk SHAP report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; color: #222; }}
    table {{ border-collapse: collapse; margin: 1rem 0 2rem; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 0.45rem 0.8rem; text-align: left; }}
    th {{ background: #f4f6f8; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    section {{ margin: 2rem 0; }}
    code {{ background: #f4f6f8; padding: 0.1rem 0.3rem; }}
  </style>
</head>
<body>
  <h1>Credit-risk SHAP report</h1>
  <p>Model <code>{html.escape(metadata['model']['model_name'])}</code>
     version <code>{html.escape(metadata['model']['model_version'])}</code>.</p>
  <p>Method: native XGBoost TreeSHAP in log-odds. Positive values increase
     <code>P(TARGET=1)</code>; values are not causal and are not calibrated
     probabilities.</p>
  <h2>Global bar chart</h2>
  <img src="bar_top20.png" alt="Global SHAP bar chart">
  <h2>Beeswarm</h2>
  <img src="beeswarm_top20.png" alt="SHAP beeswarm">
  <h2>Top raw features</h2>
  <table>
    <thead><tr><th>Rank</th><th>Feature</th><th>Mean |SHAP|</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <h2>Local examples</h2>
  {local_blocks}
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def generate_report(
    model_dir: str | Path | None = None,
    data_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    importance_output: str | Path | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    local_ids: Sequence[int] = DEFAULT_LOCAL_IDS,
) -> dict[str, Any]:
    resolved_model_dir = (
        Path(model_dir).resolve()
        if model_dir is not None
        else DEFAULT_MODEL_DIR.resolve()
    )
    resolved_data = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
    resolved_predictions = (
        Path(predictions_path)
        if predictions_path is not None
        else resolved_model_dir / "test_predictions.csv"
    )
    resolved_output = (
        Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    )
    resolved_importance = (
        Path(importance_output)
        if importance_output is not None
        else resolved_model_dir / "global_importance.json"
    )
    resolved_output.mkdir(parents=True, exist_ok=True)

    print(f"Loading model and test predictions from {resolved_model_dir}")
    explainer = get_explainer(resolved_model_dir)
    test_frame, predictions = _load_test_frame(
        resolved_data, resolved_predictions
    )
    transformed = explainer.transform(test_frame)
    probabilities = explainer.predict_probability(transformed)
    stored_probabilities = predictions["probability"].to_numpy(dtype=np.float64)
    max_probability_difference = float(
        np.max(np.abs(probabilities - stored_probabilities))
    )
    if max_probability_difference > PROBABILITY_TOLERANCE:
        raise RuntimeError(
            "Prediction reconstruction failed: "
            f"max difference {max_probability_difference:.12g} exceeds "
            f"{PROBABILITY_TOLERANCE:.12g}."
        )

    targets = test_frame[TARGET_COLUMN].to_numpy(dtype=int)
    sampled_indices = _stratified_sample_indices(targets, sample_size, seed)
    sample_frame = test_frame.iloc[sampled_indices].reset_index(drop=True)
    sample_transformed = transformed[sampled_indices]
    print(
        f"Computing native TreeSHAP for {len(sampled_indices)} test rows "
        f"(seed={seed})"
    )
    batch = explainer.compute_shap_batch(sample_transformed)
    max_additivity_error = float(batch.additivity_errors().max())
    max_probability_reconstruction_error = float(
        batch.probability_reconstruction_errors().max()
    )
    if max_additivity_error > ADDITIVITY_TOLERANCE:
        raise RuntimeError(
            "SHAP additivity check failed: "
            f"{max_additivity_error:.12g} exceeds {ADDITIVITY_TOLERANCE:.12g}."
        )
    if max_probability_reconstruction_error > PROBABILITY_TOLERANCE:
        raise RuntimeError(
            "SHAP probability reconstruction failed: "
            f"{max_probability_reconstruction_error:.12g} exceeds "
            f"{PROBABILITY_TOLERANCE:.12g}."
        )

    importance = explainer.global_importance(batch.shap_values)
    raw_names, raw_shap_values = explainer.aggregate_shap_by_raw_feature(
        batch.shap_values
    )

    print("Writing global importance and figures")
    _write_json(
        resolved_importance,
        {
            "generated_at": _now(),
            "method": "tree_shap",
            "output_space": "log_odds",
            "model_name": explainer.model_name,
            "model_version": explainer.model_version,
            "model_hash": explainer.model_hash,
            "model_dir": str(resolved_model_dir),
            "source_split": "test",
            "source_data": str(resolved_data),
            "source_predictions": str(resolved_predictions),
            "sample_size": int(len(sampled_indices)),
            "sample_seed": int(seed),
            "units": "log_odds",
            "aggregation": (
                "Per row, sum SHAP values across transformed one-hot/missing "
                "indicator columns for each raw feature, then average absolute "
                "values across sampled rows."
            ),
            "missing_indicators_merged_into_parent": True,
            "n_raw_features": int(len(raw_names)),
            "n_transformed_features": int(batch.shap_values.shape[1]),
            "base_value": float(batch.base_values[0]),
            "base_probability": float(batch.base_probability),
            "max_abs_additivity_error_sample": max_additivity_error,
            "max_abs_probability_reconstruction_error_sample": (
                max_probability_reconstruction_error
            ),
            "max_abs_probability_difference_full_test": (
                max_probability_difference
            ),
            "features": _top_feature_rows(importance),
        },
    )

    plot_global_bar(importance, resolved_output / "bar_top20.png", top_n=20)
    plot_beeswarm(
        raw_names,
        raw_shap_values,
        sample_frame,
        resolved_output / "beeswarm_top20.png",
        max_display=20,
        seed=seed,
    )

    contribution_rows: list[dict[str, Any]] = []
    for row_index, source_index in enumerate(sampled_indices):
        explanation = explainer.explanation_from_batch(
            batch,
            row_index=row_index,
            transformed=sample_transformed,
            top_k=20,
            application_id=int(test_frame.iloc[int(source_index)][ID_COLUMN]),
        )
        for rank, feature in enumerate(explanation["top_features"], start=1):
            contribution_rows.append(
                {
                    "application_id": explanation["application_id"],
                    "y_true": int(test_frame.iloc[int(source_index)][TARGET_COLUMN]),
                    "probability": explanation["probability"],
                    "base_probability": explanation["base_probability"],
                    "rank": rank,
                    "feature": feature["feature"],
                    "raw_feature": feature["raw_feature"],
                    "transformed_feature": feature["transformed_feature"],
                    "value": feature["value"],
                    "shap_value": feature["shap_value"],
                    "effect": feature["effect"],
                }
            )
    pd.DataFrame(contribution_rows).to_csv(
        resolved_output / "top_contributions.csv",
        index=False,
        float_format="%.17g",
    )

    local_examples: list[dict[str, Any]] = []
    test_ids = test_frame[ID_COLUMN].to_numpy(dtype=np.int64)
    for application_id in local_ids:
        matches = np.flatnonzero(test_ids == int(application_id))
        if len(matches) != 1:
            raise ExplanationError(
                f"Application {application_id} was not found exactly once "
                "in the test split."
            )
        row_index = int(matches[0])
        row_frame = test_frame.iloc[[row_index]]
        row_transformed = transformed[row_index : row_index + 1]
        row_batch = explainer.compute_shap_batch(row_transformed)
        explanation = explainer.explanation_from_batch(
            row_batch,
            row_index=0,
            transformed=row_transformed,
            top_k=15,
            application_id=int(application_id),
        )
        figure_name = f"waterfall_{application_id}.png"
        plot_local_waterfall(
            explanation,
            resolved_output / figure_name,
            max_display=15,
        )
        local_examples.append(
            {
                "application_id": int(application_id),
                "y_true": int(row_frame[TARGET_COLUMN].iloc[0]),
                "probability": float(explanation["probability"]),
                "base_probability": float(explanation["base_probability"]),
                "summary": str(explanation["summary"]),
                "figure": figure_name,
                "top_features": explanation["top_features"],
            }
        )

    evaluation = _read_json(resolved_model_dir / "evaluation_results.json")
    training_metadata = _read_json(resolved_model_dir / "training_metadata.json")
    scale_pos_weight = evaluation.get(
        "scale_pos_weight", training_metadata.get("scale_pos_weight")
    )
    metadata = {
        "generated_at": _now(),
        "method": "tree_shap_native_xgboost",
        "output_space": "log_odds",
        "model": {
            "model_name": explainer.model_name,
            "model_version": explainer.model_version,
            "model_hash": explainer.model_hash,
            "model_dir": str(resolved_model_dir),
        },
        "data": {
            "source_data": str(resolved_data),
            "source_predictions": str(resolved_predictions),
            "source_split": "test",
            "test_rows": int(len(test_frame)),
            "sample_size": int(len(sampled_indices)),
            "sample_seed": int(seed),
            "class_counts_sample": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(
                        sample_frame[TARGET_COLUMN].to_numpy(dtype=int),
                        return_counts=True,
                    )
                )
            },
        },
        "base_value": float(batch.base_values[0]),
        "base_probability": float(batch.base_probability),
        "scale_pos_weight": scale_pos_weight,
        "base_probability_note": (
            "The base probability reflects the training scale_pos_weight, "
            "not the population default rate, and is not calibrated."
        ),
        "feature_scope": {
            "n_raw_features": int(len(raw_names)),
            "n_transformed_features": int(batch.shap_values.shape[1]),
            "missing_indicators_merged_into_parent": True,
            "local_feature_names": (
                "Local top_features retain transformed one-hot and "
                "[missing] indicator names."
            ),
        },
        "aggregation": (
            "Per row, sum SHAP values across transformed columns for each raw "
            "feature, then average absolute values across sampled rows."
        ),
        "errors": {
            "max_abs_additivity_error_sample": max_additivity_error,
            "additivity_tolerance": ADDITIVITY_TOLERANCE,
            "max_abs_probability_reconstruction_error_sample": (
                max_probability_reconstruction_error
            ),
            "probability_tolerance": PROBABILITY_TOLERANCE,
            "max_abs_probability_difference_full_test": (
                max_probability_difference
            ),
        },
        "local_examples": local_examples,
        "artifacts": {
            "global_importance": str(resolved_importance),
            "top_contributions_csv": str(
                resolved_output / "top_contributions.csv"
            ),
            "figures": [
                str(resolved_output / "bar_top20.png"),
                str(resolved_output / "beeswarm_top20.png"),
                *[
                    str(resolved_output / example["figure"])
                    for example in local_examples
                ],
            ],
            "index_html": str(resolved_output / "index.html"),
        },
    }
    _render_index(
        resolved_output / "index.html",
        importance,
        metadata,
        local_examples,
        top_n=20,
    )

    artifact_paths = [
        resolved_importance,
        resolved_output / "top_contributions.csv",
        resolved_output / "bar_top20.png",
        resolved_output / "beeswarm_top20.png",
        *[
            resolved_output / example["figure"]
            for example in local_examples
        ],
        resolved_output / "index.html",
    ]
    metadata["artifacts"]["files"] = [
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in artifact_paths
    ]
    _write_json(resolved_output / "shap_metadata.json", metadata)

    print(f"Global importance: {resolved_importance.resolve()}")
    print(f"Figures and report: {resolved_output.resolve()}")
    print(f"Max sample additivity error: {max_additivity_error:.12g}")
    print(
        "Max sample probability reconstruction error: "
        f"{max_probability_reconstruction_error:.12g}"
    )
    print(
        "Max full-test probability difference: "
        f"{max_probability_difference:.12g}"
    )
    print(f"Base value: {float(batch.base_values[0]):.17g}")
    print(f"Base probability: {float(batch.base_probability):.17g}")
    print("Top raw features:")
    for row in importance.head(10).itertuples(index=False):
        print(f"  {int(row.rank):>2}. {row.feature}: {row.mean_abs_shap:.6f}")

    return {
        "global_importance_path": str(resolved_importance),
        "output_dir": str(resolved_output),
        "sample_size": int(len(sampled_indices)),
        "max_abs_additivity_error_sample": max_additivity_error,
        "max_abs_probability_reconstruction_error_sample": (
            max_probability_reconstruction_error
        ),
        "max_abs_probability_difference_full_test": max_probability_difference,
        "base_value": float(batch.base_values[0]),
        "base_probability": float(batch.base_probability),
        "top_features": _top_feature_rows(importance.head(10)),
        "local_examples": local_examples,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate native XGBoost TreeSHAP figures, global importance, "
            "and a static HTML report."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--importance-output", type=Path, default=None)
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--local-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_LOCAL_IDS),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    generate_report(
        model_dir=arguments.model_dir,
        data_path=arguments.data,
        predictions_path=arguments.predictions,
        output_dir=arguments.output_dir,
        importance_output=arguments.importance_output,
        sample_size=arguments.sample_size,
        seed=arguments.seed,
        local_ids=tuple(arguments.local_ids),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
