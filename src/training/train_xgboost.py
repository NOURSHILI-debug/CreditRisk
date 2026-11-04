from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from processing.preprocessing import (  # noqa: E402
    DAYS_EMPLOYED_SENTINEL,
    ID_COLUMN,
    TARGET_COLUMN,
    build_preprocessor,
    get_post_transform_feature_names,
    split_feature_columns,
)
from project_paths import DEFAULT_MODEL_DIR  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR
MODEL_NAME = "XGBoost"
MODEL_VERSION = "baseline-v2-local"
SEED = 42
THRESHOLD = 0.5
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "eval_metric": ["auc", "aucpr"],
    "tree_method": "hist",
    "n_jobs": -1,
    "device": "cpu",
}
BASELINE_REFERENCE = {
    "source": "previous Colab baseline (context only, not a tuning target)",
    "test": {
        "roc_auc": 0.7571,
        "pr_auc": 0.2386,
        "precision": 0.1943,
        "recall": 0.5835,
        "f1": 0.2915,
    },
}


class ProgressLogger(TrainingCallback):
    def __init__(self, log: logging.Logger, total_rounds: int, every: int = 50) -> None:
        self.log = log
        self.total_rounds = total_rounds
        self.every = every

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        iteration = epoch + 1
        if iteration % self.every != 0 and iteration != self.total_rounds:
            return False
        parts: list[str] = []
        for data_name, metrics in evals_log.items():
            for metric_name, values in metrics.items():
                if values:
                    parts.append(f"{data_name}-{metric_name}={values[-1]:.6f}")
        self.log.info(
            "iteration %d/%d %s",
            iteration,
            self.total_rounds,
            " ".join(parts),
        )
        return False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("train_xgboost")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(
        output_dir / "train_xgboost.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    log.propagate = False
    return log


def class_counts(values: pd.Series | np.ndarray) -> dict[str, int]:
    array = np.asarray(values)
    return {
        "0": int((array == 0).sum()),
        "1": int((array == 1).sum()),
    }


def split_summary(values: pd.Series | np.ndarray) -> dict[str, object]:
    array = np.asarray(values)
    counts = class_counts(array)
    return {
        "n_samples": int(array.size),
        "class_counts": counts,
        "positive_rate": float(array.mean()) if array.size else None,
    }


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, object]:
    truth = np.asarray(y_true).astype(int)
    probabilities = np.asarray(y_prob, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "threshold": float(threshold),
        "n_samples": int(truth.size),
        "class_counts": class_counts(truth),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "pr_auc": float(average_precision_score(truth, probabilities)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "confusion_matrix": {
            "labels": [0, 1],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": matrix.tolist(),
        },
        "predicted_positive_rate": float(predictions.mean()),
        "observed_positive_rate": float(truth.mean()),
    }


def compare_to_baseline(test_metrics: dict[str, object]) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    findings: list[str] = []
    tolerance = 0.01
    for metric_name, baseline_value in BASELINE_REFERENCE["test"].items():
        actual_value = float(test_metrics[metric_name])
        delta = actual_value - float(baseline_value)
        exceeds = abs(delta) > tolerance
        comparisons[metric_name] = {
            "baseline": float(baseline_value),
            "actual": actual_value,
            "delta": delta,
            "abs_delta": abs(delta),
            "exceeds_tolerance": bool(exceeds),
            "tolerance": tolerance,
        }
        if exceeds:
            findings.append(
                f"test {metric_name} differs from the Colab baseline by "
                f"{delta:+.4f} (absolute tolerance {tolerance:.2f})."
            )
    return {
        "tolerance": tolerance,
        "metrics": comparisons,
        "findings": findings,
    }


def artifact_record(path: Path) -> dict[str, object]:
    try:
        display_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(path.resolve())
    return {
        "path": display_path,
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
    }


def log_split(log: logging.Logger, name: str, y: pd.Series) -> None:
    summary = split_summary(y)
    log.info(
        "%s split: n=%d, class_counts=%s, positive_rate=%.6f",
        name,
        summary["n_samples"],
        summary["class_counts"],
        summary["positive_rate"],
    )


def train(
    data_path: Path = DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir)
    training_date = utc_now()
    log.info("Training %s (%s)", MODEL_NAME, MODEL_VERSION)
    log.info("Data path: %s", data_path)
    log.info("Output directory: %s", output_dir)
    log.info("Library versions: %s", library_versions())

    data = pd.read_csv(data_path)
    log.info("Loaded data shape: %s", data.shape)
    if TARGET_COLUMN not in data.columns or ID_COLUMN not in data.columns:
        raise ValueError(
            f"Expected {ID_COLUMN!r} and {TARGET_COLUMN!r} in {data_path}."
        )

    feature_columns, numeric_columns, categorical_columns = split_feature_columns(data)
    log.info(
        "Feature columns: %d (numeric=%d, categorical=%d)",
        len(feature_columns),
        len(numeric_columns),
        len(categorical_columns),
    )

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)
    X_train_full, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=SPLIT_RATIOS["validation"] + SPLIT_RATIOS["test"],
        stratify=y,
        random_state=SEED,
    )
    X_val_full, X_test_full, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        stratify=y_holdout,
        random_state=SEED,
    )
    X_train = X_train_full[feature_columns]
    X_val = X_val_full[feature_columns]
    X_test = X_test_full[feature_columns]
    log_split(log, "train", y_train)
    log_split(log, "validation", y_val)
    log_split(log, "test", y_test)

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    log.info("Fitting preprocessor on TRAIN ONLY.")
    preprocessor.fit(X_train)
    post_transform_features = get_post_transform_feature_names(preprocessor)
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    log.info(
        "Post-transform feature count: %d (train matrix=%s)",
        len(post_transform_features),
        getattr(X_train_t, "shape", None),
    )

    train_counts = class_counts(y_train)
    scale_pos_weight = train_counts["0"] / train_counts["1"]
    log.info(
        "scale_pos_weight = n_neg/n_pos on TRAIN ONLY = %d/%d = %.10f",
        train_counts["0"],
        train_counts["1"],
        scale_pos_weight,
    )

    params = dict(XGBOOST_PARAMS)
    params["scale_pos_weight"] = float(scale_pos_weight)
    model = XGBClassifier(
        **params,
        callbacks=[ProgressLogger(log, total_rounds=params["n_estimators"])],
    )

    log.info("Starting XGBoost fit with params: %s", params)
    fit_started = time.perf_counter()
    model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
    )
    fit_seconds = time.perf_counter() - fit_started
    log.info("XGBoost fit completed in %.2f seconds.", fit_seconds)

    model_path = output_dir / "xgboost_baseline.json"
    preprocessor_path = output_dir / "preprocessor.pkl"
    feature_columns_path = output_dir / "feature_columns.json"
    metadata_path = output_dir / "preprocessor_metadata.json"
    model_metadata_path = output_dir / "model_metadata.json"
    validation_predictions_path = output_dir / "validation_predictions.csv"
    test_predictions_path = output_dir / "test_predictions.csv"
    training_metadata_path = output_dir / "training_metadata.json"
    evaluation_path = output_dir / "evaluation_results.json"

    model.get_booster().save_model(str(model_path))
    joblib.dump(preprocessor, preprocessor_path)
    feature_columns_path.write_text(
        json.dumps(feature_columns, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "input_features": feature_columns,
                "numeric_features": numeric_columns,
                "categorical_features": categorical_columns,
                "post_transform_features": post_transform_features,
                "input_feature_count": len(feature_columns),
                "post_transform_feature_count": len(post_transform_features),
                "model_feature_count": int(model.n_features_in_),
                "days_employed_sentinel_replaced_with": "NaN",
                "days_employed_sentinel_value": DAYS_EMPLOYED_SENTINEL,
                "numeric_imputation": "median with missing indicators",
                "categorical_imputation": "most frequent",
                "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
                "scaler": None,
                "sklearn_version": sklearn.__version__,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    model_metadata_path.write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "model_hash": sha256_file(model_path),
                "training_date": training_date,
                "feature_columns_count": len(feature_columns),
                "post_transform_feature_count": len(post_transform_features),
                "model_feature_count": int(model.n_features_in_),
                "target": {
                    "name": TARGET_COLUMN,
                    "positive_class": 1,
                    "positive_label": "repayment difficulty",
                    "probability_semantics": "P(TARGET=1)",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    val_probabilities = model.predict_proba(X_val_t)[:, 1]
    test_probabilities = model.predict_proba(X_test_t)[:, 1]
    validation_metrics = compute_metrics(y_val, val_probabilities)
    test_metrics = compute_metrics(y_test, test_probabilities)
    log.info("Validation metrics at threshold %.2f: %s", THRESHOLD, validation_metrics)
    log.info("Test metrics at threshold %.2f: %s", THRESHOLD, test_metrics)

    pd.DataFrame(
        {
            ID_COLUMN: X_val_full[ID_COLUMN].to_numpy(),
            "y_true": y_val.to_numpy(),
            "probability": val_probabilities,
            "prediction": (val_probabilities >= THRESHOLD).astype(int),
        }
    ).to_csv(
        validation_predictions_path, index=False, float_format="%.17g"
    )
    pd.DataFrame(
        {
            ID_COLUMN: X_test_full[ID_COLUMN].to_numpy(),
            "y_true": y_test.to_numpy(),
            "probability": test_probabilities,
            "prediction": (test_probabilities >= THRESHOLD).astype(int),
        }
    ).to_csv(
        test_predictions_path, index=False, float_format="%.17g"
    )

    model_artifact = artifact_record(model_path)
    preprocessor_artifact = artifact_record(preprocessor_path)
    feature_columns_artifact = artifact_record(feature_columns_path)
    metadata_artifact = artifact_record(metadata_path)
    model_metadata_artifact = artifact_record(model_metadata_path)
    validation_predictions_artifact = artifact_record(validation_predictions_path)
    test_predictions_artifact = artifact_record(test_predictions_path)
    baseline_comparison = compare_to_baseline(test_metrics)
    elapsed_seconds = time.perf_counter() - started

    evaluation = {
        "artifact_version": "1.0",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_hash": model_artifact["sha256"],
        "generated_at": utc_now(),
        "training_date": training_date,
        "target": {
            "name": TARGET_COLUMN,
            "positive_class": 1,
            "positive_label": "repayment difficulty",
            "probability_semantics": "P(TARGET=1)",
        },
        "seed": SEED,
        "threshold": THRESHOLD,
        "split_protocol": {
            "method": "stratified train/validation/test",
            "ratios": SPLIT_RATIOS,
            "random_state": SEED,
            "fit_preprocessor_on": "train only",
            "notes": "First split 70/30; second split 50/50 of the 30% holdout.",
        },
        "splits": {
            "train": split_summary(y_train),
            "validation": split_summary(y_val),
            "test": split_summary(y_test),
        },
        "scale_pos_weight": float(scale_pos_weight),
        "scale_pos_weight_formula": "n_neg/n_pos computed on TRAIN ONLY",
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "metric_definitions": {
            "threshold": "Probability cutoff used to convert P(TARGET=1) into a binary prediction.",
            "roc_auc": "Area under the receiver operating characteristic curve.",
            "pr_auc": "Area under the precision-recall curve (average_precision_score).",
            "precision": "TP / (TP + FP) at the stated threshold; zero_division=0.",
            "recall": "TP / (TP + FN) at the stated threshold; zero_division=0.",
            "f1": "Harmonic mean of precision and recall at the stated threshold.",
            "confusion_matrix": "Confusion matrix with labels [0, 1] at the stated threshold.",
            "predicted_positive_rate": "Share of rows predicted positive at the stated threshold.",
            "observed_positive_rate": "Share of rows with TARGET=1.",
        },
        "artifacts": {
            "model": {
                **model_artifact,
                "format": "xgboost native JSON",
            },
            "preprocessor": {
                **preprocessor_artifact,
                "format": "joblib sklearn Pipeline",
            },
            "feature_columns": {
                **feature_columns_artifact,
                "format": "JSON list of raw input feature names",
            },
            "preprocessor_metadata": metadata_artifact,
            "model_metadata": model_metadata_artifact,
            "validation_predictions": validation_predictions_artifact,
            "test_predictions": test_predictions_artifact,
        },
        "library_versions": library_versions(),
        "training_elapsed_seconds": float(elapsed_seconds),
        "baseline_reference": BASELINE_REFERENCE,
        "baseline_comparison": baseline_comparison,
    }
    evaluation_path.write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    training_metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "training_date": training_date,
        "params": params,
        "scale_pos_weight": float(scale_pos_weight),
        "feature_columns_count": len(feature_columns),
        "post_transform_feature_count": len(post_transform_features),
        "model_feature_count": int(model.n_features_in_),
        "fit_seconds": float(fit_seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "model_metadata_artifact": model_metadata_artifact,
        "evaluation_artifact": artifact_record(evaluation_path),
        "library_versions": library_versions(),
    }
    training_metadata_path.write_text(
        json.dumps(training_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    log.info("Wrote model: %s", model_path)
    log.info("Wrote preprocessor: %s", preprocessor_path)
    log.info("Wrote feature columns: %s", feature_columns_path)
    log.info("Wrote evaluation artifact: %s", evaluation_path)
    log.info("Baseline findings: %s", baseline_comparison["findings"] or "none")
    log.info("Total elapsed time: %.2f seconds.", elapsed_seconds)
    print(f"Total elapsed time: {elapsed_seconds:.2f} seconds")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local XGBoost baseline.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    train(data_path=args.data_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
