from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POSITIVE_COLOR = "#c0392b"
NEGATIVE_COLOR = "#2471a3"
NEUTRAL_COLOR = "#7f8c8d"


def _feature_plot_matrix(
    raw_names: Sequence[str],
    frame: pd.DataFrame,
) -> np.ndarray:
    matrix = np.full((len(frame), len(raw_names)), np.nan, dtype=np.float64)
    for column_index, raw_name in enumerate(raw_names):
        if raw_name not in frame.columns:
            continue
        column = frame[raw_name]
        if pd.api.types.is_numeric_dtype(column):
            matrix[:, column_index] = pd.to_numeric(
                column, errors="coerce"
            ).to_numpy(dtype=np.float64)
        else:
            codes = pd.Categorical(column.astype("string")).codes
            matrix[:, column_index] = np.where(codes >= 0, codes, np.nan)
    return matrix


def _normalized_feature_values(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, 0.5, dtype=np.float64)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high) or high == low:
        return np.full(values.shape, 0.5, dtype=np.float64)
    normalized = (values - low) / (high - low)
    return np.where(np.isfinite(normalized), normalized, 0.5)


def plot_beeswarm(
    raw_names: Sequence[str],
    raw_shap_values: np.ndarray,
    frame: pd.DataFrame,
    output_path: Path,
    max_display: int = 20,
    seed: int = 42,
) -> None:
    values = np.asarray(raw_shap_values, dtype=np.float64)
    feature_values = _feature_plot_matrix(raw_names, frame)
    mean_abs = np.abs(values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:max_display]
    if top_indices.size == 0:
        raise ValueError("No features available for the beeswarm plot.")

    rng = np.random.default_rng(seed)
    figure, axis = plt.subplots(figsize=(10, max(6, 0.38 * len(top_indices))))
    y_positions = np.arange(len(top_indices))[::-1]

    for y_position, feature_index in zip(y_positions, top_indices):
        shap_column = values[:, feature_index]
        valid = np.isfinite(shap_column)
        if not valid.any():
            continue
        colors = _normalized_feature_values(feature_values[valid, feature_index])
        jitter = rng.uniform(-0.28, 0.28, size=int(valid.sum()))
        axis.scatter(
            shap_column[valid],
            y_position + jitter,
            c=colors,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=9,
            alpha=0.7,
            linewidths=0,
        )

    axis.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_yticks(y_positions)
    axis.set_yticklabels([raw_names[index] for index in top_indices])
    axis.set_xlabel("SHAP value (log-odds; positive increases P(TARGET=1))")
    axis.set_title("Global feature attribution (test sample)", loc="left")
    axis.grid(axis="x", alpha=0.2)
    figure.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
            cmap="coolwarm",
        ),
        ax=axis,
        fraction=0.025,
        pad=0.02,
        label="Feature value (low to high; ordinal for categorical)",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_global_bar(
    importance: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    data = importance.head(top_n).iloc[::-1].copy()
    if data.empty:
        raise ValueError("No features available for the global bar plot.")
    figure, axis = plt.subplots(figsize=(9, max(5, 0.36 * len(data))))
    axis.barh(data["feature"], data["mean_abs_shap"], color="#2c7fb8")
    axis.set_xlabel("Mean |SHAP value| (log-odds)")
    axis.set_title("Global importance by raw feature", loc="left")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_local_waterfall(
    explanation: Mapping[str, Any],
    output_path: Path,
    max_display: int = 15,
) -> None:
    top_features = list(explanation.get("top_features", []))[:max_display]
    base_value = float(explanation["base_value"])
    margin = float(explanation["prediction_margin"])
    used_total = sum(float(item["shap_value"]) for item in top_features)
    other_value = margin - base_value - used_total

    items = [
        (str(item["feature"]), float(item["shap_value"]))
        for item in top_features
    ]
    if abs(other_value) > 1e-12:
        items.append(("Other features", other_value))
    items.sort(key=lambda item: abs(item[1]), reverse=True)

    labels = [item[0] for item in items]
    contributions = np.asarray([item[1] for item in items], dtype=np.float64)
    starts = np.zeros(len(contributions), dtype=np.float64)
    running = base_value
    for index, contribution in enumerate(contributions):
        starts[index] = running
        running += contribution

    colors = [
        POSITIVE_COLOR
        if contribution > 0
        else NEGATIVE_COLOR
        if contribution < 0
        else NEUTRAL_COLOR
        for contribution in contributions
    ]
    figure, axis = plt.subplots(figsize=(10, max(5, 0.38 * len(items))))
    axis.barh(labels, contributions, left=starts, color=colors, alpha=0.9)
    axis.axvline(base_value, color="#555555", linestyle="--", linewidth=0.8)
    axis.axvline(margin, color="#111111", linestyle=":", linewidth=0.9)
    axis.set_xlabel("Cumulative log-odds contribution")
    title_id = explanation.get("application_id")
    probability = float(explanation["probability"])
    base_probability = float(explanation["base_probability"])
    axis.set_title(
        f"Local explanation for application {title_id} "
        f"(P={probability:.4f}, base P={base_probability:.4f})",
        loc="left",
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
