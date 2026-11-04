"""Shared project paths and model-directory resolution.

Relative model directories (including the ``CREDIT_RISK_MODEL_DIR`` override)
always resolve against the project root, never the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "local_v2"
MODEL_DIR_ENV = "CREDIT_RISK_MODEL_DIR"


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    """Resolve a model directory against the project root.

    ``None`` uses ``CREDIT_RISK_MODEL_DIR`` when set, otherwise
    ``DEFAULT_MODEL_DIR``. Relative values resolve against ``PROJECT_ROOT``;
    absolute values are returned unchanged.
    """
    value: str | Path | None = model_dir
    if value is None:
        value = os.getenv(MODEL_DIR_ENV)
    if value is None:
        return DEFAULT_MODEL_DIR
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)
