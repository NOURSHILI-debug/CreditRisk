from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = "models/local_v2"
DEFAULT_APPLICATION_DATA = "data/raw/application_train.csv"
DEFAULT_CORS_ORIGINS = ("http://localhost:8501", "http://127.0.0.1:8501")
DEFAULT_LOG_LEVEL = "INFO"


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _parse_cors_origins(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return DEFAULT_CORS_ORIGINS
    origins = tuple(
        origin.strip() for origin in value.split(",") if origin.strip()
    )
    return origins or DEFAULT_CORS_ORIGINS


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    application_data_path: Path
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    log_level: str = DEFAULT_LOG_LEVEL

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None
    ) -> "Settings":
        values = os.environ if env is None else env
        model_dir = values.get("CREDIT_RISK_MODEL_DIR", DEFAULT_MODEL_DIR)
        data_path = values.get(
            "CREDIT_RISK_APPLICATION_DATA", DEFAULT_APPLICATION_DATA
        )
        return cls(
            model_dir=_resolve_project_path(model_dir),
            application_data_path=_resolve_project_path(data_path),
            cors_origins=_parse_cors_origins(
                values.get("CREDIT_RISK_CORS_ORIGINS")
            ),
            log_level=values.get(
                "CREDIT_RISK_LOG_LEVEL", DEFAULT_LOG_LEVEL
            ).upper(),
        )
