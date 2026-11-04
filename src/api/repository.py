from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


class DuplicateApplicationIdError(ValueError):
    """Raised when the source data contains duplicate application IDs."""


class ApplicationRepository:
    """Load application features once and index them by SK_ID_CURR."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._frame: pd.DataFrame | None = None
        self._numeric_bounds: dict[str, tuple[float, float]] = {}
        self._error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._error = "Application data file is not available."
            return
        try:
            frame = pd.read_csv(self.path)
        except Exception as exc:
            self._error = f"Application data could not be read: {exc}"
            return

        if ID_COLUMN not in frame.columns:
            self._error = f"Application data is missing {ID_COLUMN}."
            return

        identifiers = pd.to_numeric(frame[ID_COLUMN], errors="coerce")
        if identifiers.isna().any():
            self._error = f"Application data contains invalid {ID_COLUMN} values."
            return
        frame[ID_COLUMN] = identifiers.astype("int64")
        duplicates = frame[ID_COLUMN].duplicated(keep=False)
        if duplicates.any():
            duplicate_values = (
                frame.loc[duplicates, ID_COLUMN].drop_duplicates().head(10).tolist()
            )
            raise DuplicateApplicationIdError(
                "Duplicate application IDs are not allowed: "
                f"{duplicate_values}"
            )

        self._numeric_bounds = self._compute_numeric_bounds(frame)
        self._frame = frame.set_index(ID_COLUMN, drop=False)

    @staticmethod
    def _compute_numeric_bounds(
        frame: pd.DataFrame,
    ) -> dict[str, tuple[float, float]]:
        bounds: dict[str, tuple[float, float]] = {}
        for column in frame.select_dtypes(include="number").columns:
            if column in {ID_COLUMN, TARGET_COLUMN}:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            bounds[str(column)] = (float(values.min()), float(values.max()))
        return bounds

    @property
    def available(self) -> bool:
        return self._frame is not None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def numeric_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self._numeric_bounds)

    def get_by_id(self, application_id: int) -> pd.Series | None:
        if self._frame is None:
            return None
        try:
            row = self._frame.loc[int(application_id)]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            raise DuplicateApplicationIdError(
                f"Multiple rows found for application {application_id}."
            )
        return row.drop(labels=[TARGET_COLUMN], errors="ignore").copy()

    def count(self) -> int:
        return 0 if self._frame is None else int(len(self._frame))

    def describe(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "row_count": self.count(),
            "has_numeric_bounds": bool(self._numeric_bounds),
            "error": self.error,
        }
