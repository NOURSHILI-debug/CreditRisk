"""Application field allowlist derived from dashboard usage.

Sources:
- ``dashboard/pages/overview.py`` category and numeric columns.
- ``dashboard/pages/application_explorer.py`` detail groups and table fields.

``TARGET`` and derived risk outputs are excluded to avoid returning outcome
labels or non-source prediction values from applicant browsing endpoints.
"""

from __future__ import annotations

APPLICATION_FIELDS: tuple[str, ...] = (
    "SK_ID_CURR",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
)

DEFAULT_APPLICATION_FIELDS: tuple[str, ...] = APPLICATION_FIELDS
FILTERABLE_APPLICATION_FIELDS: tuple[str, ...] = (
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
)
MAX_APPLICATION_LIMIT = 500
DEFAULT_APPLICATION_LIMIT = 100


def parse_fields(value: str | None) -> list[str]:
    """Parse and validate the comma-separated ``fields`` query parameter."""
    if value is None or not value.strip():
        return list(DEFAULT_APPLICATION_FIELDS)

    requested = [part.strip() for part in value.split(",")]
    if any(not part for part in requested):
        raise ValueError("fields must not contain empty names.")
    unknown = [part for part in requested if part not in APPLICATION_FIELDS]
    if unknown:
        raise ValueError(
            "Unknown application fields: "
            + ", ".join(sorted(unknown))
            + ". Allowed fields: "
            + ", ".join(APPLICATION_FIELDS)
            + "."
        )

    ordered: list[str] = []
    for field in requested:
        if field not in ordered:
            ordered.append(field)
    return ordered
