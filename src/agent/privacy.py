"""Local applicant-data minimization for agent requests."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


# The API contract is the existing source allowlist. The agent exposes at most
# one explicitly requested field, and never sends applicant values to an LLM.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "NAME_CONTRACT_TYPE": ("contract type", "name_contract_type"),
    "NAME_INCOME_TYPE": ("income type", "name_income_type"),
    "NAME_EDUCATION_TYPE": ("education type", "education", "name_education_type"),
    "NAME_FAMILY_STATUS": ("family status", "marital status", "name_family_status"),
    "NAME_HOUSING_TYPE": ("housing type", "name_housing_type"),
    "AMT_INCOME_TOTAL": ("total income", "income amount", "amt_income_total"),
    "AMT_CREDIT": ("credit amount", "credit requested", "amt_credit"),
    "AMT_ANNUITY": ("annuity", "installment amount", "amt_annuity"),
    "AMT_GOODS_PRICE": ("goods price", "amt_goods_price"),
    "DAYS_BIRTH": ("days_birth", "age in days"),
    "DAYS_EMPLOYED": ("days employed", "days_employed"),
    "CNT_CHILDREN": ("children count", "number of children", "cnt_children"),
    "CNT_FAM_MEMBERS": ("family members count", "number of family members", "cnt_fam_members"),
}
_APPLICATION_ID_PATTERN = re.compile(
    r"\b(?:application|app|applicant)"
    r"(?:\s+(?:details?|information|record|data))?"
    r"(?:\s+(?:is|for))?(?:\s+id)?\s*(?:number\s*|no\.?\s*)?[:#]?\s*"
    r"([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?!\w|\.\d)",
    re.IGNORECASE,
)


def redact_application_ids(question: str) -> str:
    """Remove parsed application IDs before sending a question to an LLM."""
    return _APPLICATION_ID_PATTERN.sub(
        lambda match: match.group(0)[: match.start(1) - match.start()]
        + "[redacted]",
        question,
    )


def requested_application_field(question: str) -> str | None:
    """Return one unambiguous API-allowlisted field explicitly requested."""
    normalized = question.casefold()
    matches = {
        field
        for field, aliases in _FIELD_ALIASES.items()
        for alias in aliases
        if re.search(rf"(?<![\w]){re.escape(alias.casefold())}(?![\w])", normalized)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def minimize_application_lookup(
    question: str, result: Mapping[str, Any]
) -> dict[str, Any]:
    """Select at most one requested field for local, deterministic rendering."""
    status = result.get("status")
    if status != "found":
        safe: dict[str, Any] = {"status": status or "unavailable"}
        message = result.get("message")
        if isinstance(message, str):
            safe["message"] = message
        return safe

    field = requested_application_field(question)
    if field is None:
        return {
            "status": "not_authorized",
            "message": (
                "A full application record is not returned. Ask for one specific "
                "allowlisted attribute."
            ),
        }
    fields = result.get("fields")
    if not isinstance(fields, Mapping) or field not in fields:
        return {
            "status": "unavailable",
            "field": field,
            "message": "The requested attribute is unavailable in the source record.",
        }
    return {"status": "found", "field": field, "value": fields[field]}


def redact_applicant_identifier_fields(value: Any) -> Any:
    """Recursively remove identifier keys from provider-facing payloads."""
    if isinstance(value, Mapping):
        return {
            key: redact_applicant_identifier_fields(item)
            for key, item in value.items()
            if str(key).casefold()
            not in {"application_id", "applicant_id", "sk_id_curr"}
        }
    if isinstance(value, list):
        return [redact_applicant_identifier_fields(item) for item in value]
    return value
