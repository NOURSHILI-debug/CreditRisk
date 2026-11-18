"""Validate schema-v1 application events from raw Kafka bytes."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime
from typing import Any

SCHEMA_VERSION = "schema-v1"

REQUIRED_FIELDS = ("event_id", "event_timestamp", "schema_version", "SK_ID_CURR")

STRING_FIELDS = (
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
)
NUMBER_FIELDS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_FAM_MEMBERS",
)
INTEGER_FIELDS = ("DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_CHILDREN")
FEATURE_FIELDS = (*STRING_FIELDS, *NUMBER_FIELDS, *INTEGER_FIELDS)

ALLOWED_FIELDS = set(REQUIRED_FIELDS) | set(FEATURE_FIELDS)


class ValidationResult(dict):
    @property
    def valid(self) -> bool:
        return self["valid"]

    @property
    def event(self) -> dict[str, Any] | None:
        return self["event"]

    @property
    def errors(self) -> list[str]:
        return self["errors"]


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_uuid(value: Any) -> bool:
    if not _is_non_empty_string(value):
        return False
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _is_iso_timestamp(value: Any) -> bool:
    if not _is_non_empty_string(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _invalid(errors: list[str]) -> ValidationResult:
    return ValidationResult(valid=False, event=None, errors=errors)


def validate_event(raw_bytes: Any) -> dict[str, Any]:
    try:
        event = json.loads(raw_bytes)
    except Exception as exc:
        return _invalid([f"invalid JSON: {exc}"])

    if not isinstance(event, dict):
        return _invalid(["event must be a JSON object"])

    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in event or event[field] is None:
            errors.append(f"missing required field: {field}")

    if "event_id" in event and event["event_id"] is not None:
        if not _is_uuid(event["event_id"]):
            errors.append("field event_id must be a UUID string")

    if "event_timestamp" in event and event["event_timestamp"] is not None:
        if not _is_iso_timestamp(event["event_timestamp"]):
            errors.append("field event_timestamp must be an ISO 8601 timestamp")

    if "schema_version" in event and event["schema_version"] is not None:
        if event["schema_version"] != SCHEMA_VERSION:
            errors.append("field schema_version must be 'schema-v1'")

    if "SK_ID_CURR" in event and event["SK_ID_CURR"] is not None:
        if not _is_integer(event["SK_ID_CURR"]):
            errors.append("field SK_ID_CURR must be an integer")
        elif event["SK_ID_CURR"] <= 0:
            errors.append("field SK_ID_CURR must be a positive integer")

    if "TARGET" in event:
        errors.append("field TARGET must not be present")

    for field in event:
        if field != "TARGET" and field not in ALLOWED_FIELDS:
            errors.append(f"unknown field: {field}")

    for field in STRING_FIELDS:
        value = event.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"field {field} must be a string or null")

    for field in NUMBER_FIELDS:
        value = event.get(field)
        if value is not None and not _is_number(value):
            errors.append(f"field {field} must be a number or null")

    for field in INTEGER_FIELDS:
        value = event.get(field)
        if value is not None and not _is_integer(value):
            errors.append(f"field {field} must be an integer or null")

    if errors:
        return _invalid(errors)
    return ValidationResult(valid=True, event=event, errors=[])
