from __future__ import annotations

import copy
import json
import uuid

import pytest

from ingestion.event_validator import validate_event


def _valid_event() -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "event_timestamp": "2026-11-11T10:15:30Z",
        "schema_version": "schema-v1",
        "SK_ID_CURR": 100002,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "AMT_INCOME_TOTAL": 202500.0,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "AMT_GOODS_PRICE": 351000.0,
        "DAYS_BIRTH": -9461,
        "DAYS_EMPLOYED": -637,
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 1.0,
    }


def _run(event: dict) -> dict:
    return validate_event(json.dumps(event).encode("utf-8"))


def _has_error(result: dict, text: str) -> bool:
    return any(text in error for error in result["errors"])


def test_valid_event_passes():
    event = _valid_event()

    result = _run(event)

    assert result["valid"] is True
    assert result["event"] == event
    assert result["errors"] == []
    assert result.valid is True
    assert result.event == event
    assert result.errors == []


def test_optional_feature_fields_may_be_null():
    event = _valid_event()
    event["AMT_CREDIT"] = None
    event["NAME_INCOME_TYPE"] = None
    event["DAYS_BIRTH"] = None

    result = _run(event)

    assert result["valid"] is True
    assert result["errors"] == []


@pytest.mark.parametrize(
    "field",
    ["event_id", "event_timestamp", "schema_version", "SK_ID_CURR"],
)
def test_missing_required_field_is_rejected(field):
    event = _valid_event()
    del event[field]

    result = _run(event)

    assert result["valid"] is False
    assert result["event"] is None
    assert _has_error(result, f"missing required field: {field}")


def test_wrong_sk_id_type_is_rejected():
    event = _valid_event()
    event["SK_ID_CURR"] = "100002"

    result = _run(event)

    assert result["valid"] is False
    assert _has_error(result, "SK_ID_CURR must be an integer")


def test_non_positive_sk_id_is_rejected():
    event = _valid_event()
    event["SK_ID_CURR"] = 0

    result = _run(event)

    assert result["valid"] is False
    assert _has_error(result, "SK_ID_CURR must be a positive integer")


def test_bad_timestamp_is_rejected():
    event = _valid_event()
    event["event_timestamp"] = "11/11/2026 10:15"

    result = _run(event)

    assert result["valid"] is False
    assert _has_error(result, "ISO 8601")


def test_wrong_schema_version_is_rejected():
    event = _valid_event()
    event["schema_version"] = "schema-v2"

    result = _run(event)

    assert result["valid"] is False
    assert _has_error(result, "schema_version")


def test_target_is_rejected():
    event = _valid_event()
    event["TARGET"] = 1

    result = _run(event)

    assert result["valid"] is False
    assert _has_error(result, "TARGET")


def test_unknown_field_is_rejected():
    event = _valid_event()
    event["UNKNOWN_FIELD"] = "x"

    result = _run(event)

    assert result["valid"] is False
    assert _has_error(result, "unknown field: UNKNOWN_FIELD")


def test_bad_event_id_is_rejected():
    event = _valid_event()
    event["event_id"] = "not-a-uuid"

    result = _run(event)

    assert result["valid"] is False
    assert _has_error(result, "event_id must be a UUID")


def test_wrong_feature_types_are_rejected():
    event = _valid_event()
    event["NAME_CONTRACT_TYPE"] = 5
    event["AMT_CREDIT"] = "406597.5"
    event["DAYS_BIRTH"] = -9461.0
    event["CNT_FAM_MEMBERS"] = True

    result = _run(event)

    assert result["valid"] is False
    assert _has_error(result, "NAME_CONTRACT_TYPE")
    assert _has_error(result, "AMT_CREDIT")
    assert _has_error(result, "DAYS_BIRTH")
    assert _has_error(result, "CNT_FAM_MEMBERS")


def test_non_object_json_is_rejected():
    result = validate_event(b"[]")

    assert result["valid"] is False
    assert result["event"] is None
    assert _has_error(result, "JSON object")


def test_garbage_bytes_do_not_crash():
    result = validate_event(b"\xff\xfe\x00not-json")

    assert result["valid"] is False
    assert result["event"] is None
    assert _has_error(result, "invalid JSON")


def test_original_event_is_not_mutated_by_validation():
    event = _valid_event()
    original = copy.deepcopy(event)

    _run(event)

    assert event == original
