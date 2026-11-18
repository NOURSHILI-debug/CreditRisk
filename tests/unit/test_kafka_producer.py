from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

from ingestion import kafka_producer as kp

REQUIRED_FIELDS = {"event_id", "event_timestamp", "schema_version", "SK_ID_CURR"}


def _row(**overrides):
    row = {
        "SK_ID_CURR": 100002,
        "TARGET": 1,
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
    row.update(overrides)
    return row


def _is_iso_utc(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


class FakeProducer:
    def __init__(self, fail_index: int | None = None):
        self.messages: list[dict] = []
        self.fail_index = fail_index
        self.flushed = False

    def produce(self, topic, key=None, value=None, on_delivery=None):
        index = len(self.messages)
        self.messages.append({"topic": topic, "key": key, "value": value})
        if index == self.fail_index:
            on_delivery(Exception("delivery failed"), None)
        else:
            on_delivery(None, object())

    def poll(self, timeout):
        return 0

    def flush(self):
        self.flushed = True


def test_build_event_has_required_metadata_and_feature_fields():
    event = kp.build_event(_row(), 0)

    assert REQUIRED_FIELDS <= event.keys()
    assert event["schema_version"] == "schema-v1"
    assert event["SK_ID_CURR"] == 100002
    assert isinstance(event["SK_ID_CURR"], int)
    for field in kp.FEATURE_FIELDS:
        assert field in event


def test_build_event_never_includes_target_or_unknown_fields():
    event = kp.build_event(_row(TARGET=1, UNKNOWN_FIELD="x"), 0)

    assert "TARGET" not in event
    assert "UNKNOWN_FIELD" not in event


def test_build_event_converts_nan_to_null():
    event = kp.build_event(
        _row(AMT_CREDIT=math.nan, NAME_INCOME_TYPE=math.nan), 0
    )

    assert event["AMT_CREDIT"] is None
    assert event["NAME_INCOME_TYPE"] is None
    text = json.dumps(event, allow_nan=False)
    assert "NaN" not in text
    assert '"AMT_CREDIT": null' in text


def test_event_timestamp_is_valid_iso_utc_and_increases():
    first = kp.build_event(_row(), 0)
    second = kp.build_event(_row(), 1)

    first_time = datetime.fromisoformat(
        first["event_timestamp"].replace("Z", "+00:00")
    )
    second_time = datetime.fromisoformat(
        second["event_timestamp"].replace("Z", "+00:00")
    )

    assert _is_iso_utc(first["event_timestamp"])
    assert first_time.tzinfo is not None
    assert second_time - first_time == timedelta(seconds=3)


def test_make_invalid_event_removes_or_breaks_required_field():
    event = kp.build_event(_row(), 0)

    invalid = kp.make_invalid_event(event, random.Random(0))

    assert invalid is not event
    assert event["SK_ID_CURR"] == 100002
    assert _is_iso_utc(event["event_timestamp"])
    has_missing_sk_id = "SK_ID_CURR" not in invalid
    has_bad_timestamp = not _is_iso_utc(invalid.get("event_timestamp", ""))
    assert has_missing_sk_id or has_bad_timestamp
    assert "TARGET" not in invalid


def test_produce_events_sends_json_with_sk_id_curr_as_key():
    producer = FakeProducer()
    tracker = kp.DeliveryTracker()

    invalid = kp.produce_events(
        [_row(SK_ID_CURR=1), _row(SK_ID_CURR=2)],
        producer,
        "loan-applications",
        delay=0,
        bad_ratio=0,
        tracker=tracker,
    )

    assert invalid == 0
    assert tracker.delivered == 2
    assert tracker.failed == 0
    assert producer.flushed is True
    assert [message["key"] for message in producer.messages] == [b"1", b"2"]
    for message in producer.messages:
        assert message["topic"] == "loan-applications"
        payload = json.loads(message["value"].decode("utf-8"))
        assert payload["schema_version"] == "schema-v1"
        assert "TARGET" not in payload


def test_produce_events_counts_failed_deliveries():
    producer = FakeProducer(fail_index=0)
    tracker = kp.DeliveryTracker()

    kp.produce_events(
        [_row(), _row()],
        producer,
        "loan-applications",
        delay=0,
        bad_ratio=0,
        tracker=tracker,
    )

    assert tracker.delivered == 1
    assert tracker.failed == 1


def test_produce_events_with_bad_ratio_one_makes_all_events_invalid():
    producer = FakeProducer()
    tracker = kp.DeliveryTracker()

    invalid = kp.produce_events(
        [_row(), _row(), _row()],
        producer,
        "loan-applications",
        delay=0,
        bad_ratio=1.0,
        tracker=tracker,
    )

    assert invalid == 3
    assert tracker.delivered == 3
    for message in producer.messages:
        payload = json.loads(message["value"].decode("utf-8"))
        assert "SK_ID_CURR" not in payload or not _is_iso_utc(
            payload["event_timestamp"]
        )


def test_read_applications_uses_path_and_ignores_target(tmp_path: Path):
    csv_path = tmp_path / "applications.csv"
    csv_path.write_text(
        "SK_ID_CURR,TARGET,NAME_CONTRACT_TYPE,AMT_CREDIT\n"
        "100002,1,Cash loans,406597.5\n"
        "100003,0,Revolving loans,12000.0\n",
        encoding="utf-8",
    )

    rows = kp.read_applications(csv_path, limit=1)

    assert len(rows) == 1
    assert rows[0]["SK_ID_CURR"] == 100002
    assert "TARGET" not in rows[0]
