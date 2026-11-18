"""Replay application_train.csv rows into Kafka as schema-v1 events."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

SCHEMA_VERSION = "schema-v1"
DEFAULT_TOPIC = "loan-applications"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:19092"
SIMULATED_START = datetime(2026, 11, 11, 10, 15, 30, tzinfo=timezone.utc)
TIMESTAMP_STEP = timedelta(seconds=3)

FEATURE_FIELDS = (
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
INTEGER_FIELDS = ("SK_ID_CURR", "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_CHILDREN")


def _clean_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _event_timestamp(event_index: int) -> str:
    moment = SIMULATED_START + TIMESTAMP_STEP * event_index
    return moment.isoformat().replace("+00:00", "Z")


def build_event(row: Mapping[str, Any], event_index: int) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_id": str(uuid.uuid4()),
        "event_timestamp": _event_timestamp(event_index),
        "schema_version": SCHEMA_VERSION,
    }
    for field in ("SK_ID_CURR", *FEATURE_FIELDS):
        value = _clean_value(row.get(field))
        if field in INTEGER_FIELDS and value is not None:
            value = int(value)
        event[field] = value
    return event


def make_invalid_event(
    event: Mapping[str, Any], rng: random.Random
) -> dict[str, Any]:
    invalid = dict(event)
    reason = rng.choice(("missing_sk_id", "bad_timestamp"))
    if reason == "missing_sk_id":
        invalid.pop("SK_ID_CURR", None)
    else:
        invalid["event_timestamp"] = "11/11/2026 10:15"
    return invalid


def should_make_invalid(rng: random.Random, bad_ratio: float) -> bool:
    return bad_ratio > 0 and rng.random() < bad_ratio


class DeliveryTracker:
    def __init__(self) -> None:
        self.delivered = 0
        self.failed = 0

    def callback(self, err: Any, msg: Any) -> None:
        if err is not None:
            self.failed += 1
        else:
            self.delivered += 1


def read_applications(csv_path: str | Path, limit: int) -> list[dict[str, Any]]:
    allowed = {"SK_ID_CURR", *FEATURE_FIELDS}
    frame = pd.read_csv(
        csv_path,
        usecols=lambda column: column in allowed,
        nrows=limit,
    )
    return frame.to_dict(orient="records")


def create_producer(bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS) -> Any:
    from confluent_kafka import Producer

    return Producer({"bootstrap.servers": bootstrap_servers})


def produce_events(
    rows: Iterable[Mapping[str, Any]],
    producer: Any,
    topic: str,
    delay: float,
    bad_ratio: float,
    tracker: DeliveryTracker,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random.Random()
    invalid_on_purpose = 0
    for event_index, row in enumerate(rows):
        event = build_event(row, event_index)
        if should_make_invalid(rng, bad_ratio):
            event = make_invalid_event(event, rng)
            invalid_on_purpose += 1
        key = event.get("SK_ID_CURR")
        payload = json.dumps(event, allow_nan=False).encode("utf-8")
        try:
            producer.produce(
                topic,
                key=str(key).encode("utf-8") if key is not None else None,
                value=payload,
                on_delivery=tracker.callback,
            )
            producer.poll(0)
        except Exception:
            tracker.failed += 1
        if delay > 0:
            time.sleep(delay)
    producer.flush()
    return invalid_on_purpose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--bad-ratio", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be zero or greater")
    if args.delay < 0:
        parser.error("--delay must be zero or greater")
    if not 0 <= args.bad_ratio <= 1:
        parser.error("--bad-ratio must be between 0 and 1")

    rows = read_applications(args.csv, args.limit)
    producer = create_producer()
    tracker = DeliveryTracker()
    invalid_on_purpose = produce_events(
        rows,
        producer,
        DEFAULT_TOPIC,
        args.delay,
        args.bad_ratio,
        tracker,
    )
    print(
        "Summary: "
        f"sent={tracker.delivered}, "
        f"failed={tracker.failed}, "
        f"invalid_on_purpose={invalid_on_purpose}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
