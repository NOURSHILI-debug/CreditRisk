"""Consume application events, store them, and route invalid events to the DLQ."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from ingestion.event_validator import validate_event

INPUT_TOPIC = "loan-applications"
DLQ_TOPIC = "loan-applications-dlq"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:19092"
DEFAULT_GROUP_ID = "credit-risk-consumer"
DEFAULT_TIMEOUT = 10.0

LOGGER = logging.getLogger(__name__)


def create_consumer(group_id: str) -> Any:
    from confluent_kafka import Consumer

    return Consumer(
        {
            "bootstrap.servers": DEFAULT_BOOTSTRAP_SERVERS,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
        }
    )


def create_dlq_producer() -> Any:
    from confluent_kafka import Producer

    return Producer({"bootstrap.servers": DEFAULT_BOOTSTRAP_SERVERS})


def build_storage(kind: str) -> Any:
    if kind == "s3":
        from storage.s3_storage import S3Storage

        return S3Storage()
    from storage.local_storage import LocalStorage

    return LocalStorage()


def send_to_dlq(
    producer: Any, topic: str, raw_bytes: bytes, errors: list[str]
) -> None:
    headers = [("errors", json.dumps(errors).encode("utf-8"))]
    producer.produce(topic, value=raw_bytes, headers=headers)
    producer.flush()


def _commit(consumer: Any, message: Any, logger: logging.Logger) -> bool:
    try:
        consumer.commit(message=message, asynchronous=False)
    except Exception as exc:
        logger.error("Failed to commit offset; message will be retried: %s", exc)
        return False
    return True


def process_message(
    message: Any,
    storage: Any,
    dlq_producer: Any,
    consumer: Any,
    logger: logging.Logger = LOGGER,
    dlq_topic: str = DLQ_TOPIC,
) -> str:
    raw_bytes = message.value() or b""
    result = validate_event(raw_bytes)

    if not result["valid"]:
        try:
            storage.save_rejected(raw_bytes, result["errors"])
            send_to_dlq(dlq_producer, dlq_topic, raw_bytes, result["errors"])
        except Exception as exc:
            logger.error(
                "Failed to handle rejected event; offset not committed: %s", exc
            )
            return "failed"
        return "rejected" if _commit(consumer, message, logger) else "failed"

    try:
        storage.save_valid(result["event"])
    except Exception as exc:
        logger.error(
            "Failed to store valid event; offset not committed: %s", exc
        )
        return "failed"
    return "valid" if _commit(consumer, message, logger) else "failed"


def consume(
    consumer: Any,
    dlq_producer: Any,
    storage: Any,
    max_messages: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    logger: logging.Logger = LOGGER,
    input_topic: str = INPUT_TOPIC,
    dlq_topic: str = DLQ_TOPIC,
) -> dict[str, int]:
    counts = {"read": 0, "valid": 0, "rejected": 0, "failed": 0}
    consumer.subscribe([input_topic])
    try:
        while max_messages is None or counts["read"] < max_messages:
            message = consumer.poll(timeout)
            if message is None:
                break
            if message.error():
                logger.error("Kafka consumer error: %s", message.error())
                break
            counts["read"] += 1
            status = process_message(
                message, storage, dlq_producer, consumer, logger, dlq_topic
            )
            counts[status] += 1
            if status == "failed":
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user; stopping consumer")
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", choices=("s3", "local"), default="local")
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--group-id", default=DEFAULT_GROUP_ID)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_messages is not None and args.max_messages < 0:
        parser.error("--max-messages must be zero or greater")
    if args.timeout < 0:
        parser.error("--timeout must be zero or greater")

    storage = build_storage(args.storage)
    consumer = create_consumer(args.group_id)
    dlq_producer = create_dlq_producer()
    try:
        counts = consume(
            consumer,
            dlq_producer,
            storage,
            max_messages=args.max_messages,
            timeout=args.timeout,
        )
    finally:
        consumer.close()
        dlq_producer.flush()

    print(
        "Summary: "
        f"read={counts['read']}, "
        f"valid={counts['valid']}, "
        f"rejected={counts['rejected']}, "
        f"failed={counts['failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
