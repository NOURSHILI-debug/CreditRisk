from __future__ import annotations

import json

from ingestion import kafka_consumer as kc

EVENT_ID = "5b0f3d2e-8f3a-4a4e-9d0c-2f6b1a7c9e10"


def _valid_event():
    return {
        "event_id": EVENT_ID,
        "event_timestamp": "2026-11-11T10:15:30Z",
        "schema_version": "schema-v1",
        "SK_ID_CURR": 100002,
        "AMT_CREDIT": 406597.5,
    }


class FakeMessage:
    def __init__(self, value, error=None):
        self._value = value
        self._error = error

    def value(self):
        return self._value

    def error(self):
        return self._error


class FakeConsumer:
    def __init__(self, events=None, messages=None, commit_error=None):
        self.events = events if events is not None else []
        self.messages = list(messages or [])
        self.commit_error = commit_error
        self.commits = []
        self.subscriptions = []
        self.closed = False

    def subscribe(self, topics):
        self.subscriptions.append(topics)

    def poll(self, timeout):
        return self.messages.pop(0) if self.messages else None

    def commit(self, message=None, asynchronous=True):
        if self.commit_error is not None:
            raise self.commit_error
        self.commits.append(message)
        self.events.append(("commit", message))

    def close(self):
        self.closed = True


class FakeProducer:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.messages = []
        self.flushes = 0

    def produce(self, topic, value=None, key=None, headers=None):
        record = {
            "topic": topic,
            "value": value,
            "key": key,
            "headers": headers,
        }
        self.messages.append(record)
        self.events.append(("dlq", record))

    def flush(self):
        self.flushes += 1


class RecordingStorage:
    def __init__(self, events=None, fail_valid=False, fail_rejected=False):
        self.events = events if events is not None else []
        self.fail_valid = fail_valid
        self.fail_rejected = fail_rejected
        self.valid_events = []
        self.rejected = []

    def save_valid(self, event):
        if self.fail_valid:
            raise RuntimeError("S3 is down")
        self.valid_events.append(event)
        self.events.append(("save_valid", event))

    def save_rejected(self, raw_bytes, errors):
        if self.fail_rejected:
            raise RuntimeError("S3 is down")
        self.rejected.append((raw_bytes, errors))
        self.events.append(("save_rejected", raw_bytes, errors))


def test_valid_message_is_stored_then_committed():
    events = []
    storage = RecordingStorage(events)
    consumer = FakeConsumer(events)
    producer = FakeProducer(events)
    message = FakeMessage(json.dumps(_valid_event()).encode("utf-8"))

    status = kc.process_message(message, storage, producer, consumer)

    assert status == "valid"
    assert storage.valid_events == [_valid_event()]
    assert consumer.commits == [message]
    assert producer.messages == []
    assert [event[0] for event in events] == ["save_valid", "commit"]


def test_invalid_message_is_stored_sent_to_dlq_then_committed():
    events = []
    storage = RecordingStorage(events)
    consumer = FakeConsumer(events)
    producer = FakeProducer(events)
    raw = json.dumps(_valid_event() | {"TARGET": 1}).encode("utf-8")
    message = FakeMessage(raw)

    status = kc.process_message(message, storage, producer, consumer)

    assert status == "rejected"
    assert len(storage.rejected) == 1
    saved_raw, errors = storage.rejected[0]
    assert saved_raw == raw
    assert any("TARGET" in error for error in errors)
    assert consumer.commits == [message]
    assert len(producer.messages) == 1
    dlq = producer.messages[0]
    assert dlq["topic"] == kc.DLQ_TOPIC
    assert dlq["value"] == raw
    header_name, header_value = dlq["headers"][0]
    assert header_name == "errors"
    assert json.loads(header_value.decode("utf-8")) == errors
    assert [event[0] for event in events] == [
        "save_rejected",
        "dlq",
        "commit",
    ]


def test_storage_failure_for_valid_event_does_not_commit():
    events = []
    storage = RecordingStorage(events, fail_valid=True)
    consumer = FakeConsumer(events)
    producer = FakeProducer(events)
    message = FakeMessage(json.dumps(_valid_event()).encode("utf-8"))

    status = kc.process_message(message, storage, producer, consumer)

    assert status == "failed"
    assert consumer.commits == []
    assert producer.messages == []
    assert events == []


def test_storage_failure_for_rejected_event_does_not_commit_or_send_dlq():
    events = []
    storage = RecordingStorage(events, fail_rejected=True)
    consumer = FakeConsumer(events)
    producer = FakeProducer(events)
    raw = json.dumps(_valid_event() | {"TARGET": 1}).encode("utf-8")
    message = FakeMessage(raw)

    status = kc.process_message(message, storage, producer, consumer)

    assert status == "failed"
    assert consumer.commits == []
    assert producer.messages == []
    assert events == []


def test_commit_failure_returns_failed():
    events = []
    storage = RecordingStorage(events)
    consumer = FakeConsumer(events, commit_error=RuntimeError("commit failed"))
    producer = FakeProducer(events)
    message = FakeMessage(json.dumps(_valid_event()).encode("utf-8"))

    status = kc.process_message(message, storage, producer, consumer)

    assert status == "failed"
    assert consumer.commits == []


def test_consume_counts_valid_and_rejected_messages():
    events = []
    storage = RecordingStorage(events)
    valid_message = FakeMessage(json.dumps(_valid_event()).encode("utf-8"))
    invalid_raw = json.dumps(_valid_event() | {"TARGET": 1}).encode("utf-8")
    invalid_message = FakeMessage(invalid_raw)
    consumer = FakeConsumer(
        events, messages=[valid_message, invalid_message]
    )
    producer = FakeProducer(events)

    counts = kc.consume(
        consumer,
        producer,
        storage,
        max_messages=2,
        timeout=0.01,
    )

    assert counts == {"read": 2, "valid": 1, "rejected": 1, "failed": 0}
    assert consumer.subscriptions == [[kc.INPUT_TOPIC]]
    assert consumer.commits == [valid_message, invalid_message]
    assert len(storage.valid_events) == 1
    assert len(storage.rejected) == 1


def test_consume_stops_at_max_messages():
    events = []
    storage = RecordingStorage(events)
    messages = [
        FakeMessage(json.dumps(_valid_event()).encode("utf-8"))
        for _ in range(3)
    ]
    consumer = FakeConsumer(events, messages=messages)
    producer = FakeProducer(events)

    counts = kc.consume(
        consumer,
        producer,
        storage,
        max_messages=1,
        timeout=0.01,
    )

    assert counts == {"read": 1, "valid": 1, "rejected": 0, "failed": 0}


def test_consume_stops_after_storage_failure():
    events = []
    storage = RecordingStorage(events, fail_valid=True)
    messages = [
        FakeMessage(json.dumps(_valid_event()).encode("utf-8"))
        for _ in range(2)
    ]
    consumer = FakeConsumer(events, messages=messages)
    producer = FakeProducer(events)

    counts = kc.consume(
        consumer,
        producer,
        storage,
        max_messages=None,
        timeout=0.01,
    )

    assert counts == {"read": 1, "valid": 0, "rejected": 0, "failed": 1}
    assert consumer.commits == []
