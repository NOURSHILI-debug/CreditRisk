from __future__ import annotations

import hashlib
import json

import pytest

from storage.local_storage import LocalStorage
from storage.s3_storage import S3Storage

EVENT_ID = "5b0f3d2e-8f3a-4a4e-9d0c-2f6b1a7c9e10"


def _event(**overrides):
    event = {
        "event_id": EVENT_ID,
        "event_timestamp": "2026-11-11T10:15:30Z",
        "schema_version": "schema-v1",
        "SK_ID_CURR": 100002,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "AMT_CREDIT": 406597.5,
    }
    event.update(overrides)
    return event


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = Body


def test_local_storage_valid_path_and_content(tmp_path):
    storage = LocalStorage(tmp_path)

    path = storage.save_valid(_event())

    assert path == (
        tmp_path / "raw" / "date=2026-11-11" / f"{EVENT_ID}.json"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == _event()


def test_local_storage_same_valid_event_overwrites(tmp_path):
    storage = LocalStorage(tmp_path)

    first = storage.save_valid(_event())
    second = storage.save_valid(_event())

    assert first == second
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_local_storage_rejected_path_and_content(tmp_path):
    storage = LocalStorage(tmp_path)
    raw = json.dumps(_event(TARGET=1)).encode("utf-8")
    errors = ["field TARGET must not be present"]

    path = storage.save_rejected(raw, errors)

    assert path == (
        tmp_path / "rejected" / "date=2026-11-11" / f"{EVENT_ID}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["errors"] == errors
    assert payload["raw"] == raw.decode("utf-8")


def test_local_storage_rejected_without_event_id_uses_hash(tmp_path):
    storage = LocalStorage(tmp_path)
    raw = b"not json"
    expected_id = hashlib.sha256(raw).hexdigest()

    first = storage.save_rejected(raw, ["invalid JSON"])
    second = storage.save_rejected(raw, ["invalid JSON"])

    assert first == second
    assert first.name == f"{expected_id}.json"
    assert first.parent.name.startswith("date=")


def test_s3_storage_valid_key_format_and_same_key():
    client = FakeS3Client()
    storage = S3Storage(bucket="test-bucket", client=client)

    first = storage.save_valid(_event())
    second = storage.save_valid(_event())

    assert first == f"raw/date=2026-11-11/{EVENT_ID}.json"
    assert second == first
    assert len(client.objects) == 1
    body = client.objects[("test-bucket", first)]
    assert json.loads(body.decode("utf-8")) == _event()


def test_s3_storage_rejected_key_format_and_same_key():
    client = FakeS3Client()
    storage = S3Storage(bucket="test-bucket", client=client)
    raw = json.dumps(_event(TARGET=1)).encode("utf-8")
    errors = ["field TARGET must not be present"]

    first = storage.save_rejected(raw, errors)
    second = storage.save_rejected(raw, errors)

    assert first == f"rejected/date=2026-11-11/{EVENT_ID}.json"
    assert second == first
    assert len(client.objects) == 1
    body = json.loads(client.objects[("test-bucket", first)].decode("utf-8"))
    assert body["errors"] == errors
    assert body["raw"] == raw.decode("utf-8")


def test_s3_storage_reads_bucket_from_environment(monkeypatch):
    monkeypatch.setenv("CREDIT_RISK_BUCKET", "env-bucket")
    storage = S3Storage(client=FakeS3Client())

    assert storage.bucket == "env-bucket"


def test_s3_storage_requires_bucket(monkeypatch):
    monkeypatch.delenv("CREDIT_RISK_BUCKET", raising=False)

    with pytest.raises(ValueError):
        S3Storage(client=FakeS3Client())
