"""Local filesystem storage for valid and rejected application events."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
DEFAULT_BASE_DIR = Path("local_lake")


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_value(event: Any, field: str) -> Any:
    return event.get(field) if isinstance(event, dict) else None


def event_date(event: Any) -> str:
    moment = _parse_timestamp(_event_value(event, "event_timestamp"))
    date = moment.date() if moment is not None else datetime.now(timezone.utc).date()
    return date.isoformat()


def _safe_id(value: Any) -> str | None:
    if isinstance(value, str) and value and _SAFE_ID.fullmatch(value):
        return value
    return None


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _event_id_or_hash(event: Any) -> str:
    event_id = _safe_id(_event_value(event, "event_id"))
    if event_id is not None:
        return event_id
    encoded = json.dumps(
        event, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return _hash_bytes(encoded)


def to_bytes(raw: Any) -> bytes:
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, str):
        return raw.encode("utf-8")
    return str(raw).encode("utf-8")


def _parsed_raw(raw: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _rejected_date(raw: bytes) -> str:
    event = _parsed_raw(raw)
    moment = _parse_timestamp(_event_value(event, "event_timestamp"))
    date = moment.date() if moment is not None else datetime.now(timezone.utc).date()
    return date.isoformat()


def _rejected_id(raw: bytes) -> str:
    event = _parsed_raw(raw)
    event_id = _safe_id(_event_value(event, "event_id"))
    if event_id is not None:
        return event_id
    return _hash_bytes(raw)


def valid_key(event: Any) -> str:
    return f"raw/date={event_date(event)}/{_event_id_or_hash(event)}.json"


def rejected_key(raw_bytes: Any) -> str:
    raw = to_bytes(raw_bytes)
    return f"rejected/date={_rejected_date(raw)}/{_rejected_id(raw)}.json"


def rejected_payload(raw_bytes: Any, errors: list[str]) -> dict[str, Any]:
    raw = to_bytes(raw_bytes)
    return {"raw": raw.decode("utf-8", errors="replace"), "errors": list(errors)}


class LocalStorage:
    def __init__(self, base_dir: str | Path = DEFAULT_BASE_DIR):
        self.base_dir = Path(base_dir)

    def _write(self, key: str, text: str) -> Path:
        path = self.base_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def save_valid(self, event: dict[str, Any]) -> Path:
        text = json.dumps(event, ensure_ascii=False, indent=2)
        return self._write(valid_key(event), text)

    def save_rejected(self, raw_bytes: Any, errors: list[str]) -> Path:
        payload = rejected_payload(raw_bytes, errors)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return self._write(rejected_key(raw_bytes), text)
