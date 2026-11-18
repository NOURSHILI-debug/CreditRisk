"""S3 storage for valid and rejected application events."""

from __future__ import annotations

import json
import os
from typing import Any

from storage.local_storage import rejected_key, rejected_payload, valid_key

BUCKET_ENV = "CREDIT_RISK_BUCKET"
DEFAULT_REGION = "us-east-1"


class S3Storage:
    def __init__(
        self,
        bucket: str | None = None,
        client: Any = None,
        region: str = DEFAULT_REGION,
    ):
        self.bucket = bucket or os.getenv(BUCKET_ENV)
        if not self.bucket:
            raise ValueError(f"{BUCKET_ENV} is not set")
        self.region = region
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def save_valid(self, event: dict[str, Any]) -> str:
        key = valid_key(event)
        body = json.dumps(event, ensure_ascii=False, indent=2).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return key

    def save_rejected(self, raw_bytes: Any, errors: list[str]) -> str:
        key = rejected_key(raw_bytes)
        payload = rejected_payload(raw_bytes, errors)
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return key
