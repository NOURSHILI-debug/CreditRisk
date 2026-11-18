# Kafka consumer

## Purpose

The consumer reads `schema-v1` application events from `loan-applications`.
Valid events are stored by event date. Invalid events are stored under
`rejected/` and sent to `loan-applications-dlq` with the errors.

## How it works

1. The CLI builds a local or S3 storage object, a Kafka consumer, and a DLQ
   producer.
2. The consumer joins the group and subscribes to `loan-applications`.
3. `consumer.poll(timeout)` reads one message. `None` means the timeout expired,
   so the loop stops.
4. `validate_event(raw_bytes)` parses the payload and returns `valid`, `event`,
   and `errors`.
5. A valid event goes to `storage.save_valid(event)`. After that succeeds, the
   consumer commits the message offset.
6. An invalid event goes to `storage.save_rejected(raw_bytes, errors)` and then
   to `loan-applications-dlq`. The errors are sent in a header named `errors`.
   After both calls complete without an exception, the consumer commits the
   message offset.
7. If storage or commit fails, the error is logged, the offset is not committed,
   and the loop stops. The message can be read again later.
8. `Ctrl+C` is caught, the consumer closes, and a summary is printed:
   `Summary: read=<n>, valid=<n>, rejected=<n>, failed=<n>`.

Auto-commit is disabled with `enable.auto.commit=False` and
`enable.auto.offset.store=False`.

## Validation rules

| Check | Invalid when |
|---|---|
| JSON | The bytes are not valid JSON. |
| JSON object | The top-level value is not an object. |
| Required fields | `event_id`, `event_timestamp`, `schema_version`, or `SK_ID_CURR` is missing or `null`. |
| `event_id` | It is not a UUID string. |
| `event_timestamp` | It is not a valid ISO 8601 timestamp. |
| `schema_version` | It is not `schema-v1`. |
| `SK_ID_CURR` | It is not an integer or is not positive. |
| `TARGET` | It is present in the event. |
| Unknown fields | A field is not in `docs/event_schema.md`. |
| String features | A value is not a string and not `null`. |
| Number features | A value is not a finite integer or float and not `null`. |
| Integer features | A value is not an integer and not `null`. |

Optional feature fields may be `null`. A validator result always has the form
`{"valid": bool, "event": dict | None, "errors": [str, ...]}`.

## Storage layout

Local storage uses `local_lake` by default:

```text
local_lake/raw/date=2026-11-11/5b0f3d2e-8f3a-4a4e-9d0c-2f6b1a7c9e10.json
local_lake/rejected/date=2026-11-11/5b0f3d2e-8f3a-4a4e-9d0c-2f6b1a7c9e10.json
```

S3 uses the same keys under the bucket from `CREDIT_RISK_BUCKET`:

```text
s3://<bucket>/raw/date=2026-11-11/5b0f3d2e-8f3a-4a4e-9d0c-2f6b1a7c9e10.json
s3://<bucket>/rejected/date=2026-11-11/5b0f3d2e-8f3a-4a4e-9d0c-2f6b1a7c9e10.json
```

A valid file contains the event JSON. A rejected file contains:

```json
{
  "raw": "<the original payload text>",
  "errors": ["field TARGET must not be present"]
}
```

The date comes from `event_timestamp`. If a rejected payload has no parseable
timestamp, the current UTC date is used.

The object id is `event_id` when it is present and safe for a file name.
Otherwise it is a SHA-256 hash: of the event JSON for valid saves, or of the raw
bytes for rejected saves. Replaying the same event writes the same key and
overwrites the previous object.

## How to run

Local storage, from the project root in Windows PowerShell:

```powershell
Set-Location C:\Users\SINORFI\Desktop\CreditRisk
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location).Path 'src')
python -m ingestion.kafka_consumer --storage local --max-messages 20 --group-id credit-risk-consumer --timeout 10
```

S3 storage:

```powershell
Set-Location C:\Users\SINORFI\Desktop\CreditRisk
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location).Path 'src')
python -m pip install "boto3>=1.28"
$env:CREDIT_RISK_BUCKET = "your-bucket-name"
python -m ingestion.kafka_consumer --storage s3 --max-messages 20 --group-id credit-risk-consumer --timeout 10
```

Boto3 uses its normal AWS credential chain. No keys are stored in the code or
repository.

Options:

- `--storage s3|local` (default `local`): choose the storage backend.
- `--max-messages N` (default: no limit): stop after `N` messages. With no
  value, keep reading until the timeout or `Ctrl+C`.
- `--group-id ID` (default `credit-risk-consumer`): Kafka consumer group.
- `--timeout SECONDS` (default `10`): how long `poll` waits for a message before
  the consumer stops.

## How to check results

Read DLQ messages with `rpk`:

```powershell
docker exec -it redpanda rpk topic consume loan-applications-dlq --brokers localhost:19092 --num 10
```

List S3 objects:

```powershell
aws s3 ls "s3://$env:CREDIT_RISK_BUCKET/raw/" --recursive
aws s3 ls "s3://$env:CREDIT_RISK_BUCKET/rejected/" --recursive
```

List local objects:

```powershell
Get-ChildItem .\local_lake -Recurse
```

## Design choices

- Offsets are committed manually after storage, so a stored event is never
  skipped because of an early commit.
- The consumer stops on a storage or commit failure instead of moving to the
  next message, which avoids silently losing the failed message.
- Storage keys are deterministic, so replaying an event overwrites the same
  object instead of creating duplicates.
- Invalid events go to both `rejected/` and the DLQ so the stored copy and the
  stream copy can be inspected.
- `TARGET` is rejected because it is the historical outcome label and must never
  be part of an event.

## Tests

The unit tests are split by component and use no broker or AWS:

- `tests/unit/test_event_validator.py`: valid event, each validation failure,
  `TARGET`, unknown fields, wrong types, and garbage bytes.
- `tests/unit/test_storage.py`: local paths and content, deterministic keys,
  hash fallback, S3 key format, environment bucket, and a fake S3 client.
- `tests/unit/test_kafka_consumer.py`: valid storage, rejected storage plus DLQ,
  commit order, storage failure without commit, commit failure, and loop counts.

Run them with:

```powershell
python -m pytest tests\unit\test_event_validator.py tests\unit\test_storage.py tests\unit\test_kafka_consumer.py -q
```

## Limits

- No Lambda handler is implemented.
- No CloudWatch metrics or log shipping is implemented.
- No retry or backoff loop is implemented; the consumer stops after a storage
  or commit failure.
- No DLQ replay tool is implemented.
- The consumer does not create the bucket or the Kafka topics.
- No schema registry is used; validation uses the local `schema-v1` rules.
