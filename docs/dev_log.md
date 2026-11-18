# Development log

## 2026-11-18 - Kafka producer (`feat/kafka-producer`)

### Steps

1. Read `docs/event_schema.md` and the `application_train.csv` header to confirm
   the `schema-v1` metadata fields and feature fields.
2. Created `C:\Users\SINORFI\Desktop\CreditRisk\src\ingestion\kafka_producer.py`:
   builds `schema-v1` events from CSV rows, sends JSON to Kafka with
   `SK_ID_CURR` as the key, supports bad-event injection, and prints a summary.
3. Created `C:\Users\SINORFI\Desktop\CreditRisk\tests\unit\test_kafka_producer.py`:
   nine offline tests for event building, `NaN` to `null`, timestamps, bad
   events, delivery counting, and a mocked producer.
4. Created `C:\Users\SINORFI\Desktop\CreditRisk\docs\kafka_producer.md`: purpose,
   flow, event examples, PowerShell commands, `rpk` check, design choices,
   tests, and limits.
5. Updated `C:\Users\SINORFI\Desktop\CreditRisk\pyproject.toml`: declared the
   runtime dependencies, including `confluent-kafka>=2.0.0`.
6. Updated `C:\Users\SINORFI\Desktop\CreditRisk\README.md`: added the one-line
   link to `docs/kafka_producer.md`.
7. Created `C:\Users\SINORFI\Desktop\CreditRisk\docs\dev_log.md`: this entry.

### Commands and results

- `.\.venv\Scripts\python.exe -m pytest tests\unit\test_kafka_producer.py -q`
  -> `9 passed in 0.55s`.
- `.\.venv\Scripts\python.exe -m ruff check src\ingestion\kafka_producer.py tests\unit\test_kafka_producer.py`
  -> `All checks passed!`.
- Offline sanity check with `read_applications('data/raw/application_train.csv', 1)`
  and `build_event(...)` -> `TARGET` absent, valid JSON, integer `SK_ID_CURR`,
  timestamp `2026-11-11T10:15:30Z`.

### Not done

- Did not run the producer or `rpk` against Redpanda, because the task said to
  run only the unit tests.
- Did not add `confluent-kafka` to `requirements.txt`, because the file rules
  allowed only `pyproject.toml` for a missing dependency. It is declared in
  `pyproject.toml` and is already installed in `.venv`.
- Did not implement the consumer, dead-letter queue routing, or S3 writing; those
  are outside this task.

## 2026-11-18 - Kafka consumer, validator, and storage (`feat/kafka-consumer`)

### Steps

1. Read `docs/event_schema.md` and the existing producer to reuse the
   `schema-v1` field names and rules.
2. Filled `C:\Users\SINORFI\Desktop\CreditRisk\src\ingestion\event_validator.py`:
   validates raw JSON bytes against `schema-v1` and returns `valid`, `event`, and
   `errors` without crashing on bad input.
3. Filled `C:\Users\SINORFI\Desktop\CreditRisk\src\storage\local_storage.py`:
   saves valid and rejected events under `local_lake/raw` and
   `local_lake/rejected` with deterministic keys.
4. Filled `C:\Users\SINORFI\Desktop\CreditRisk\src\storage\s3_storage.py`:
   writes the same keys to S3 using the bucket from `CREDIT_RISK_BUCKET` and the
   `us-east-1` region.
5. Filled `C:\Users\SINORFI\Desktop\CreditRisk\src\ingestion\kafka_consumer.py`:
   reads messages, validates them, stores valid events, sends invalid events to
   storage and the DLQ, commits after storage, and prints a summary.
6. Filled `C:\Users\SINORFI\Desktop\CreditRisk\tests\unit\test_storage.py` and
   created `C:\Users\SINORFI\Desktop\CreditRisk\tests\unit\test_event_validator.py`
   and `C:\Users\SINORFI\Desktop\CreditRisk\tests\unit\test_kafka_consumer.py`:
   offline tests with fake storage, S3 client, Kafka consumer, and producer.
7. Created `C:\Users\SINORFI\Desktop\CreditRisk\docs\kafka_consumer.md`: purpose,
   flow, validation table, storage layout, PowerShell commands, checks, design
   choices, tests, and limits.
8. Updated `C:\Users\SINORFI\Desktop\CreditRisk\pyproject.toml`: added
   `boto3>=1.28.0`.
9. Updated `C:\Users\SINORFI\Desktop\CreditRisk\README.md`: added the one-line
   link to `docs/kafka_consumer.md` next to the producer link.
10. Updated `C:\Users\SINORFI\Desktop\CreditRisk\docs\dev_log.md`: this entry.

### Commands and results

- `.\.venv\Scripts\python.exe -m pytest tests\unit\test_event_validator.py tests\unit\test_storage.py tests\unit\test_kafka_consumer.py -q`
  -> `33 passed in 0.30s`.
- `.\.venv\Scripts\python.exe -m ruff check src\ingestion\event_validator.py src\ingestion\kafka_consumer.py src\storage\local_storage.py src\storage\s3_storage.py tests\unit\test_event_validator.py tests\unit\test_storage.py tests\unit\test_kafka_consumer.py`
  -> `All checks passed!`.

### Not done

- Did not run the consumer against Redpanda or AWS, because the task said not to.
- Did not install or call boto3; the S3 tests use a fake client.
- Did not add `boto3` to `requirements.txt`, because the file rules allowed only
  `pyproject.toml` for a missing dependency.
- Did not implement Lambda, CloudWatch, retries, or DLQ replay; those are outside
  this task.
