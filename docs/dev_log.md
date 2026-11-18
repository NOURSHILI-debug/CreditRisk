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
