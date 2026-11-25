# Kafka producer

## Purpose

The producer replays rows from `data/raw/application_train.csv` as JSON events on
the Redpanda/Kafka topic `loan-applications`. Each event follows
`schema-v1` from `docs/event_schema.md`. It is for local testing of the streaming
path, not live banking traffic.

## How it works

1. `read_applications` reads only the allowlisted columns from the CSV, with an
   optional row limit.
2. `build_event` creates one event per row: a UUID `event_id` derived
   deterministically from `schema_version` and `SK_ID_CURR`, a simulated
   `event_timestamp`, `schema_version`, `SK_ID_CURR`, and the schema feature
   fields.
3. Missing values (`NaN`) become `null`. Integer fields are converted to
   integers when they are not `null`. `TARGET` is never copied.
4. If `--bad-ratio` is above zero, a share of events is replaced by an invalid
   event: either `SK_ID_CURR` is removed or `event_timestamp` is set to a
   non-ISO value.
5. Each event is serialized with `json.dumps(..., allow_nan=False)` and sent with
   `SK_ID_CURR` as the message key. A delivery callback counts success and
   failure.
6. `producer.flush()` waits for delivery, then the producer prints a summary:
   `Summary: sent=<n>, failed=<n>, invalid_on_purpose=<n>`.

The fixed values used by the producer are:

- Topic: `loan-applications`
- Bootstrap server: `localhost:19092`
- Simulated start: `2026-11-11T10:15:30Z`
- Timestamp step: 3 seconds per event

## Event format

Valid event:

```json
{
  "event_id": "5b0f3d2e-8f3a-4a4e-9d0c-2f6b1a7c9e10",
  "event_timestamp": "2026-11-11T10:15:30Z",
  "schema_version": "schema-v1",
  "SK_ID_CURR": 100002,
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
  "CNT_FAM_MEMBERS": 1.0
}
```

Invalid event produced on purpose (bad timestamp):

```json
{
  "event_id": "5b0f3d2e-8f3a-4a4e-9d0c-2f6b1a7c9e10",
  "event_timestamp": "11/11/2026 10:15",
  "schema_version": "schema-v1",
  "SK_ID_CURR": 100002,
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
  "CNT_FAM_MEMBERS": 1.0
}
```

The other invalid form removes `SK_ID_CURR`. `TARGET` is never in an event.

## How to run

From the project root in Windows PowerShell:

```powershell
Set-Location C:\Users\SINORFI\Desktop\CreditRisk
.\.venv\Scripts\Activate.ps1
python -m pip install "confluent-kafka>=2.0"
python .\src\ingestion\kafka_producer.py --csv .\data\raw\application_train.csv --limit 100 --delay 0 --bad-ratio 0
```

Options:

- `--csv PATH` (required): path to `application_train.csv`.
- `--limit N` (default `100`): maximum number of rows to read.
- `--delay SECONDS` (default `0`): pause after each produced event.
- `--bad-ratio R` (default `0`): share of events made invalid on purpose, from
  `0` to `1`.

Example with a slower replay and some invalid events:

```powershell
python .\src\ingestion\kafka_producer.py --csv .\data\raw\application_train.csv --limit 20 --delay 0.5 --bad-ratio 0.2
```

## How to check it worked

Read events from the topic with Redpanda's `rpk` inside the running container:

```powershell
docker exec -it redpanda rpk topic consume loan-applications --brokers localhost:19092 --num 10
```

## Design choices

- `SK_ID_CURR` is the message key so all events for one applicant share a
  partition and stay in order.
- `event_id` is a deterministic UUID5 derived from `schema_version` and
  `SK_ID_CURR`, so replaying the same row overwrites the same storage object
  instead of creating a duplicate.
- `TARGET` is excluded because it is the historical outcome label, is not
  available at prediction time, and would leak the label.
- Timestamps are simulated because replay uses old CSV rows; a fixed start plus
  a few seconds per event gives repeatable ordering without claiming real event
  time.

## Tests

The unit tests are in `tests/unit/test_kafka_producer.py` and do not need a
running broker:

- `test_build_event_has_required_metadata_and_feature_fields`: required fields
  and all feature fields are present.
- `test_build_event_never_includes_target_or_unknown_fields`: `TARGET` and
  unknown columns are dropped.
- `test_build_event_converts_nan_to_null`: `NaN` becomes `null` and the JSON has
  no `NaN`.
- `test_event_timestamp_is_valid_iso_utc_and_increases`: the timestamp is ISO
  8601 UTC and increases by 3 seconds.
- `test_make_invalid_event_removes_or_breaks_required_field`: bad-event
  generation either removes `SK_ID_CURR` or breaks the timestamp.
- `test_produce_events_sends_json_with_sk_id_curr_as_key`: the fake producer
  receives JSON with `SK_ID_CURR` as the key.
- `test_produce_events_counts_failed_deliveries`: the delivery callback counts
  one failure and one success.
- `test_produce_events_with_bad_ratio_one_makes_all_events_invalid`: a bad ratio
  of `1` marks every event invalid.
- `test_read_applications_uses_path_and_ignores_target`: the CSV path is used,
  the limit works, and `TARGET` is not read.

Run them with:

```powershell
python -m pytest tests\unit\test_kafka_producer.py -q
```

## Limits

- No consumer is implemented here.
- No dead-letter queue routing is implemented; invalid events are only produced
  on purpose for later testing.
- No S3 writing is implemented.
- The producer does not run the full schema validation; it only builds events
  and optionally breaks them.
- The topic and bootstrap server are constants in the module, not CLI options.
