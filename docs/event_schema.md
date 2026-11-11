# Application event schema

## Purpose

One event represents one loan application sent through Kafka. Historical rows
from `data/raw/application_train.csv` are replayed as these events.

## Version

`schema-v1`

## Required metadata

| Field | Type | Required | Description |
|---|---|---|---|
| `event_id` | string (UUID) | yes | Unique identifier for this event. |
| `event_timestamp` | string (ISO 8601) | yes | Time the event was produced, e.g. `2026-11-11T10:15:30Z`. |
| `SK_ID_CURR` | integer | yes | Applicant ID from `application_train.csv`. Must be a positive integer. |
| `schema_version` | string | yes | Must be `schema-v1`. |

## Feature fields

These fields come from `data/raw/application_train.csv` and match the applicant
allowlist in `src/api/fields.py` and `docs/api_contract.md`.

| Field | Type | Required | Description |
|---|---|---|---|
| `NAME_CONTRACT_TYPE` | string | optional | Loan contract type. |
| `NAME_INCOME_TYPE` | string | optional | Applicant income type. |
| `NAME_EDUCATION_TYPE` | string | optional | Applicant education level. |
| `NAME_FAMILY_STATUS` | string | optional | Applicant family status. |
| `NAME_HOUSING_TYPE` | string | optional | Applicant housing type. |
| `AMT_INCOME_TOTAL` | number | optional | Total applicant income. |
| `AMT_CREDIT` | number | optional | Requested credit amount. |
| `AMT_ANNUITY` | number | optional | Loan annuity amount. |
| `AMT_GOODS_PRICE` | number | optional | Price of the goods being financed. |
| `DAYS_BIRTH` | integer | optional | Applicant age in days (negative). |
| `DAYS_EMPLOYED` | integer | optional | Days employed (negative). |
| `CNT_CHILDREN` | integer | optional | Number of children. |
| `CNT_FAM_MEMBERS` | number | optional | Number of family members. |

`TARGET` is the historical outcome label. It must never be in an event.

`installments_payments.csv` is not part of an application event. Its columns
(`SK_ID_PREV`, `SK_ID_CURR`, `NUM_INSTALMENT_VERSION`,
`NUM_INSTALMENT_NUMBER`, `DAYS_INSTALMENT`, `DAYS_ENTRY_PAYMENT`,
`AMT_INSTALMENT`, `AMT_PAYMENT`) are used only for repayment-sequence modeling.

## Validation rules

An event is invalid when:

- a required metadata field is missing (`event_id`, `event_timestamp`,
  `SK_ID_CURR`, or `schema_version`);
- a field has the wrong type (for example `SK_ID_CURR` is not an integer, or
  `AMT_CREDIT` is not a number);
- `event_timestamp` is not a valid ISO 8601 timestamp;
- `schema_version` is not `schema-v1`;
- `SK_ID_CURR` is not a positive integer;
- `TARGET` is present;
- an unknown field is present.

## Invalid events

Invalid events are not processed. They are sent to the Kafka dead-letter queue
and written to the `rejected/` prefix in S3 for inspection. Valid events
continue to the normal processing and storage path.

## Valid example

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

## Invalid example

```json
{
  "event_id": "5b0f3d2e-8f3a-4a4e-9d0c-2f6b1a7c9e10",
  "event_timestamp": "11/11/2026 10:15",
  "schema_version": "schema-v1",
  "SK_ID_CURR": "100002",
  "AMT_CREDIT": 406597.5,
  "TARGET": 1
}
```

This event is invalid because the timestamp is not ISO 8601, `SK_ID_CURR` is a
string instead of an integer, and `TARGET` is present.
