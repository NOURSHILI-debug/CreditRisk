# API contract (single source of truth)

This document defines the HTTP contract between the FastAPI service in `src/api`
and the dashboard data client in `src/dashboard`.

The API returns the probability of `TARGET = 1` (repayment difficulty). It is
**not** an approval or rejection decision. Risk bands are provisional and must
not drive automatic lending decisions.

> **WARNING**
> Responses contain applicant-derived data. Run the service on localhost only.
> Authentication and authorization are required before any real deployment;
> the current service has neither.

## Canonical prediction statuses

`PredictionStatus` is the dashboard's existing vocabulary:

`available`, `no_prediction`, `unavailable`, `model_unavailable`, `invalid`.

Mapping (implemented in `src/api/status.py`):

| Case | HTTP | `status` | Notes |
|---|---:|---|---|
| Valid probability in `[0, 1]` | 200 | `available` | `probability` is a finite number. |
| Valid application, provider has no prediction | 200 | `no_prediction` | `probability` is `null`; never defaulted. |
| Unknown `application_id` on a prediction endpoint | 404 | `no_prediction` | **To be confirmed by Person 2.** |
| Model or preprocessor missing/incompatible | 503 | `model_unavailable` | Includes missing evaluation artifact. |
| Application data source unavailable | 503 | `unavailable` | |
| Malformed input or inference failure | 422 | `invalid` | |
| Model returns an invalid probability | 500 | `invalid` | Booleans, NaN, infinities, out-of-range. |
| Unknown application on `GET /applications/{id}` | 404 | `invalid` | Application lookup, not prediction. |
| Duplicate application IDs | 409 | `invalid` | Explicit error, never an arbitrary row. |
| Unknown fields or pagination bounds | 422 | `invalid` | |
| Unknown route | 404 | `invalid` | |

`predict_credit_risk()` in `src/inference/predictor.py` keeps its legacy
`success` / `error` values. Only the HTTP API uses canonical statuses.

## Uniform error shape

Every non-2xx response uses the same keys. `probability` is always `null` on
errors and is never defaulted.

```json
{
  "status": "invalid",
  "message": "Application 999999999 was not found.",
  "application_id": null,
  "probability": null,
  "risk_level": null,
  "model_name": null,
  "model_version": null,
  "prediction_timestamp": null,
  "explanation": null
}
```

Prediction errors may populate `application_id`, `model_name`,
`model_version` and `prediction_timestamp`; the keys never change.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service name, version and `/docs` pointer. |
| GET | `/health` | Model and data availability. Always 200; `status` is `ok` or `degraded`. |
| GET | `/models` | Model metadata and provisional thresholds. 503 `model_unavailable` when artifacts are missing. |
| GET | `/model/evaluation?split=test` | Canonical evaluation rows plus provenance. `split` is `test` (default) or `validation`. |
| GET | `/model/prediction-summary?split=test` | Counts and risk bands derived from saved validation/test predictions. |
| GET | `/model/global-importance` | Raw `global_importance.json` with local paths removed. |
| GET | `/applications` | Paginated application rows with an explicit field allowlist. |
| GET | `/applications/{application_id}` | One allowlisted source record; optional `fields` filter. |
| POST | `/predict` | Predict by `application_id` or `features`. Optional `include_explanation=true`. |
| GET | `/applications/{application_id}/prediction` | Predict by application ID. Optional `include_explanation=true`. |

### `GET /`

```json
{
  "service": "Credit Risk Intelligence API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "data_available": true,
  "model_name": "XGBoost",
  "model_version": "baseline-v2-local",
  "model_hash": "75fd9738cb1a709b47f64c89126e8c5344d3a4d6368ef16c6870c960360a8497"
}
```

### `GET /applications`

Parameters:

| Name | Default | Rules |
|---|---|---|
| `fields` | all allowlisted fields | Comma-separated, deduplicated. Unknown names → 422 `invalid`. |
| `limit` | `100` | `1..500` (hard cap). Out of range → 422 `invalid`. |
| `offset` | `0` | `>= 0`. Rows are stably ordered by `SK_ID_CURR`. |
| `application_id` | omitted | Exact ID filter. |
| `filter_field`, `filter_value` | omitted | Exact category filter; both required together. Fields are limited to contract, income, and education type. |

Allowlist (derived from Overview and Application Explorer usage):
`SK_ID_CURR`, `NAME_CONTRACT_TYPE`, `NAME_INCOME_TYPE`,
`NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`,
`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`,
`DAYS_BIRTH`, `DAYS_EMPLOYED`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`.
`TARGET`, `RISK_PROBABILITY`, and `RISK_CATEGORY` are excluded to avoid
exposing outcome labels or non-source prediction values through applicant browsing.

Real response:

```json
{
  "total": 307511,
  "limit": 2,
  "offset": 0,
  "fields": ["SK_ID_CURR", "AMT_CREDIT"],
  "items": [
    {"SK_ID_CURR": 100002, "AMT_CREDIT": 406597.5},
    {"SK_ID_CURR": 100003, "AMT_CREDIT": 1293502.5}
  ]
}
```

Response schema: `ApplicationListResponse` with `total` (int), `limit` (int),
`offset` (int), `fields` (string[]), `items` (object[]).

### `GET /applications/{application_id}`

Returns only the allowlisted applicant fields nested under `application`;
`TARGET` and other outcome labels are never returned by this endpoint. Use
`fields` (same allowlist) to trim the response. Unknown ID → 404 `invalid`.
Duplicate IDs → 409 `invalid` with an explicit message.

Real response for
`/applications/120321?fields=SK_ID_CURR,AMT_CREDIT,NAME_INCOME_TYPE`:

```json
{
  "application_id": 120321,
  "application": {
    "SK_ID_CURR": 120321,
    "AMT_CREDIT": 269550.0,
    "NAME_INCOME_TYPE": "Working"
  }
}
```

### `POST /predict` and `GET /applications/{id}/prediction`

Request body (exactly one input):

```json
{"application_id": 120321}
```

```json
{"features": {"AMT_CREDIT": 1000.0, "EXT_SOURCE_2": 0.35, "CODE_GENDER": "M"}}
```

`features` must contain exactly the model feature columns. Unknown keys, missing
keys, `TARGET`, non-numeric numeric values, non-finite values and values outside
the observed training range return 422 `invalid`. Explicit JSON `null` is
treated as missing and imputed.

`include_explanation=true` adds SHAP output. The probability is identical with
and without the explanation.

Real response:

```json
{
  "status": "available",
  "application_id": 120321,
  "probability": 0.4915030896663666,
  "risk_level": "MEDIUM",
  "model_name": "XGBoost",
  "model_version": "baseline-v2-local",
  "prediction_timestamp": "2026-11-04T14:38:51.296923+00:00",
  "explanation": null,
  "message": "Prediction generated successfully."
}
```

Real unknown-ID response (`GET /applications/999999999/prediction`):

```json
{
  "status": "no_prediction",
  "application_id": 999999999,
  "probability": null,
  "risk_level": null,
  "model_name": "XGBoost",
  "model_version": "baseline-v2-local",
  "prediction_timestamp": "2026-11-04T14:39:37.757617+00:00",
  "explanation": null,
  "message": "Application 999999999 was not found."
}
```

Real missing-model response (`503`):

```json
{
  "status": "model_unavailable",
  "application_id": 120321,
  "probability": null,
  "risk_level": null,
  "model_name": "XGBoost",
  "model_version": "unknown",
  "prediction_timestamp": "2026-11-04T14:39:42.778634+00:00",
  "explanation": null,
  "message": "Model artifacts are unavailable."
}
```

Real missing-data response (`503`):

```json
{
  "status": "unavailable",
  "application_id": 120321,
  "probability": null,
  "risk_level": null,
  "model_name": "XGBoost",
  "model_version": "baseline-v2-local",
  "prediction_timestamp": "2026-11-04T14:39:44.316215+00:00",
  "explanation": null,
  "message": "Application data is unavailable."
}
```

### `GET /model/evaluation`

`split=test` by default; `split=validation` is also accepted. Rows come only
from the saved `evaluation_results.json`; values are never recomputed or
defaulted. LSTM is absent. Rows can be passed directly to
`pandas.DataFrame(payload["rows"])`, which satisfies
`normalize_model_evaluation_results` (`model`, `metric`, `value`).

Real response (test split):

```json
{
  "split": "test",
  "rows": [
    {"model": "XGBoost", "metric": "ROC-AUC", "value": 0.7560700206065816},
    {"model": "XGBoost", "metric": "PR-AUC", "value": 0.23680797485855454},
    {"model": "XGBoost", "metric": "Precision", "value": 0.19301075268817205},
    {"model": "XGBoost", "metric": "Recall", "value": 0.5784103114930182},
    {"model": "XGBoost", "metric": "F1-score", "value": 0.2894383230314432}
  ],
  "provenance": {
    "artifact_version": "1.0",
    "model_name": "XGBoost",
    "model_version": "baseline-v2-local",
    "model_hash": "75fd9738cb1a709b47f64c89126e8c5344d3a4d6368ef16c6870c960360a8497",
    "generated_at": "2026-11-03T19:06:31.104064+00:00",
    "training_date": "2026-11-03T19:03:39.493127+00:00",
    "threshold": 0.5,
    "split": "test",
    "split_protocol": {
      "method": "stratified train/validation/test",
      "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
      "random_state": 42,
      "fit_preprocessor_on": "train only",
      "notes": "First split 70/30; second split 50/50 of the 30% holdout."
    },
    "n_samples": 46127,
    "class_counts": {"0": 42403, "1": 3724},
    "source": "evaluation_results.json",
    "lstm_available": false
  },
  "confusion_matrix": {
    "labels": [0, 1],
    "tn": 33397,
    "fp": 9006,
    "fn": 1570,
    "tp": 2154,
    "matrix": [[33397, 9006], [1570, 2154]]
  }
}
```

Missing or unreadable evaluation artifact → 503 `model_unavailable`.

### `GET /model/prediction-summary`

Returns counts computed from the saved `{split}_predictions.csv` records.
`prediction_count`, `predicted_outcomes`, and `risk_distribution` describe
those scored validation/test rows only; they do not represent predictions for
every application in `application_train.csv`. Risk bands use the canonical
values returned in `/models`. The source CSV is read and validated by the API;
the dashboard does not synthesize or recalculate these counts.

## Explanation schema

When `include_explanation=true` and the explainer is available, `explanation`
contains:

```text
method: "tree_shap"
output_space: "log_odds"
application_id: int | null
base_value: number
base_probability: number
prediction_margin: number
contribution_sum: number
probability: number
top_features: [
  {
    feature: string,
    raw_feature: string,
    transformed_feature: string,
    value: number,
    raw_value: number | string | null,
    shap_value: number,
    effect: "increases_risk" | "decreases_risk" | "neutral"
  }
]
summary: string
note: string
```

`value` is the transformed model input; `raw_value` is the source applicant
value when available. The dashboard uses `raw_value` for display, and uses the
SHAP contribution direction only as a model association, not as a causal claim.

## Risk bands (provisional)

Bands are returned as `risk_level` and exposed in `/models`:

```text
LOW    probability < 0.30
MEDIUM 0.30 <= probability < 0.70
HIGH   probability >= 0.70
```

They are provisional, not validated lending thresholds, and must not drive
automatic approval or rejection. The decision threshold is `0.5` for the saved
artifact.

## Change policy

- `docs/api_contract.md` is the single source of truth.
- Breaking changes (removing/renaming fields, changing status values, changing
  HTTP codes, changing probability semantics) require Person 2's agreement
  before implementation.
- Additive changes must be documented here and reflected in
  `docs/openapi.json`.
- `src/dashboard` and `src/agent` are owned by Person 2. The API may import
  `dashboard.data_access` read-only in tests but must not modify either tree.
