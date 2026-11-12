# Prediction schema

## Version

`schema-v1`

## Score meaning

`probability` is the probability that `TARGET = 1`, meaning repayment
difficulty (default). It is not an approval or rejection decision and is not
calibrated.

## Success response

Returned by `POST /predict` and `GET /applications/{application_id}/prediction`.

| Field | Type | Notes |
|---|---|---|
| `application_id` | integer or null | `SK_ID_CURR` when available. |
| `status` | string | `available` on success. |
| `probability` | number in `[0, 1]` | `P(TARGET = 1)`. |
| `risk_level` | `LOW`, `MEDIUM`, `HIGH`, or null | Provisional band from `probability`. |
| `model_name` | string | Example: `XGBoost`. |
| `model_version` | string | Example: `baseline-v2-local`. |
| `prediction_timestamp` | string | ISO 8601 UTC timestamp. |
| `explanation` | object or null | SHAP block when `include_explanation=true`. |
| `message` | string | Human-readable result message. |

The current API uses `risk_level`, not `risk_label`, and
`prediction_timestamp`, not `timestamp`. It does not return `threshold` in the
prediction response. The decision threshold is exposed by `GET /models` as
`decision_threshold`; the saved artifact uses `0.5`. Risk bands are
`LOW < 0.30`, `MEDIUM 0.30-0.60`, and `HIGH >= 0.60`.

## Status values

| Status | HTTP | When it is used |
|---|---:|---|
| `available` | 200 | A valid probability in `[0, 1]` was produced. |
| `no_prediction` | 200 or 404 | The provider has no prediction, or the application ID is unknown. |
| `invalid` | 409, 422, or 500 | Malformed input, duplicate ID, unknown field, or invalid model output. |
| `unavailable` | 503 | The application data source is unavailable. |
| `model_unavailable` | 503 | Model or preprocessor artifacts are missing or incompatible. |

## Error response

Every non-2xx response uses the same keys. `probability` is always `null` and is
never replaced by mock data or a default prediction.

| Field | Type | Notes |
|---|---|---|
| `status` | string | One of the canonical status values. |
| `message` | string | Reason for the failure. |
| `application_id` | integer or null | Present when known. |
| `probability` | null | Always `null` on errors. |
| `risk_level` | null | Always `null` on errors. |
| `model_name` | string or null | Present when known. |
| `model_version` | string or null | Present when known. |
| `prediction_timestamp` | string or null | ISO 8601 UTC timestamp when available. |
| `explanation` | null | Always `null` on errors. |

The code does not use a separate `error_code` field. It uses `status` plus
`message`.

## Insufficient history

For the future LSTM path, an application without enough repayment history must
return `status: "no_prediction"` with `probability: null` and a clear reason in
`message`, for example: `"Insufficient repayment history for sequence
prediction."` No sequence or probability is fabricated.

## SHAP explanation

When `include_explanation=true`, `explanation` contains:

| Field | Type | Notes |
|---|---|---|
| `method` | string | `tree_shap`. |
| `output_space` | string | `log_odds`. |
| `application_id` | integer or null | Applicant ID when available. |
| `base_value` | number | Model base value in log-odds. |
| `base_probability` | number | Sigmoid of `base_value`; not calibrated. |
| `prediction_margin` | number | Base value plus all SHAP contributions. |
| `contribution_sum` | number | Sum of SHAP values. |
| `probability` | number | Same `P(TARGET = 1)` as the prediction. |
| `top_features` | array | Top local feature contributions. |
| `summary` | string | Descriptive top risk-increasing and risk-decreasing features. |
| `note` | string | Reminder that values are additive in log-odds. |

Each `top_features` item has:

| Field | Type | Notes |
|---|---|---|
| `feature` | string | Display name, for example `EXT_SOURCE_3`. |
| `raw_feature` | string | Raw source column. |
| `transformed_feature` | string | Transformed model feature. |
| `value` | number | Transformed input value. |
| `raw_value` | number, string, or null | Source applicant value when available. |
| `shap_value` | number | SHAP contribution in log-odds. |
| `effect` | string | `increases_risk`, `decreases_risk`, or `neutral`. |

Positive `shap_value` increases `P(TARGET = 1)`; negative decreases it. SHAP
values are model attributions, not causal effects.

## Examples

### available

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

### no_prediction (future LSTM case: insufficient history)

The current XGBoost API uses `no_prediction` for an unknown application ID
(HTTP 404). The example below shows the required future LSTM behavior.

```json
{
  "status": "no_prediction",
  "application_id": 120321,
  "probability": null,
  "risk_level": null,
  "model_name": "LSTM",
  "model_version": "unknown",
  "prediction_timestamp": "2026-11-11T10:15:30+00:00",
  "explanation": null,
  "message": "Insufficient repayment history for sequence prediction."
}
```

### invalid

```json
{
  "status": "invalid",
  "message": "Invalid request.",
  "application_id": null,
  "probability": null,
  "risk_level": null,
  "model_name": "XGBoost",
  "model_version": "baseline-v2-local",
  "prediction_timestamp": "2026-11-11T10:15:30+00:00",
  "explanation": null
}
```

### unavailable

```json
{
  "status": "unavailable",
  "message": "Application data is unavailable.",
  "application_id": 120321,
  "probability": null,
  "risk_level": null,
  "model_name": "XGBoost",
  "model_version": "baseline-v2-local",
  "prediction_timestamp": "2026-11-04T14:39:44.316215+00:00",
  "explanation": null
}
```

### model_unavailable

```json
{
  "status": "model_unavailable",
  "message": "Model artifacts are unavailable.",
  "application_id": 120321,
  "probability": null,
  "risk_level": null,
  "model_name": "XGBoost",
  "model_version": "unknown",
  "prediction_timestamp": "2026-11-04T14:39:42.778634+00:00",
  "explanation": null
}
```

## Differences from the requested/current docs

- The code returns `risk_level`; the request asks for `risk_label`.
- The code returns `prediction_timestamp`; the request asks for `timestamp`.
- The prediction response has no `threshold` field. The threshold is exposed by
  `GET /models` as `decision_threshold` and is `0.5` in the saved artifact.
- Errors have no `error_code` field. They use `status` and `message`, plus the
  standard null-filled response keys.
- `docs/api_contract.md` states `HIGH >= 0.70`; the code in
  `src/inference/predictor.py` and `docs/model_contract.md` use `HIGH >= 0.60`.
- The legacy `predict_credit_risk()` helper returns `success`, `error`, or
  `unavailable`; only the HTTP API uses the canonical status values above.
