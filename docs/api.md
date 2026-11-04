# Credit Risk Intelligence API

FastAPI service that loads the XGBoost artifacts and the application dataset and
serves predictions, model metadata, evaluation results and SHAP explanations.

> **WARNING**
> These endpoints return applicant-derived data. Run the service on localhost
> only. Authentication and authorization are required before any real
> deployment; the current service has neither. Do not expose it to a network.

## Status values

Prediction responses currently use the legacy `status` values `"success"`,
`"error"` and `"unavailable"`. The API returns the probability of `TARGET=1`
(repayment difficulty). It is not an approval or rejection decision.

The canonical dashboard status mapping (`"available"` and related values) is
being defined in the API/dashboard contract step. Until then, the dashboard
normalizer does not accept the API's `"success"` value directly.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CREDIT_RISK_MODEL_DIR` | `models/local_v2` | Model artifact directory. Relative values resolve against the project root. |
| `CREDIT_RISK_APPLICATION_DATA` | `data/raw/application_train.csv` | Application CSV with `SK_ID_CURR`. Relative values resolve against the project root. |
| `CREDIT_RISK_CORS_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Comma-separated allowed dashboard origins. |
| `CREDIT_RISK_LOG_LEVEL` | `INFO` | Python log level. |

## Start the service (PowerShell)

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

The first startup loads the application CSV and model artifacts, so readiness
can take several seconds. Use `GET /health` to confirm readiness.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Model and data availability. Always returns 200; `status` is `ok` or `degraded`. |
| GET | `/models` | Model name, version, hash, training date, library versions and provisional thresholds. 503 when artifacts are missing. |
| GET | `/model/evaluation` | Raw `evaluation_results.json` contents. 503 when missing/unreadable. |
| GET | `/model/global-importance` | Raw `global_importance.json` contents with local paths removed. 404 when absent, 503 when unreadable. |
| POST | `/predict` | Predict by `application_id` or by `features` (exactly one). Query `include_explanation=true` adds SHAP. |
| GET | `/applications/{application_id}/prediction` | Predict by application ID. Query `include_explanation=true` adds SHAP. |

`GET /applications/{application_id}` and a paginated `GET /applications` are
planned in the application-endpoints step and are **not implemented yet**.

### Request bodies

```json
{"application_id": 120321}
```

```json
{"features": {"AMT_CREDIT": 1000.0, "EXT_SOURCE_2": 0.35, "CODE_GENDER": "M"}}
```

`features` must contain exactly the model's feature columns. Unknown keys,
missing keys, `TARGET`, non-numeric numeric values, non-finite values and values
outside the observed training range return 422. Explicit JSON `null` is treated
as missing and imputed by the fitted preprocessor.

## Real examples

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

### `POST /predict`

```json
{"application_id": 120321}
```

```json
{
  "status": "success",
  "application_id": 120321,
  "probability": 0.4915030896663666,
  "risk_level": "MEDIUM",
  "model_name": "XGBoost",
  "model_version": "baseline-v2-local",
  "prediction_timestamp": "2026-11-04T13:46:35.990983+00:00",
  "explanation": null,
  "message": "Prediction generated successfully."
}
```

### `GET /model/global-importance` (summary)

```json
{
  "method": "tree_shap",
  "feature_count": 120,
  "top_feature": "EXT_SOURCE_3",
  "top_mean_abs_shap": 0.4117468318986612
}
```

### Error example

`GET /applications/999999999/prediction` returns HTTP 404 with a JSON body whose
`status` is `"error"` and `probability` is `null`.

## Error codes

| Code | Meaning |
|---|---|
| 404 | Application ID not found. |
| 422 | Invalid request: both/neither inputs, unknown/missing features, invalid types or ranges. |
| 503 | Model artifacts or application data unavailable. |
| 500 | Unexpected server error (logged with `X-Request-ID`). |

Every response includes an `X-Request-ID` header; a client-supplied `X-Request-ID`
is echoed when printable and at most 64 characters.
