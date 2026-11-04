# Dashboard prediction integration boundary

A FastAPI prediction service is implemented in `src/api/` (see
[`docs/api.md`](api.md)). It returns the probability of `TARGET=1` from the
loaded XGBoost artifacts and can include SHAP explanations. The dashboard still
uses its mock provider by default and does not consume the API yet; it never
uses the synthetic `RISK_PROBABILITY` field or derives a prediction from
application data.

The API currently returns the legacy `status` value `"success"`. The canonical
dashboard status mapping (`"available"` and related values) is defined in the
API/dashboard contract step, after which an API-backed dashboard provider can
pass API payloads through `normalize_prediction_response`.

## Provider response

When an inference provider is implemented, the dashboard data-access adapter
should pass its response to `dashboard.data_access.normalize_prediction_response`
using this mapping:

```text
{
  "probability": 0.0,             # required numeric probability in [0, 1]
  "model_name": "provider value", # optional
  "model_version": "provider value", # optional
  "prediction_timestamp": "provider timestamp", # optional
  "explanation": "provider output" # optional; actual provider output only
}
```

The dashboard converts a valid `[0, 1]` probability to a percentage for display.
It does not assign a risk category or an approval decision. Missing model
metadata is displayed as `N/A`.

A provider may return `None` when it has no prediction, or a mapping with
`"status": "unavailable"`, `"status": "model_unavailable"`, or
`"status": "no_prediction"` and an optional `"message"`. Missing, non-numeric,
non-finite, or out-of-range probabilities are rejected and are never presented
as successful predictions. Explanations are displayed only when the provider
supplies them.

`get_credit_risk_prediction(application_id)` is the data-access boundary used by
the page. It obtains predictions through the selected dashboard data provider
and returns the existing `CreditRiskPrediction` response contract. The mock
provider explicitly reports prediction functionality as unavailable. No model,
prediction, or explanation is fabricated by the dashboard.

## Dashboard data provider

Provider selection is centralized in `dashboard.data_access` and uses the
`CREDIT_RISK_DATA_PROVIDER` environment variable. The default and only
implemented provider is `mock`. Its application records are synthetic and are
labeled as demo data in the dashboard. Selecting a provider name without an
adapter raises `DataSourceUnavailableError` for application and model-result
retrieval; prediction lookup returns an explicit `unavailable` response.

Provider adapters implement the `DashboardDataProvider` interface for
application records, predictions, and model evaluation rows. Application
records are pandas DataFrames; evaluation results are either `None` or a
DataFrame containing `model`, `metric`, and `value`. Invalid response types
raise `MalformedProviderResponseError`. Empty application frames and absent
evaluation results remain valid unavailable-data states, not fabricated data.