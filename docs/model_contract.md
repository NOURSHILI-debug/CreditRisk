# Model Contract

This contract covers the local XGBoost baseline (`baseline-v2-local`) and the
artifacts produced by `src/training/train_xgboost.py`.

## Probability semantics

The returned `probability` is **P(TARGET = 1)**, the probability of repayment
difficulty. It is the positive-class column of `model.predict_proba(...)`.

This is verified by
`tests/unit/test_predictor.py::test_probability_is_p_target_1_from_predict_proba`,
which compares the predictor output with
`model.predict_proba(preprocessor.transform(X))[:, index_of_class_1]`.

## Prediction response

`predict_credit_risk(application, model_dir=None, include_explanation=False)`
returns a dictionary with these fields:

| Field | Type | Notes |
|---|---|---|
| `status` | string | `success`, `error`, or `unavailable` |
| `application_id` | integer or null | `SK_ID_CURR` when present and valid |
| `probability` | float in `[0, 1]` or null | `P(TARGET = 1)`; null on failure |
| `risk_level` | `LOW`, `MEDIUM`, `HIGH`, or null | Provisional bands; null on failure |
| `model_name` | string | `XGBoost` |
| `model_version` | string | `baseline-v2-local` |
| `model_hash` | string or null | SHA256 of the native JSON model artifact |
| `prediction_timestamp` | string | ISO 8601 UTC timestamp |
| `explanation` | object or null | Native local TreeSHAP when requested; null by default |
| `message` | string | Human-readable result or failure message |

`include_explanation` defaults to `False`. Default predictions never compute
SHAP. When it is `True`, the response adds `explanation` with the local
attribution payload described below.

Success example fields:

```json
{
  "status": "success",
  "application_id": 100002,
  "probability": 0.9290850758552551,
  "risk_level": "HIGH",
  "model_name": "XGBoost",
  "model_version": "baseline-v2-local",
  "model_hash": "75fd9738cb1a709b47f64c89126e8c5344d3a4d6368ef16c6870c960360a8497",
  "prediction_timestamp": "2026-11-03T19:06:31.129000+00:00",
  "explanation": null,
  "message": "Prediction generated successfully."
}
```

### Failure behavior

- Missing required feature: `status="error"`, `probability=null`, message names
  the missing columns.
- Invalid input type or more than one row: `status="error"`,
  `probability=null`.
- Model or preprocessor missing/unloadable: `status="unavailable"`,
  `probability=null`.
- There is **no default probability** and **no silent fallback to mock data**.
- An unseen categorical value is handled by
  `OneHotEncoder(handle_unknown="ignore")`; it does not crash and does not
  produce a fabricated value.

### Explanation

`explanation` is `null` by default. With `include_explanation=True`, the
predictor produces native XGBoost TreeSHAP values via
`booster.predict(DMatrix(X), pred_contribs=True)`.

The local payload contains:

| Field | Type | Notes |
|---|---|---|
| `method` | string | `tree_shap` |
| `output_space` | string | `log_odds` |
| `application_id` | integer or null | `SK_ID_CURR` when present |
| `base_value` | float | Model base value in log-odds |
| `base_probability` | float | Sigmoid of the base value; not calibrated |
| `prediction_margin` | float | Base value plus all SHAP contributions |
| `probability` | float | `P(TARGET = 1)` from `predict_proba` |
| `top_features` | list | Top local contributions with display/raw/transformed names, value, SHAP value, and effect |
| `summary` | string | Descriptive largest risk-increasing/decreasing features |
| `note` | string | Log-odds interpretation reminder |

SHAP values are model attributions, not causal effects. Feature importance is
not substituted for SHAP values. Local names retain one-hot categories and
`[missing]` indicators; global importance merges them back to raw features.

If explanation generation is requested and fails, the response is an error
response with `probability=null`; it does not return a prediction without the
requested explanation. See `docs/explainability.md` for the full method,
caveats, generated figures, and verification tolerances.

### Risk bands

The risk bands are **provisional and not validated lending thresholds**:

- `LOW`: `probability < 0.30`
- `MEDIUM`: `0.30 <= probability < 0.60`
- `HIGH`: `probability >= 0.60`

They are decision support only, not automatic approval or rejection.

## Artifacts

Local baseline artifacts are written to `models/local_v2/` first. Promotion to
`models/` requires explicit approval.

| Artifact | Format | Purpose |
|---|---|---|
| `models/local_v2/xgboost_baseline.json` | XGBoost native JSON | Model loaded by `XGBClassifier.load_model` |
| `models/local_v2/preprocessor.pkl` | joblib sklearn `Pipeline` | Fit on train only; transforms raw application columns |
| `models/local_v2/feature_columns.json` | JSON list of strings | Raw input feature names for `predictor.py` |
| `models/local_v2/preprocessor_metadata.json` | JSON | Numeric/categorical/post-transform feature names |
| `models/local_v2/model_metadata.json` | JSON | Model name/version/hash used by the predictor |
| `models/local_v2/global_importance.json` | JSON | All 120 raw features ranked by mean \|SHAP\| |
| `models/local_v2/training_metadata.json` | JSON | Parameters, versions, timing |
| `models/local_v2/evaluation_results.json` | JSON | Validation/test metrics and artifact hashes |
| `models/local_v2/validation_predictions.csv` | CSV | `SK_ID_CURR`, `y_true`, `probability`, `prediction` |
| `models/local_v2/test_predictions.csv` | CSV | `SK_ID_CURR`, `y_true`, `probability`, `prediction` |

Tracked SHAP report artifacts live in `docs/figures/shap/`:
`bar_top20.png`, `beeswarm_top20.png`, three local waterfalls,
`top_contributions.csv`, `shap_metadata.json`, and `index.html`.

Pinned runtime for the artifacts:

- Python `3.13.5`
- `xgboost==3.2.0`
- `scikit-learn==1.6.1`
- `pandas==3.1.0`
- `numpy==2.5.4`
- `scipy==1.18.1`
- `joblib==1.6.0`
- `matplotlib==3.11.2`

The preprocessor is a single sklearn `Pipeline`:

1. Replace `DAYS_EMPLOYED == 365243` with `NaN`.
2. Numeric: median imputation with missing indicators.
3. Categorical: most-frequent imputation plus
   `OneHotEncoder(handle_unknown="ignore")`.
4. Drop `SK_ID_CURR` and `TARGET` from model features.

The model uses the artifact hash in `model_hash`. `model_version` remains
`baseline-v2-local`; the full SHA256 is provided separately.

## Evaluation artifact schema

`models/local_v2/evaluation_results.json` includes:

- `model_name`, `model_version`, `model_hash`
- `training_date`, `generated_at`
- `target` (including `probability_semantics = "P(TARGET=1)"`)
- `seed`, `threshold` (0.5)
- `split_protocol` and `splits` (70/15/15 stratified, seed 42)
- `scale_pos_weight` (computed on train only)
- `metrics.validation` and `metrics.test`, each containing:
  `threshold`, `n_samples`, `class_counts`, `roc_auc`, `pr_auc`, `precision`,
  `recall`, `f1`, `confusion_matrix`, `predicted_positive_rate`,
  `observed_positive_rate`
- `metric_definitions`
- `artifacts` with relative paths, filenames, SHA256 hashes, and sizes
- `library_versions`
- `baseline_reference` and `baseline_comparison` (context only; no tuning)

Actual local test metrics from this run:

| Metric | Local test | Colab reference | Delta |
|---|---:|---:|---:|
| ROC-AUC | 0.756070 | 0.7571 | -0.001030 |
| PR-AUC | 0.236808 | 0.2386 | -0.001792 |
| Precision (0.5) | 0.193011 | 0.1943 | -0.001289 |
| Recall (0.5) | 0.578410 | 0.5835 | -0.005090 |
| F1 (0.5) | 0.289438 | 0.2915 | -0.002062 |

All deltas are within the 0.01 context tolerance. No metric was tuned or
hardcoded.

Artifact SHA256 values from this run:

- Model: `75fd9738cb1a709b47f64c89126e8c5344d3a4d6368ef16c6870c960360a8497`
- Preprocessor: `cc225934d0766b08f1fcb1cdda99bf915437c397470a0a775a701bad29eabe34`
- Feature columns: `da3d28bad4d5ed351949a11d0d4c09d24855c085eaecdf9089d79bb4563e59ab`
- Model metadata: `8a7f6e5d07e23110a69fa49ad7186c1fef176c1de947a6edafb933faf8c1a506`

The predictor reads `model_metadata.json` when present. If it is absent (the
old `models/` artifacts), the predictor falls back to `model_version="baseline-v1"`.
This run leaves `models/local_v2/` as the v2 location; promotion to `models/`
is pending approval.

## Commands

```powershell
# Train and write models/local_v2 artifacts
.venv\Scripts\python.exe src\training\train_xgboost.py

# Generate global importance and SHAP figures
.venv\Scripts\python.exe src\explainability\generate_shap_report.py

# Run tests
.venv\Scripts\python.exe -m pytest tests/unit -q

# Dependency consistency
.venv\Scripts\python.exe -m pip check
```

`CREDIT_RISK_MODEL_DIR` can point the predictor at another artifact directory
(for example `models/local_v2`).

## Open questions for Person 2

1. What exact input and output does `normalize_model_evaluation_results`
   expect? The function is not present in `src/dashboard/data_access.py` yet.
2. What wrapper signature and return type should `get_credit_risk_prediction`
   use? `CreditRiskPrediction` is not defined yet.
3. Should the dashboard read `models/` or `models/local_v2/` before promotion?
4. Does the dashboard expect `model_version` to include the artifact hash, or is
   the separate `model_hash` field preferred?
5. Should `risk_level` be `LOW`/`MEDIUM`/`HIGH` or display labels such as
   `Low Risk`?
6. Which local `top_features` fields should the dashboard display, and should
   it default `include_explanation` to `False` or `True`?
