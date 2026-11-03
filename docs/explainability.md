# SHAP Explainability

This document covers the native XGBoost TreeSHAP layer for
`baseline-v2-local`. SHAP explains the existing model; it does not retrain,
tune, or change the prediction contract.

## Method

- TreeSHAP is computed with XGBoost directly:
  `booster.predict(DMatrix(X), pred_contribs=True)`.
- The `shap` package is not required.
- SHAP values are in **log-odds**. Positive values increase `P(TARGET = 1)`;
  negative values decrease it.
- The base value is `0.11322737485170364`; the corresponding base probability
  is `0.5282766402556108`.
- The base probability reflects `scale_pos_weight = 11.38746619094205` from
  training. It is **not** the population default rate and is **not calibrated**.
- Default predictions remain unchanged and do not compute SHAP. Explanations
  are generated only when requested with `include_explanation=True`.

## Global attribution

`models/local_v2/global_importance.json` is the dashboard-ready artifact.

- Source split: held-out test, 46,127 rows.
- Sample: 2,000 stratified rows, seed 42.
- Scope: 120 raw features; 306 transformed model features.
- Aggregation: for each row, sum SHAP values across transformed one-hot and
  missing-indicator columns for each raw feature; then average absolute values
  across the sampled rows.
- All 62 `num__missingindicator_*` columns are merged into their parent feature
  for global importance. Local `top_features` keep the `[missing]` detail.
- Units: `log_odds`.

Top raw features in the generated sample:

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `EXT_SOURCE_3` | 0.411747 |
| 2 | `EXT_SOURCE_2` | 0.351668 |
| 3 | `AMT_GOODS_PRICE` | 0.289577 |
| 4 | `AMT_CREDIT` | 0.248542 |
| 5 | `EXT_SOURCE_1` | 0.176940 |
| 6 | `CODE_GENDER` | 0.129686 |
| 7 | `DAYS_BIRTH` | 0.125184 |
| 8 | `DAYS_EMPLOYED` | 0.122800 |
| 9 | `AMT_ANNUITY` | 0.120342 |
| 10 | `NAME_EDUCATION_TYPE` | 0.098542 |

## Verification gates

The report generator fails instead of publishing inconsistent values if:

- reconstructed probabilities differ from `test_predictions.csv` by more than
  `1e-6`;
- SHAP additivity differs by more than `1e-4`;
- reconstructed probabilities from the SHAP output margin differ by more than
  `1e-6`.

Actual generated values:

| Check | Result | Tolerance |
|---|---:|---:|
| Sample max additivity error | `6.242479059892503e-06` | `1e-4` |
| Sample max probability reconstruction error | `7.085003705942228e-08` | `1e-6` |
| Full-test probability difference | `0.0` | `1e-6` |

## Local examples

Local examples use the same model and preprocessor as the predictor. The three
verified test applications are:

| `SK_ID_CURR` | Observed `TARGET` | Probability | Interpretation |
|---:|---:|---:|---|
| `172753` | 1 | `0.9589290618896484` | HIGH |
| `339432` | 0 | `0.0002128510` | LOW |
| `120321` | 0 | `0.4915030896663666` | BORDERLINE / MEDIUM |

The summary text names the largest risk-increasing and risk-decreasing
features. It is descriptive, not a lending recommendation.

## Predictor integration

```python
from inference.predictor import predict_credit_risk

result = predict_credit_risk(
    application,
    model_dir="models/local_v2",
    include_explanation=True,
)
```

With explanations enabled, `result["explanation"]` contains:

- `method`: `tree_shap`
- `output_space`: `log_odds`
- `application_id`, `probability`, `base_value`, `base_probability`
- `prediction_margin`
- `top_features`: display name, raw feature, transformed feature, value, SHAP
  value, and effect direction
- `summary`: descriptive top positive/negative features
- `note`: log-odds interpretation reminder

If explanation generation is requested and fails, the request returns an error
response rather than a prediction without an explanation. Default requests keep
`explanation = null`.

## Generated artifacts

| Path | Purpose |
|---|---|
| `models/local_v2/global_importance.json` | All 120 raw features ranked by mean \|SHAP\| |
| `docs/figures/shap/bar_top20.png` | Global importance bar chart |
| `docs/figures/shap/beeswarm_top20.png` | Beeswarm of top raw features |
| `docs/figures/shap/waterfall_172753.png` | Local waterfall for high-risk example |
| `docs/figures/shap/waterfall_339432.png` | Local waterfall for low-risk example |
| `docs/figures/shap/waterfall_120321.png` | Local waterfall for borderline example |
| `docs/figures/shap/top_contributions.csv` | Top 20 contributions for each sampled row |
| `docs/figures/shap/shap_metadata.json` | Scope, errors, versions, hashes, local examples |
| `docs/figures/shap/index.html` | Static report with figures and tables |

Regenerate with:

```powershell
.venv\Scripts\python.exe src\explainability\generate_shap_report.py
```

The default report uses `models/local_v2`, `test_predictions.csv`, and seed 42.
Use `--model-dir`, `--data`, `--predictions`, `--output-dir`,
`--importance-output`, `--sample-size`, `--seed`, and `--local-ids` to override.

## Interpretation caveats

- SHAP values are model attributions, not causal effects.
- Correlated features can share credit; do not read a single feature as an
  isolated cause.
- The model probabilities are not calibrated. Risk bands remain provisional and
  are not validated lending thresholds.
- One-hot categories are shown as `feature=value` locally; global importance is
  always aggregated back to the raw feature.
- A missing-indicator contribution is merged into its parent globally and shown
  as `feature [missing]` locally.
