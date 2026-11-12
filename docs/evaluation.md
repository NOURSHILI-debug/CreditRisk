# Evaluation protocol

## Purpose

One fixed evaluation protocol for XGBoost, XGBoost with installment features,
and LSTM. The goal is a fair comparison on the same applicants.

## Split

The data is split with stratification on `TARGET` and a fixed random seed.

- Seed: `42`.
- First split: `70%` train, `30%` holdout.
- Second split: `50/50` of the `30%` holdout.
- Final ratios: `70%` train, `15%` validation, `15%` test.

The same split is used for every model.

## Rules

- The test set is touched only once, at the end. It is not used for tuning or
  model selection.
- Preprocessing is fitted on the training set only, then applied to validation
  and test.
- All models are compared on the same held-out applicants. Saved predictions
  keep `SK_ID_CURR` so rows can be aligned by applicant.

## Metrics

| Metric | Definition |
|---|---|
| ROC-AUC | Ranking quality across all thresholds. |
| PR-AUC | Area under the precision-recall curve (`average_precision_score`). |
| KS | Maximum separation between positive and negative score distributions. |
| Recall at fixed precision | Recall at an agreed precision operating point. |
| Brier score | Mean squared error between predicted probability and true label. |
| Calibration | Agreement between predicted probabilities and observed default rates. |

The current training code reports ROC-AUC, PR-AUC, precision, recall, F1,
confusion matrix, predicted positive rate, and observed positive rate at
threshold `0.5`. See the differences section.

## Imbalance handling

The current XGBoost run uses:

```text
scale_pos_weight = n_negative / n_positive
```

It is computed on the training split only and passed to `XGBClassifier`.

## Confidence intervals

Confidence intervals are produced by bootstrap resampling on the test set. Each
metric is reported with its bootstrap interval. The test labels are not used to
change the model.

## Results location

Results are written to `models/local_v2/`:

| File | Format | Contents |
|---|---|---|
| `evaluation_results.json` | JSON | Split protocol, seed, threshold, metrics, artifacts, baseline comparison. |
| `validation_predictions.csv` | CSV | `SK_ID_CURR`, `y_true`, `probability`, `prediction`. |
| `test_predictions.csv` | CSV | `SK_ID_CURR`, `y_true`, `probability`, `prediction`. |
| `training_metadata.json` | JSON | Parameters, timings, and artifact records. |

## LSTM leakage rule

For each application, build the sequence only from installment and payment data
available before the application date. Do not use later payments, the
application outcome, or test labels.

## Differences between this protocol and the current code

- Only XGBoost is implemented. `src/training/train_lstm.py` and the
  installment-feature training path are empty.
- The current code does not compute KS, recall at fixed precision, Brier score,
  or calibration.
- The current code does not compute bootstrap confidence intervals.
- The current code uses validation only as the `eval_set` during fitting; the
  test split is scored once after training.
- The current code has no explicit shared split file, but the saved
  `validation_predictions.csv` and `test_predictions.csv` include `SK_ID_CURR`
  so future models can be aligned to the same applicants.
