# Credit Risk Intelligence Platform — Audit Executive Summary

**Repository:** `C:\Users\souro\Desktop\CreditRisk`
**Branch/commit audited:** `main` @ `4389013ae178e3b0280811b1f025b10b7ff9f118` (clean tree, up to date with `origin/main`)
**Full report:** `docs/PROJECT_FULL_TECHNICAL_AUDIT.md`
**Audit date:** 2026-11-18

---

## What works (implemented and verified)

| Area | What exists | Verification |
|---|---|---|
| XGBoost training | `src/training/train_xgboost.py`: stratified 70/15/15 split (seed 42), `scale_pos_weight` on train only, `XGBClassifier` (binary:logistic, 1000 estimators, depth 6, lr 0.05), metrics (ROC-AUC/PR-AUC/precision/recall/F1/confusion matrix/KS at threshold 0.5), artifact serialization with SHA-256 records | Code complete; unit-tested with synthetic fixtures. Not re-run on real data (dataset absent). |
| XGBoost inference + SHAP | `src/inference/predictor.py` (`CreditRiskPredictor`) + `src/explainability/shap_explainer.py` (native XGBoost TreeSHAP, `pred_contribs=True`, log-odds, additivity & probability-reconstruction checks) | Unit-tested with fixtures. `shap` package is NOT used/required. |
| FastAPI API | `src/api/main.py` + routers (`/health`, `/models`, `/model/evaluation`, `/prediction-summary`, `/global-importance`, `/applications`, `/predict`, `/{id}/prediction`), canonical statuses, field allowlist (`TARGET` excluded), CORS, uniform error shape | Contract- and unit-tested via `TestClient` + `httpx.MockTransport`. Not live-verified (no server on port 8000). |
| Streamlit dashboard | 4 pages (Overview, Application Explorer, Model Performance, AI Assistant), dual provider mode (API or deterministic mock), pagination, status-preserving prediction display | Verified in headless `AppTest` mock mode; all pages render without exceptions. |
| LangGraph agent | `src/agent/graph.py`: 6-tool allowlist (`get_dashboard_metrics`, `get_application_details`, `get_prediction`, `explain_prediction`, `get_model_comparison`, `generate_report`), Python-extracted/validated application IDs, LLM-input ID redaction, local rendering of applicant results, retrieved-data marked untrusted, lending-decision refusal | Tested with fake models (routing, privacy, injection, status preservation). Live LLM not run (opt-in only). |
| Report generation | `src/agent/report_generator.py`: deterministic, source-grounded, periods 7d/30d/90d/all, explicit period-coverage limitations (period not applied to snapshot sources) | Unit-tested including partial-source failures. |
| Test suite | 73 test files across `tests/unit` and `tests/integration` | **246 passed, 55 skipped, 0 failed** (audit execution). All skips due to absent `models/local_v2`, absent `data/raw/application_train.csv`, or opt-in Ollama integration. |

## What is incomplete / not implemented

| Component | Status | Key evidence |
|---|---|---|
| Dataset (`application_train.csv`) | ABSENT | `data/raw/` does not exist; gitignored; no download script |
| Kafka producer/consumer/validation/replay | NOT IMPLEMENTED | `src/ingestion/kafka_producer.py`, `kafka_consumer.py`, `event_validator.py` all 0 lines; schema only in `docs/event_schema.md` |
| S3 raw/processed storage | NOT IMPLEMENTED | `src/storage/s3_storage.py`, `local_storage.py` empty; Terraform (`aws/main.tf`) references an existing bucket only |
| Glue / Lambda | NOT IMPLEMENTED | `aws/glue/`, `aws/lambda/` contain empty READMEs only |
| PyTorch LSTM | NOT IMPLEMENTED | `train_lstm.py`, `sequence_builder.py`, `train_utils.py`, `training/preprocessing.py`, `test_lstm.py` all 0 lines; `torch` not a dependency |
| Evaluation/comparison module | EMPTY | `src/evaluation/*.py` all 0 lines; comparison logic lives in `dashboard/data_access.py` |
| Contracts/schemas | EMPTY | `src/contracts/*.py` and `schemas/*.json` all 0 lines |
| Monitoring | EMPTY | `src/monitoring/*.py` both 0 lines |
| Config YAMLs | EMPTY | `configs/kafka.yaml`, `xgboost.yaml`, `preprocessing.yaml`, `lstm.yaml`, `aws.yaml` all 0 lines |
| Docs | PARTIAL | `docs/architecture.md` empty; SageMaker doc stale (says ml.m5.large/$0.115, code says ml.g5.xlarge/$1.40) |

## Architecture overview (intended vs. actual)

**Intended:** Dataset → Kafka → Processing → S3 → XGBoost + LSTM → SageMaker → FastAPI → Streamlit + LangGraph agent.

**Actual:** Only the right-hand portion exists in code (XGBoost training/inference, FastAPI, Streamlit, agent, reports). The left-hand streaming/storage layer (Kafka, validation, S3, Glue/Lambda) and the LSTM branch have **zero implementation**. Data and model artifacts are absent, so the API cannot serve real predictions and `/health` would report `degraded`.

## Most important technical findings

1. **No live end-to-end flow has been verified.** The API was never running on port 8000 during validation; dataset and `models/local_v2/` are absent. Dashboard and agent are verified **only with synthetic mock data**.
2. **A fair XGBoost-vs-LSTM comparison is impossible today** — LSTM is not implemented. Comparison logic correctly withholds a head-to-head until both models have complete, protocol-compatible results.
3. **Security boundaries are code-enforced but not live-proven.** Allowlists, ID redaction, local applicant rendering, and untrusted-evidence prompts exist and are unit-tested; however, there is **no authentication or authorization on the API**, and live-LLM adherence to prompt-injection rules was not tested (opt-in only). Prompt injection is not "solved".
4. **Stored XGBoost metrics exist from a prior SageMaker run** (`sagemaker_run/model/evaluation_results.json`: test ROC-AUC 0.7564, PR-AUC 0.2362, threshold 0.5, 46,127 test rows) but the model binary and preprocessor are not in the repo.
5. **Environment drift:** README/CI target Python 3.11; local venv runs 3.13.3 with pandas 3.1.0. No lock file.

## Major blockers

1. `data/raw/application_train.csv` missing — blocks training, API data layer, all live flows.
2. `models/local_v2/` missing — blocks model loading, predictions, SHAP, `/health` = ok.
3. No running backend (port 8000) — blocks live dashboard/API integration and demonstration.
4. LSTM not implemented — blocks LSTM inference and XGBoost-vs-LSTM comparison.

## Immediate next steps

1. **Person 1:** place the authorized `application_train.csv` at `data/raw/`; run `python src/training/train_xgboost.py` to generate `models/local_v2/`; share artifact hashes/versions; implement or formally de-scope LSTM sequences/model/training and the Kafka/S3 path.
2. **Person 2:** start `uvicorn api.main:app` with artifacts in place and verify `GET /health` returns `ok`; run the dashboard against the live API; execute artifact-backed integration tests; run live LLM prompt-injection red-team.
3. **Shared:** complete `docs/architecture.md` and empty `configs/*.yaml`/`schemas/*.json`; add dependency lock file and CI coverage; add authentication before any real deployment; resolve Python 3.11 vs 3.13.3 version drift.

Detailed evidence for every statement above is in `docs/PROJECT_FULL_TECHNICAL_AUDIT.md`.