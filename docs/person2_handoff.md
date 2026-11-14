# Person 2 Technical Handoff — Groups A–D

**Project:** Credit Risk Intelligence Platform  
**Owner:** Person 2 — Dashboard, LLM Agent and Application Engineering  
**Prepared:** 2026-11-14  
**Current branch:** `feature/person2-e2e-validation`

## Executive summary

The current branch contains the Person 2 Group A–D implementation represented by the commit history below. Group A and Group B functionality landed together in the Group B commit; there is no standalone Group A commit. The historical branch `feature/person2-group-a-fixes` currently points to the same commit as `main`, so its name alone is not evidence of a separate Group A patch. The mock provider and related Group A behavior are nevertheless present in the current branch through commit `7896917`.

Groups C and D are separate commits and are ancestors of the current branch. The branch has three commits beyond `origin/main`, and at audit time it matched `origin/feature/person2-e2e-validation` with a clean worktree. No Group A–D source change was found missing from this branch. The current branch does **not** establish live API/model integration: the API was not running, and the expected dataset and model artifact directory were absent during Group D validation.

The branch's base history also includes the earlier Ollama integration (`130a173`, merged before Groups A–D). `src/agent/graph.py` supports Ollama and optional OpenAI configuration; this provider work is inherited baseline functionality, not a new Group A–D commit.

The most important integration gates are the real application data and XGBoost artifacts for the currently implemented inference path, plus an available backend process. LSTM artifacts alone are not sufficient: repository documentation says the LSTM training and installment-feature paths are empty, so a sequence-generation and inference implementation/API path must also be agreed and delivered before LSTM results can be integrated.

## Git branch and commit audit

At the start of the handoff task:

```text
HEAD: feature/person2-e2e-validation
HEAD: 4b4e992fd54ad381d154d077b7b3cdcb091527fc
origin/feature/person2-e2e-validation: same commit
Commits ahead of origin/main: 3
Worktree: clean
```

The three commits ahead of `origin/main`, in order, are:

| Group attribution | Commit | Files / evidence |
|---|---|---|
| Groups A + B | `78969172ed0e3bd7c07d2ad5b45c1b03537051af` — `feat: integrate dashboard and agent with API contracts` | Adds `src/dashboard/providers/mock_provider.py` and its tests while extending API-backed data access, dashboard pages, agent wrappers, and reports. The commit is mixed-scope; no independent Group A commit is in the current ancestry. |
| Group C | `8df1e1685e978dd54b95a79a75ab9b24b9d1cbb4` — `feat(agent): harden prompt injection protection` | Changes `src/agent/prompts.py` and adds user-input/retrieved-data injection tests in `tests/unit/test_agent.py`. |
| Group D | `4b4e992fd54ad381d154d077b7b3cdcb091527fc` — `test(person2): complete dashboard and agent validation` | Adds `docs/person2_validation_report.md` and Streamlit tests for dashboard search, details, pagination, and entrypoint startup. |

The `feature/person2-group-a-fixes` and `feature/optional-mock-data-provider` local branch names point to `0279866`, the same base commit as `main`; they do not add a separate commit to this branch. The source history shows the mock provider arriving in the combined A/B commit above. This is a commit-attribution caveat, not a missing implementation in the current tree.

No changes were made to application code during this handoff task. This handoff document is the only file added by this task. No commit or push was made.

## Groups A–D implementation details

### Group A — Mock provider and reliability

Implemented behavior:

- The dashboard defaults to the real FastAPI source. A user can explicitly select `api` or `mock` in the sidebar; the initial setting can also be selected through `CREDIT_RISK_DATA_MODE`.
- API errors remain errors. Data-access failures do not silently route to mock data.
- The mock provider creates a deterministic, in-memory, Home Credit-shaped synthetic sample. It labels records and predictions as illustrative, does not expose `TARGET` in applicant browsing, and does not write generated application files into the repository.
- Mock predictions and explanatory factors are illustrative examples, not trained-model inference or SHAP. Mock mode does not provide evaluation scores.
- Model comparison requires complete model metrics and compatible split/protocol provenance. Missing metrics remain unavailable; synthetic results cannot qualify as a trained-model comparison.
- Report sections expose unavailable data and unsupported time coverage. Aggregate/snapshot sources without timestamped records are not presented as period-filtered results.

There is no separate Group A commit. These capabilities are included in commit `7896917` with the API integration work.

### Group B — Real API integration

Implemented behavior:

- The dashboard and agent share `src/dashboard/data_access.py` as the provider adapter. In API mode it calls the FastAPI client; in explicit mock mode it dispatches to the synthetic provider.
- The existing API client is `src/dashboard/api_client.py`. It existed before the Group B commit and is reused rather than replaced by a new client in that commit.
- Dashboard pages consume API-backed health, application pages/details, prediction and explanation responses, model metadata, evaluation results, and prediction summaries through the shared adapter.
- Application results are paginated and filtered through API request parameters. Application lookup distinguishes invalid IDs, missing records, duplicates, and unavailable data.
- Prediction responses preserve meaningful statuses such as `available`, `no_prediction`, `not_found`, `unavailable`, `model_unavailable`, `invalid`, and `error`; absent probabilities are not replaced with zero.
- Agent tools use the same configured data-access path for application details, prediction/explanation, aggregate metrics, model comparison, and report generation.
- Existing contract documentation and backend schemas specify the HTTP response shapes. Unit tests with `httpx.MockTransport` validate client/adapter contract behavior.

**Live integration caveat:** These contracts and mocked transport tests do not prove that a running local backend has the required files or returns real data. Group D attempted `GET http://127.0.0.1:8000/health`; the connection failed and there was no listener on port 8000. `data/raw/application_train.csv` and `models/local_v2` were absent. API integration tests that require those artifacts were skipped.

### Group C — Agent security

Implemented and preserved behavior:

- LangGraph structured routing selects among the existing six tools: `get_dashboard_metrics`, `get_application_details`, `get_prediction`, `explain_prediction`, `get_model_comparison`, and `generate_report`.
- Python validates route decisions against the allowlist and extracts/validates application IDs independently of model-supplied tool arguments.
- Application IDs are redacted from the router's LLM input. Applicant-specific tool results are rendered locally and are not passed into the response LLM. Application lookup is field-allowlisted and does not expose `TARGET`.
- Prompts require answers to use retrieved evidence, distinguish unavailable/partial data, avoid invented metrics, predictions, timestamps, and explanations, and label mock data.
- Group C explicitly treats user input and retrieved tool values as untrusted content. Tests exercise hostile instructions in both locations and assert the routing/privacy and response-prompt boundaries.
- Tool exceptions and unavailable data become safe, explicit statuses/messages; they are not converted into optimistic risk values or fabricated findings.

The prompt boundary is tested with unit/fake-model tests. No live model was used to claim that an LLM will always follow those instructions.

### Group D — End-to-end validation

Group D added headless UI checks using Streamlit `AppTest` for:

- Entry-point rendering in explicit mock mode.
- Application ID search followed by loading selected details.
- Pagination from offset 0 to offset 25, including the final-page row/count display.
- Existing selector checks for API-default and explicit provider switching.

Existing tests also cover unavailable API behavior, missing prediction statuses, evaluation provenance and missing metrics, agent tool execution/failures, injection protections, and report coverage/partial failures.

The complete suite recorded in `docs/person2_validation_report.md` was **239 passed, 55 skipped**. The 55 skips require missing application/model artifacts or an explicitly enabled live Ollama integration. The dashboard startup verification used in-process `AppTest`, not a separate browser/server launch. Group D did not verify an actual backend.

## File-level implementation inventory

| Area | Relevant files | Contribution / note |
|---|---|---|
| Provider configuration and mock data | `src/dashboard/app.py`, `src/dashboard/data_access.py`, `src/dashboard/providers/__init__.py`, `src/dashboard/providers/mock_provider.py` | Provider selection and dispatch, explicit API default, deterministic synthetic provider, no silent fallback. `app.py` and `data_access.py` changes are in the combined A/B commit. |
| Dashboard pages | `src/dashboard/pages/overview.py`, `src/dashboard/pages/application_explorer.py`, `src/dashboard/pages/model_comparison.py` | API-backed overview, application search/pagination/details/prediction statuses, and provenance-aware metrics display. |
| Existing API client | `src/dashboard/api_client.py` | Existing HTTP client reused by the Group B adapter. It was not among the files changed by commit `7896917`. |
| Agent workflow and privacy | `src/agent/graph.py`, `src/agent/prompts.py`, `src/agent/tools.py`, `src/agent/privacy.py`, `src/agent/state.py` | Existing graph, allowlist, tool wrappers, prompt policy, context, and privacy rules. Group B changed graph/prompts/tools; Group C changed prompts and added injection tests. |
| Report generation | `src/agent/report_generator.py` | Aggregate sourced report, partial/unavailable sections, and explicit period coverage/limitations. |
| Contracts and run documentation | `src/api/schemas.py`, `src/api/routers/`, `docs/api_contract.md`, `docs/prediction_schema.md`, `docs/evaluation.md`, `docs/development_setup.md` | Backend contracts and local setup instructions; Group B updated evaluation/setup docs, while the API schemas and API contract documents predate these A–D commits. |
| Group A/B tests | `tests/unit/test_mock_provider.py`, `tests/unit/test_dashboard_data_access.py`, `tests/unit/test_agent.py`, `tests/unit/test_application_explorer.py`, `tests/unit/test_model_comparison.py`, `tests/unit/test_report_generator.py` | Provider behavior, adapter contracts, tool outputs/statuses, dashboard behavior, comparison provenance, and report coverage. |
| Group C tests | `tests/unit/test_agent.py` | Routing allowlist, ID privacy, prompt injection in user input and retrieved content, errors, grounding/status behavior. |
| Group D tests and report | `tests/unit/test_application_explorer.py`, `tests/unit/test_mock_provider.py`, `docs/person2_validation_report.md` | New UI workflows and detailed validation record with live-vs-mocked distinctions. |
| Live integration suites | `tests/integration/test_dashboard_api.py`, `tests/integration/test_api_integration.py`, `tests/integration/test_ollama_agent.py` | Existing integration checks; artifact-backed checks skip without dataset/models, and Ollama checks require explicit opt-in. |

The agent model factory in `src/agent/graph.py` defaults to OpenAI and supports Ollama through `AGENT_LLM_PROVIDER=ollama` with the configured local model. OpenAI remains an alternative provider requiring its own API key. The opt-in Ollama integration tests are separate from mocked graph tests.

## Person 2 completion matrix

Status meanings: **COMPLETE** means implementation and the stated local test evidence are present; **PARTIALLY COMPLETE** means the code/contracts are present but only some runtime scenarios are verified; **BLOCKED** means an external artifact/service is required to finish validation; **NOT IMPLEMENTED** means the feature is absent from the current implementation.

| Feature | Implementation status | Relevant source files | Tests performed | Dependencies | Remaining limitations |
|---|---|---|---|---|---|
| Real API default and explicit provider switching | **COMPLETE** | `src/dashboard/app.py`, `src/dashboard/data_access.py` | Unit/AppTest provider default and selector switch | None for local selection behavior | Live API remains offline during Group D validation. |
| Synthetic provider, labels, and no silent fallback | **COMPLETE** | `src/dashboard/providers/mock_provider.py`, `src/dashboard/data_access.py`, dashboard pages | Mock provider and page tests; forced API error tests | None for synthetic demo behavior | Mock data is not real applicant data or model performance. |
| Application ID search, detail, and pagination | **COMPLETE** | `src/dashboard/pages/application_explorer.py`, `src/dashboard/data_access.py`, `src/dashboard/api_client.py` | AppTest search/detail/pagination; API transport contract tests | Running API and real dataset for live validation | No real records were queried in this handoff. |
| XGBoost prediction and status handling | **PARTIALLY COMPLETE** | `src/dashboard/data_access.py`, `src/dashboard/api_client.py`, `src/agent/tools.py`, `src/api/routers/predictions.py` | Unit status/error tests and mocked transport contract tests; artifact-backed integration skipped | Backend, dataset, XGBoost/preprocessor artifacts | Real inference endpoint was not exercised. |
| SHAP prediction explanation | **PARTIALLY COMPLETE** | `src/agent/tools.py`, dashboard `application_explorer.py`, `src/api/routers/predictions.py`, `docs/prediction_schema.md` | Unit/schema tests; real artifact-backed tests skipped | Running API and compatible XGBoost/preprocessor/SHAP dependencies | No live SHAP response was verified. |
| Model comparison with provenance | **PARTIALLY COMPLETE** | `src/dashboard/data_access.py`, dashboard `model_comparison.py`, `src/agent/tools.py`, `docs/evaluation.md` | Unit tests for missing values, split/protocol mismatch, synthetic results | Real evaluation artifacts for both models and matching provenance | Only XGBoost is implemented per `docs/evaluation.md`; no LSTM results are available. |
| Aggregate report and period coverage | **PARTIALLY COMPLETE** | `src/agent/report_generator.py`, `src/agent/tools.py` | Unit tests for available/partial/unavailable sources and unsupported period filtering | Backend summary/evaluation sources | Real-data report content and timestamp-filtered coverage were not validated. |
| Six-tool LangGraph routing and safe failure handling | **COMPLETE** | `src/agent/graph.py`, `src/agent/tools.py`, `src/agent/state.py` | Fake-model/tool unit tests across the six allowlisted tools and failures | LLM provider for live natural-language routing; API for real data | Test behavior is simulated, not a live agent session. |
| Agent privacy and prompt injection boundaries | **COMPLETE** | `src/agent/privacy.py`, `src/agent/prompts.py`, `src/agent/graph.py` | Unit tests for ID redaction, local applicant responses, injected user/retrieved content | Live LLM to evaluate provider adherence | Tests verify prompts/boundaries, not universal model compliance. |
| LSTM model inference and sequence generation | **NOT IMPLEMENTED** | `docs/evaluation.md`, future API/model integration | Artifact-dependent test coverage does not implement this feature | LSTM pipeline, sequence contract, model artifact, API path | `docs/evaluation.md` says `train_lstm.py` and installment-feature training are empty. |
| Backend and artifact-backed end-to-end validation | **BLOCKED** | API client/adapters and `tests/integration/` | Live `/health` connection attempt failed; artifact-backed tests skipped | Running API, application data, XGBoost artifacts | Cannot claim end-to-end production-source validation until delivered. |
| Streamlit dashboard headless startup/workflows | **COMPLETE** | `src/dashboard/app.py`, `tests/unit/test_application_explorer.py`, `tests/unit/test_mock_provider.py` | AppTest entrypoint, search/details, pagination, provider switching | Streamlit installed in project environment | Browser/server interaction was not tested. |

## Person 1 dependencies and requested deliverables

The statuses below distinguish current hard blockers from conditional model capabilities and coordination items. Existing API schemas/docs are useful, but they do not substitute for a running service and artifacts.

| # | Dependency and priority | Why Person 2 needs it / feature affected | Exact deliverable requested from Person 1 | Current state / classification |
|---:|---|---|---|---|
| 1 | Running backend API and health endpoint — **mandatory blocker** | Verifies dashboard/API connectivity and gates all live application, prediction, evaluation, and agent retrieval checks. | Start command or deployment URL; reachable `GET /health` response showing `status`, `data_available`, `model_loaded`, model name/version/hash; confirm bind host/port. | No listener at `127.0.0.1:8000` during Group D. Live API verification **BLOCKED**. |
| 2 | Real application dataset and processed data — **mandatory blocker** | Application search/details and ID-based inference require the actual data source with expected columns and IDs. | Authorized dataset location, format/schema/version, preprocessing expectations, record count, and exact local path or storage location; confirm data load through health endpoint. | `data/raw/application_train.csv` absent. Live lookup/inference **BLOCKED**. |
| 3 | XGBoost model artifacts — **mandatory for XGBoost runtime** | Required for actual score generation, SHAP, and artifact-backed model/evaluation metadata. | Complete compatible `models/local_v2` artifact bundle (model, preprocessor, feature columns, metadata, evaluation and prediction artifacts) with hashes/versions and placement instructions. | `models/local_v2` absent. Real XGBoost prediction/evaluation **BLOCKED**. |
| 4 | LSTM artifacts and sequence-generation interface — **conditional blocker for LSTM** | Needed only to claim LSTM inference/evaluation or a two-model comparison; dashboard currently shows LSTM unavailable. An artifact without a compatible sequence builder/API is insufficient. | Either (a) state LSTM is out of scope and confirm UI should keep reporting unavailable, or (b) deliver implemented sequence-generation contract, temporal cutoff/leakage rules, model artifact and inference/API response examples. | `docs/evaluation.md` says LSTM training and installment-feature paths are empty. LSTM inference is **NOT IMPLEMENTED**; head-to-head comparison is blocked until both models exist. |
| 5 | Prediction API contract and example responses — **mandatory handshake; contract already documented** | Person 2’s client normalizes field names/statuses; a mismatch can break prediction display and agent responses. | Confirm the running service implements `docs/api_contract.md` and `docs/prediction_schema.md`; provide sanitized examples for available, no-prediction/unknown ID, unavailable, model-unavailable, invalid, and explanation responses. | Contract schemas/docs exist in repo; deployed response behavior is not live-verified. Contract implementation **COMPLETE**, deployed confirmation **BLOCKED**. |
| 6 | Evaluation results for both models — **mandatory for comparison; XGBoost-only display can work separately** | Model page/agent can display individual available metrics, but direct comparison requires actual metrics for both models. | `evaluation_results.json` or equivalent API payload for XGBoost and LSTM on the same split; include all supported metric rows, missing-value status, confusion matrices as applicable, and artifact/model identity. | Docs report current training only implements XGBoost. Both-model comparison **BLOCKED**; XGBoost evaluation awaits artifacts. |
| 7 | Evaluation provenance — **mandatory for direct comparison** | Prevents comparing incomparable splits, protocols, populations, thresholds, or artifact versions. | For each model: split name and applicant alignment/IDs or shared split identifier, protocol/version, sample count/class counts, threshold, metric definitions, generation time, and model/preprocessor hashes. | Provenance validation is implemented; no real paired model provenance was available. Direct comparison **BLOCKED**. |
| 8 | SHAP or other real prediction explanations — **mandatory for explanation feature; optional for basic score retrieval** | The explanation page/tool must distinguish real SHAP from illustrative mock factors and avoid inventing contributions. | Confirm the XGBoost API returns the documented `tree_shap` fields (`output_space`, contributions, method, feature names/effects) and provide one sanitized prediction/explanation response or explain a supported alternate method. | Schema/docs and wrappers exist; no live explanation response verified. Real explanation feature **PARTIALLY COMPLETE**. |
| 9 | Pipeline ingestion and processing status — **conditional for snapshot dashboard; mandatory if demonstrating live pipeline freshness** | Clarifies whether current records are fresh/processed and whether expected source tables are ready; current mock data has no event/pipeline-status integration. | State pipeline stage/status, latest completed ingestion/processing time, row counts, data version, failures/retries, and how the API exposes readiness; if pipeline is not part of the demo, explicitly confirm snapshot-only scope. | No Person 2 pipeline status integration. Snapshot dashboard can operate without it; fresh/live pipeline claims **BLOCKED**. |
| 10 | Complete local run instructions — **mandatory for reproducible joint handoff** | Both teammates need repeatable commands and environment settings to reproduce API, dashboard, and agent results. | Confirm/update `docs/development_setup.md` with Python/dependency setup, dataset/artifact placement, API command, dashboard command, Ollama/model command or provider settings, environment variables, ports, and test commands. | Setup doc exists, including API and mock dashboard commands. Artifact placement and complete actual backend readiness still need confirmation; reproducibility **PARTIALLY COMPLETE**. |

## Joint integration roadmap

### Priority 0 — Agree the runnable integration bundle

- [ ] Person 1 confirms which dataset and model-artifact versions are the handoff baseline.
- [ ] Provide the deliverables in items 1–3 and 5 above, or explicitly mark any unavailable capability as out of scope.
- [ ] Confirm local paths, environment variables, ports, and sanitized test application IDs. Do not share credentials or real applicant data in chat/logs.

### Priority 1 — Verify the real API and application records

- [ ] Start the API using the agreed environment and verify `GET /health` reports data/model readiness accurately.
- [ ] Exercise application search/pagination and `GET /applications/{id}` with a permitted test ID; verify field allowlisting and that `TARGET` is not returned to applicant browsing.
- [ ] Exercise unknown, malformed, duplicate (if fixture exists), and service-unavailable responses. Confirm the dashboard stays in API mode and reports limitations without mock fallback.

### Priority 2 — Verify XGBoost prediction and explanation

- [ ] Call the prediction endpoint without explanation and compare its status, probability, risk band, model/version, and timestamp to the schema.
- [ ] Call with explanation enabled; validate SHAP method, output space, contribution values, and response identity against the same application and score.
- [ ] Test unknown application, missing data, missing/incompatible model, invalid output, and unavailable explanation cases. Confirm null probabilities/contributions remain unavailable rather than becoming zero.

### Priority 3 — Verify evaluation and comparison

- [ ] Retrieve validation/test evaluation payloads and check missing metric values, model identities, sample counts, threshold, split, protocol, and artifact hashes.
- [ ] If both model evaluations exist, confirm same split/protocol and applicant population before expecting a direct comparison chart or ranking.
- [ ] If LSTM remains unimplemented, verify that dashboard and agent clearly label it unavailable and avoid a head-to-head claim.

### Priority 4 — Validate the agent against real data

- [ ] Configure the agreed provider (Ollama or otherwise) and run the live opt-in integration tests.
- [ ] Ask the six tool-backed questions: dashboard metrics, application lookup, prediction, explanation, model comparison, and report.
- [ ] Verify selected tools, sourced values, privacy minimization, prompt-injection handling, missing-data behavior, and report coverage against actual API results.
- [ ] Keep tests with fake providers separate in the results from real API/provider runs.

### Priority 5 — Final dashboard demonstration and sign-off

- [ ] Run the dashboard in `api` mode, show API health, search/pagination, details, prediction/explanation, and model provenance.
- [ ] Demonstrate API outage behavior, then explicitly switch to mock mode and show synthetic labels; switch back to API to confirm mode changes are deliberate.
- [ ] Generate a report and explain unavailable sections and period coverage.
- [ ] Update this handoff with exact commands, artifact versions, live results, and any remaining blockers before release/merge.

## Test evidence and remaining blockers

The committed Group D validation report records:

```text
Focused dashboard, agent, and report tests: 185 passed
Full suite: 239 passed, 55 skipped
```

The full-suite skips are due to absent application/model artifacts and live Ollama tests requiring explicit opt-in. Streamlit UI checks use `AppTest`. API client contract tests use mock HTTP transport. The live `GET /health` check failed to connect; it is not counted as an integration pass.

Commands used (PowerShell, from the repository root):

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location).Path 'src')
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp .pytest-group-d-focused `
  tests\unit\test_application_explorer.py `
  tests\unit\test_mock_provider.py `
  tests\unit\test_dashboard_data_access.py `
  tests\unit\test_dashboard_overview.py `
  tests\unit\test_model_comparison.py `
  tests\unit\test_agent.py `
  tests\unit\test_report_generator.py -q

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp .pytest-group-d-full -q
```

Remaining blockers are the live API and data/model bundle, real artifact-backed XGBoost prediction/evaluation/SHAP verification, and the absent LSTM training/sequence path if LSTM comparison is expected. The actual browser/server dashboard demonstration and live LLM behavior are also still to be performed after the runtime dependencies are ready.

## Recommended next steps

1. Person 1 supplies or confirms the mandatory API, dataset, XGBoost artifact, prediction-response, and evaluation-provenance deliverables.
2. Person 1 explicitly scopes LSTM as unavailable for this integration or provides the missing LSTM sequence/model/API implementation.
3. Both teammates complete the prioritized integration roadmap and keep live checks distinct from mocked tests.
