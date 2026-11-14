# Person 2 — Group D End-to-End Validation & Reliability

**Branch:** `feature/person2-e2e-validation`  
**Validation date:** 2026-11-14  
**Validation scope:** Streamlit dashboard, LangGraph agent, dashboard/API client contracts, tests, and local integration readiness.

## 1. Executive summary

The dashboard and agent code paths were reviewed and exercised with the project's unit tests and Streamlit's in-process `AppTest`. The dashboard entrypoint rendered successfully in mock mode and in API mode with the expected disconnected status. Provider selection, API-default behavior, application search and pagination, details, prediction status handling, evaluation provenance, report generation, agent routing, and privacy boundaries have unit-level coverage.

The full suite completed with **239 passed and 55 skipped**. The skips are expected: the real application CSV and `models/local_v2` artifacts are absent, and live Ollama tests are opt-in. A direct request to `http://127.0.0.1:8000/health` could not connect; no backend was listening on port 8000. Therefore, no live backend or live LLM integration is claimed.

## 2. Dashboard validation results

| Feature | Status | Evidence / limitation |
|---|---|---|
| Dashboard entrypoint and default Overview render | **COMPLETE** | `AppTest.from_file` ran `src/dashboard/app.py` without uncaught exceptions. The check was headless, not a browser/server launch. |
| Real API is the default provider | **COMPLETE** | Unit coverage verifies the default mode is `api` / `fastapi`. |
| Explicit mock mode and switching between providers | **COMPLETE** | Streamlit selector tests verify `api` ↔ `mock`; mock mode labels synthetic output. API errors do not dispatch to mock mode. |
| API failure messaging and no silent mock fallback | **COMPLETE** | Page tests verify unavailable API results render explicit status and no sample applicants, metrics, or evaluation scores. |
| Application ID search and details | **COMPLETE** | Added an `AppTest` workflow that submits an ID search and loads the matching details. Existing tests cover retaining a non-first selection and hiding `TARGET`. |
| Application pagination | **COMPLETE** | Added an `AppTest` workflow that advances from offset 0 to 25 and verifies the displayed final-page rows and count. |
| Prediction status display | **COMPLETE** | Tests cover available, no prediction, model unavailable, unavailable, not found, and invalid responses without displaying a fake zero score. |
| Model comparison and evaluation provenance | **COMPLETE** | Tests verify direct comparison is withheld without compatible provenance, missing XGBoost/LSTM metrics remain unavailable, and conflicting LSTM rows are rejected. |
| Report generation and coverage limitations | **PARTIALLY COMPLETE** | Unit tests cover supported periods, partial/unavailable source sections, missing metrics, timestamp/period distinction, and agent routing. No live data source was available for an end-to-end report. |

## 3. Agent validation results

The existing six-tool allowlist and LangGraph execution path were preserved. Existing unit tests exercise routing and execution for dashboard metrics, application lookup, prediction, explanation, model comparison, and report generation. Tests cover invalid IDs, missing applications and predictions, unavailable models, tool exceptions, incomplete results, and failure-safe responses. Applicant-specific results continue to use local response rendering and are not sent to the response LLM.

| Agent feature | Status | Evidence / limitation |
|---|---|---|
| Six-tool allowlist and routing | **COMPLETE** | Unit tests verify all six tools remain registered and selected tools execute through the allowlist. |
| Invalid IDs, missing records/predictions, unavailable model, and tool failures | **COMPLETE** | Existing graph and tool tests verify explicit failure statuses and non-fabricated responses. |
| No invented predictions, metrics, or explanations | **COMPLETE** | Tests verify unavailable values are not replaced by zeroes or generated scores/SHAP values. |
| Mock versus real source labeling | **COMPLETE** | Mock provider and agent tests verify synthetic labels and the absence of mock evaluation metrics. |
| Prompt injection and retrieved-data trust boundary | **COMPLETE** | Existing prompt protections plus tests verify malicious user instructions cannot bypass ID redaction and malicious retrieved text is identified to the response model as untrusted evidence. Model adherence in a live provider remains unverified. |
| Live Ollama / live LLM tool selection | **BLOCKED** | Ollama integration tests require `RUN_OLLAMA_INTEGRATION=1`; no live Ollama call was run for this validation. |

## 4. API integration results

The API client and dashboard adapters use the existing contracts for health, application paging/detail, prediction, model metadata, evaluation, and prediction summaries. Unit tests exercise these contracts through `httpx.MockTransport`; those results are contract-level tests, not evidence of a running backend.

| Check | Status | Result |
|---|---|---|
| Live `GET /health` at `127.0.0.1:8000` | **BLOCKED** | Connection failed; no listener was present on port 8000. |
| Live application search/details | **BLOCKED** | No live API/data source was available. Contract behavior is covered by mocked transport tests. |
| Live prediction/evaluation endpoints | **BLOCKED** | Dataset and model artifacts are missing, and the backend was not running. Relevant local API integration tests were skipped. |
| API error/status mapping | **COMPLETE** | Unit tests cover API errors and unavailable/invalid outcomes without switching to mock data. |

Local environment checks found:

- `data/raw/application_train.csv`: absent.
- `models/local_v2`: absent.
- Port 8000 API listener: absent.
- Port 8501 Streamlit listener: absent at the time checked.

## 5. Security validation

- Application IDs are parsed and validated before data access; malformed and non-positive IDs are rejected.
- The application table omits `TARGET`; application details are limited to the configured field allowlist.
- API failures do not silently select synthetic mock data.
- Mock mode explicitly labels records, predictions, and explanations as synthetic; it supplies no trained-model evaluation metrics.
- Prediction and explanation outputs are status-checked and minimized; applicant-specific content is rendered locally outside the LLM response context.
- Prompt rules treat user instructions and retrieved fields as untrusted; tests verify ID redaction and the tool allowlist boundary.
- No credentials or secrets were read, printed, or added to the report.

## 6. Tests executed and results

Focused dashboard, agent, and report tests:

```text
185 passed in 11.93s
```

Full suite:

```text
239 passed, 55 skipped in 21.14s
```

The 55 skips are due to absent model artifacts/application data and the opt-in live Ollama integration tests. The added UI workflow tests use Streamlit `AppTest`; no real browser was used. The AppTest startup check in API mode rendered the Overview with visible API-disconnected/unavailable states and no uncaught exceptions. The mock-mode entrypoint also rendered without uncaught exceptions.

## 7. Bugs discovered and fixes

No production-code defect was confirmed during this validation. The audit identified missing direct UI coverage for application search/pagination and the dashboard entrypoint. Added tests for those workflows and verified them in the focused and full test runs. No production feature was rebuilt or refactored.

## 8. Remaining limitations

- Live backend integration is unverified because the backend is not running and the application CSV is missing.
- Real inference, SHAP, saved evaluation metrics, and artifact-backed report content are unverified because `models/local_v2` is absent.
- Live Ollama behavior is unverified; its tests were skipped unless explicitly enabled.
- The dashboard launch check was headless `AppTest`, not a separate Streamlit server/browser session.
- The tests validate that prompt defenses are supplied to the model; they cannot establish that every live model will always obey them.

## 9. Updated Person 2 task completion table

| Person 2 work area | Status | Notes |
|---|---|---|
| Dashboard UI entrypoint and core workflows | **COMPLETE** | Headless startup, provider switching, search, pagination, details, and prediction status handling verified. |
| Dashboard behavior on API failure | **COMPLETE** | Failures are visible and do not trigger a synthetic fallback. |
| Model evaluation display and provenance checks | **COMPLETE** | Missing metrics and invalid comparison provenance are reported. |
| Report generation behavior | **PARTIALLY COMPLETE** | Unit coverage is present; live source report validation awaits backend/data. |
| Agent tools, routing, and failure handling | **COMPLETE** | Six allowlisted tools and error cases covered by unit tests. |
| Agent privacy and prompt-injection safeguards | **COMPLETE** | Existing protections preserved and user/retrieved injection tests added in the Group C work. |
| Live API and artifact-backed end-to-end validation | **BLOCKED** | Requires the API service, application dataset, and model artifacts. |
| Live Ollama integration validation | **BLOCKED** | Requires a running local Ollama server and opt-in integration setting. |

## 10. Recommended next actions

1. Provide `data/raw/application_train.csv` and the expected `models/local_v2` artifacts through the established Person 1 handoff.
2. Start the FastAPI service and rerun the skipped API integration tests against the actual local application.
3. Start Ollama with the configured model and run the opt-in Ollama tests.
4. Perform a browser-level review of the dashboard after the backend is available; the current UI tests are headless.
5. Review this report and the test changes before any commit or push.
