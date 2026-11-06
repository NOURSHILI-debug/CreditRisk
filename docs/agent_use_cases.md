# LLM Agent Use Cases and Proposed Interfaces

## 1. Objective and scope

The agent will let authorized project users ask questions about application
records, available predictions, dashboard summaries, model evaluations, and
reports. It will retrieve evidence through validated agent tools backed by the
dashboard data-access layer, then explain only what those results support.

The agent is an information and reporting interface, not a credit decision
system. It must not approve or reject an application, fabricate application
fields, predictions, metrics, or explanations, or make unsupported financial,
causal, or historical claims.

This document defines the approved initial agent-facing tools. The five
data tools are implemented as deterministic wrappers in `src/agent/tools.py`.
Report generation remains future work. No chat interface is implemented.

## 2. Users and supported questions

Initial users are project analysts, data scientists, and engineering team
members inspecting the platform's available data and model outputs. The agent
supports factual questions answerable from retrieved project data, for example:

- "What application fields are available for application 1001?"
- "Is a prediction available for application 1001?"
- "How many applications have a valid target label?"
- "Which evaluation metrics are available for XGBoost and LSTM?"
- "What explanation did the prediction provider return?"
- "Summarize the available results for this reporting period."

The agent must identify synthetic data as mock/demo data and distinguish an
absent result from a zero value or a negative finding.

## 3. Current implementation facts

The dashboard data-access contract is implemented in
[`src/dashboard/data_access.py`](../src/dashboard/data_access.py):

| Existing function or type | Actual behavior |
|---|---|
| `get_provider_name() -> str`, `is_mock_mode() -> bool` | The current data-access layer reports `fastapi` and `False`, respectively. The dashboard and agent share the configured `APIClient`; no provider-selection setting is implemented in this layer. |
| `get_applications() -> pandas.DataFrame` | Returns a copy of the configured provider's records. An empty DataFrame is valid. An unavailable source raises `DataSourceUnavailableError`; a non-DataFrame response raises `MalformedProviderResponseError`. |
| `get_application_by_id(application_id, applications=None) -> ApplicationLookupResult` | Accepts a positive integer or a trimmed digit-only string; booleans, floats, zero/negative IDs, and other strings are invalid. Result status is `found`, `invalid_id`, `not_found`, `unavailable`, or `duplicate`. The `application` is a copied Series only for `found`. An empty source or missing `SK_ID_CURR` returns `unavailable`. |
| `get_overview_metrics(applications=None) -> dict[str, Any]` | Returns `total_applications`, `target_0`, `target_1`, `difficulty_rate`, `unlabeled`, and `risk_distribution`. Target counts/rate use only valid `TARGET` values 0 and 1. Missing or wholly unusable target data produces `None` for target counts/rate; a missing `RISK_CATEGORY` produces an empty distribution. If it loads the source itself, provider errors propagate. |
| `get_valid_target_values(applications) -> Series | None` | Returns only recognized binary target values, or `None` if `TARGET` is missing or has no usable values. |
| `get_credit_risk_prediction(application_id) -> CreditRiskPrediction` | Validates the ID and calls the shared API client for fresh inference. It validates finite probabilities in `[0, 1]`, returns available model metadata, and distinguishes HTTP 404 (`not_found`) from HTTP 200 (`no_prediction`). The probability-only agent tool disables SHAP; `explain_prediction` explicitly requests it. |
| `get_model_evaluation_results() -> DataFrame | None` | Returns copied model/metric/value rows or `None`. The mock provider returns `None`. Unavailable sources and malformed provider types raise data-access errors. |
| `normalize_model_evaluation_results(results)` | Validates the existing `model`, `metric`, `value` contract and returns `(values, statuses, issues)`. Supported models are XGBoost and LSTM; supported metrics are ROC-AUC, PR-AUC, Precision, Recall, and F1-score. Valid scores are finite values from 0 to 1. Missing, invalid, or duplicate rows remain unavailable and are reported in statuses/issues. |
| `DataSourceUnavailableError`, `MalformedProviderResponseError` | Explicit provider/source errors. Pages handle these errors and render an unavailable state rather than inventing results. |

The API client uses the configured FastAPI endpoint for application data and
fresh inference. Saved validation/test prediction CSVs are loaded separately
and used for aggregate summaries; they are not queried by application ID.

The Overview page calculates category, target-by-category, and numeric
distributions from the loaded DataFrame for presentation; they are not all
returned by `get_overview_metrics`. Model-comparison result normalization is
already shared through `normalize_model_evaluation_results`. Application
lookup and prediction display already use the data-access layer.

The single-agent orchestration is implemented in `src/agent/graph.py` with
typed state in `src/agent/state.py` and routing/grounding instructions in
`src/agent/prompts.py`. `src/agent/report_generator.py` remains an empty
placeholder. `src/dashboard/components/prediction_view.py` does not currently
define a prediction interface.

The four implemented wrappers in `src/agent/tools.py` return JSON-compatible
dictionaries and use `source_mode` (`mock` or `configured`) to identify
provenance. Their detailed statuses and output shapes are:

| Implemented tool | Actual output and behavior |
|---|---|
| `get_application_details(application_id: int | str) -> dict[str, Any]` | Returns `status`, `application_id`, `fields`, `null_fields`, and `source_mode`, plus `message` where applicable. It delegates to `get_application_by_id`; null fields are listed separately. It omits the dashboard-derived `RISK_PROBABILITY` and `RISK_CATEGORY` fields so the synthetic risk score is not confused with an application fact or prediction. |
| `get_dashboard_metrics() -> dict[str, Any]` | Returns `status`, `metrics`, and `source_mode`, plus `message` when unavailable or invalid. It loads one application snapshot and passes it to `get_overview_metrics`. An empty dataset has unavailable status while preserving the actual zero total and unavailable target metrics. |
| `get_model_comparison() -> dict[str, Any]` | Retrieves saved API evaluation rows and returns legacy `models`/`statuses`/`issues` plus per-model `evaluations`, `evaluation_metadata`, `comparison_available`, and `comparison_note`. Only actual metric values are returned; no missing score is filled. |
| `get_prediction(application_id: int | str) -> dict[str, Any]` | Validates the ID, requests fresh inference through `get_credit_risk_prediction` with explanations disabled, and returns `status`, `application_id`, `probability`, `risk_level`, model name/version, prediction timestamp, message, and `source_mode`. Optional prediction values remain null; SHAP output is not included. |
| `explain_prediction(application_id: int | str) -> dict[str, Any]` | Validates the ID and requests fresh inference with SHAP enabled through the same API/data-access path. Returns prediction/model metadata and validated SHAP summary fields and contributions, while omitting raw applicant values and transformed feature values. |

Expected `DataSourceUnavailableError` results are returned with an explicit
unavailable status; malformed provider or non-JSON-compatible payloads are
returned with invalid status. Unexpected exceptions are not silently caught.

## 4. Use cases

| Use case | Example questions | Required input | Proposed tool | Expected response | Possible failure or unavailable state |
|---|---|---|---|---|---|
| Application details | "Show the available details for application 1001." "Which fields are missing?" | One positive application ID | `get_application_details(application_id)` | Implemented: JSON-compatible lookup status, non-null application fields, and source-null field names. Omits dashboard-derived risk score/category fields. | Invalid ID, not found, duplicate ID, unavailable source/ID column, malformed provider response, or no fields to display. |
| Credit-risk prediction | "Is a prediction available for application 1001?" "What probability and model version were returned?" | One positive application ID | `get_prediction(application_id)` | Runs fresh inference via `GET /applications/{application_id}/prediction` with `include_explanation=false`. Returns a JSON-compatible result with status, application ID, probability, risk level, available model name/version and timestamp, message, and source mode. Missing optional values remain null. | Invalid ID, application not found, valid application with no prediction, model unavailable, API connection failure, prediction execution failure, or malformed response. |
| Dashboard metrics | "How many applications have valid targets?" "What are the target counts and available risk categories?" | No required input; optional scope only if the source supports it | `get_dashboard_metrics()` | Implemented: shared metric values and definitions, with unavailable target values remaining null. Empty data returns unavailable status and the actual zero total. | Provider unavailable, empty dataset, missing/invalid `TARGET`, absent risk field, malformed response, or mock-data source. |
| Model comparison | "Compare the available XGBoost and LSTM metrics." "Which metrics are missing?" | No required input; optional period only if evaluation outputs support filtering | `get_model_comparison()` | Retrieves saved evaluation rows via the API. Preserves actual values in legacy model/metric mapping and per-model records; marks `comparison_available` true only when both models have complete metrics and shared split/population, target, metric-definition, threshold, and split-protocol metadata is present. | No evaluation results, one model absent, partial/invalid/duplicate metric rows, unavailable provider, malformed response, or incomplete/inconsistent evaluation protocol. |
| Prediction explanation | "What explains the prediction for application 1001?" | One positive application ID | `explain_prediction(application_id)` | Runs fresh inference via `GET /applications/{application_id}/prediction` with `include_explanation=true`. Returns the prediction probability and risk/model metadata, SHAP method/output-space metadata, additive summary values, and the provider-ordered feature contributions. Only feature name, SHAP value, and verified effect are sent for each feature. | Invalid ID, application not found, valid application without a prediction, model unavailable, explanation unavailable, malformed response, API connection failure, or explanation execution failure. |
| Report generation | "Summarize the available model results for the selected period." "Create a current portfolio summary." | A requested period/scope string; it must be supported by the underlying source to filter data | `generate_report(period)` | Structured summary of retrieved application metrics and/or model results, with source, mock/real label, coverage, and unavailable sections identified. | Period not supported by source data, missing timestamps, unavailable inputs, partial results, or no evidence for requested trends. |

`TARGET=1` means repayment difficulty; `TARGET=0` means no defined repayment
difficulty. Neither label is an approval decision. A target distribution must
show its labeled denominator and not treat missing or invalid target values as
either class.

## 5. Proposed agent-facing tool interfaces

These are proposed Python-level wrappers. They should return JSON-compatible
objects for agent consumption rather than expose pandas objects or Streamlit
elements. The structures below define the intended wrapper contract; they are
not existing return types. Existing APIs and their return values are listed in
Section 3. Agent wrappers must reuse those APIs rather than load data or
reimplement calculations.

All tools should return only retrieved values plus explicit status/provenance
metadata. They should not turn exceptions into successful-looking empty or
zero results. At minimum, a wrapper should distinguish a successful result,
an unavailable result, invalid input, and malformed source data, either through
a documented result status or a raised, handled tool error.

### `get_application_details(application_id)`

- **Purpose:** Retrieve one application and summarize its available fields.
- **Input:** `application_id: int | str`. Apply the data-access validation:
  positive integer or trimmed digit-only string; reject booleans, floats, and
  non-positive values.
- **Source/mapping:** `get_application_by_id(application_id)`, which calls
  `get_applications()` when no DataFrame is supplied. Do not query files or
  storage from the agent.
- **Proposed output:** `{status, application_id, fields, null_fields, message}`.
  `fields` contains only source-provided, non-null values. `null_fields` lists
  source-provided fields whose value is null. Do not claim that a field absent
  from the row is null unless a source schema identifies it as expected.
- **Validation/errors:** Preserve lookup statuses `invalid_id`, `not_found`,
  `duplicate`, and `unavailable`; catch and report
  `DataSourceUnavailableError` / `MalformedProviderResponseError` as tool
  errors. Never select one duplicate record arbitrarily.
- **Status:** Implemented in `src/agent/tools.py` as a thin adapter using the
  existing lookup function. JSON-safe conversion and missing/null presentation
  are implemented. It excludes the dashboard-derived risk score/category
  fields from application details.

### `get_prediction(application_id)`

- **Purpose:** Request fresh inference for one application through the configured
  FastAPI service. This does not retrieve a previously saved per-application
  prediction record.
- **Input:** A positive integer or trimmed digit-only string. Booleans, floats,
  zero/negative IDs, and other strings are rejected before the API is called.
- **Source/mapping:** `get_credit_risk_prediction(application_id,
  include_explanation=False)` uses the shared `APIClient` and
  `GET /applications/{application_id}/prediction`. The GET route calls
  `PredictionService` and runs the loaded predictor against the application
  row. Saved validation/test prediction CSVs are used separately for aggregate
  summaries.
- **Output:** A JSON-compatible object containing `status`, `application_id`,
  `probability`, `risk_level`, `model_name`, `model_version`,
  `prediction_timestamp`, `message`, and `source_mode`. Missing optional
  prediction fields remain null. The normal tool result excludes SHAP
  explanations and applicant feature values.
- **Status/errors:** `available` for a valid prediction; `invalid` for invalid
  input, HTTP 422 inference validation failures, or malformed responses;
  `not_found` for the prediction endpoint's HTTP 404; `no_prediction` for HTTP
  200 without a prediction; `model_unavailable` when model artifacts cannot be
  used; `unavailable` for API connection/service unavailability; and `error`
  for other HTTP 5xx prediction failures. Probability is never filled or
  substituted.
- **Privacy:** `TARGET` is not returned by the application repository for
  inference, and the normal prediction tool does not include explanation
  feature contributions or raw feature values.

### `get_dashboard_metrics()`

- **Purpose:** Return application counts, target distribution, and supported
  risk statistics.
- **Input:** None in the initial interface. A date/filter argument is not
  supported until the provider exposes a matching filterable dataset.
- **Source/mapping:** `get_applications()` followed by
  `get_overview_metrics(applications)` so the page snapshot is reused. Do not
  reimplement target counts or read data independently.
- **Proposed output:** The current metrics mapping:
  `{total_applications, target_0, target_1, difficulty_rate, unlabeled,
  risk_distribution}`. `difficulty_rate` is the share of valid labeled rows
  with `TARGET=1`; `risk_distribution` contains category counts. `None` means
  target statistics are unavailable; an empty risk distribution means no
  supported risk category was present. A wrapper may add a target share
  breakdown only when it is calculated from these retrieved counts and clearly
  identifies its denominator.
- **Validation/errors:** Empty data yields zero total and unavailable target
  statistics, not fabricated observations. Missing columns are handled as
  described in Section 3. Report data-source errors explicitly.
- **Status:** Implemented in `src/agent/tools.py` for whole-source metrics.
  Time periods, streaming freshness, and metrics beyond the current returned
  mapping require additional provider contracts.

### `get_model_comparison()`

- **Purpose:** Return actual available evaluation results for XGBoost and LSTM.
- **Input:** None initially. Period filtering is not supported by the current
  evaluation retrieval function.
- **Source/mapping:** `get_model_evaluation_results()` followed by
  `normalize_model_evaluation_results(results)`. The retrieval uses
  `APIClient.evaluation()` and `GET /model/evaluation`; it never opens model
  artifacts inside the agent. The API reads the saved XGBoost
  `evaluation_results.json` artifact and preserves its protocol metadata.
- **Output:** Retains `models` (model-to-metric mapping), per-model `statuses`,
  `issues`, overall `status`, and `source_mode`. Also returns `evaluations`
  (one record each for XGBoost and LSTM, with per-model evaluation status,
  split, actual metrics or null, and model version when supplied),
  `evaluation_metadata` (split, source, timestamp, threshold, sample and class
  counts, target convention, metric definitions, and split protocol),
  `comparison_available`, and `comparison_note`. Metrics are actual finite
  scores from 0 to 1; missing metrics remain null.
- **Availability today:** The backend currently exposes the saved XGBoost
  evaluation results only and reports `lstm_available=false`. Although an
  LSTM configuration and test file are present, `src/training/train_lstm.py`
  is empty and no LSTM model implementation, training pipeline, evaluation
  artifact, or inference integration was found under `src/` or `models/`.
  The agent reports LSTM evaluation as unavailable and does not substitute
  XGBoost values.
- **Comparison validity:** A direct comparison is permitted only when both
  models have complete metric sets and the shared response includes a dataset
  split, sample/class population, `TARGET=1` positive-class semantics,
  definitions for each metric, a valid threshold, and split-protocol metadata.
  Otherwise metrics are reported separately and `comparison_available` is
  false. No overall winner is inferred; any discussion must remain
  metric-specific and grounded in returned values.
- **Validation/errors:** `None` or an empty result means no evaluation
  results. Missing required columns, unknown identifiers, duplicate rows, and
  invalid scores remain visible as invalid/partial status and issues. API
  connection/artifact failures are unavailable; malformed payloads are
  invalid; unexpected failures return error without internal exception details.

### `explain_prediction(application_id)`

- **Purpose:** Request fresh inference and its genuine TreeSHAP explanation for
  an application's prediction.
- **Input:** `application_id: int | str`, validated as above.
- **Source/mapping:** Calls
  `get_credit_risk_prediction(application_id, include_explanation=True)`,
  which uses the centralized dashboard API client and
  `GET /applications/{application_id}/prediction?include_explanation=true`.
  The GET endpoint runs new inference for that application and asks the
  startup-loaded `PredictionService` explainer for SHAP output. It does not
  retrieve a previously saved per-application prediction. Saved validation
  and test prediction records are used separately for aggregate evaluation.
- **Output:** A JSON-compatible dictionary containing `status`,
  `application_id`, `prediction_probability`, `risk_level`, `model_name`,
  `model_version`, `explanation_method`, `output_space`, `base_value`,
  `base_probability`, `prediction_margin`, `contribution_sum`, `top_features`,
  `message`, and `source_mode`. Each top feature contains only `feature`,
  `shap_value`, and `effect`. Optional numeric and model metadata fields remain
  null when the provider does not supply them; the tool does not pad or invent
  contributions.
- **Validation/errors:** `available` is returned only for a valid prediction
  and a supported TreeSHAP/log-odds payload. `not_found`, `no_prediction`,
  `model_unavailable`, and `unavailable` are preserved from the data-access
  result; missing/failed SHAP output returns `explanation_unavailable`.
  Invalid IDs and malformed responses return `invalid`; unexpected execution
  errors return `error`. Error messages do not include internal exception
  details.
- **Interpretation and privacy:** SHAP contributions are additive in log-odds
  space for `P(TARGET=1)`: positive values increase the model score for
  repayment difficulty, negative values decrease it, and zero is neutral.
  These are model associations, not causal explanations. Raw feature values,
  transformed feature values, `TARGET`, and applicant identifiers other than
  the requested ID are excluded from the agent-facing explanation.

### `generate_report(period)`

- **Purpose:** Produce a structured narrative from retrieved, supported
  application metrics and/or model evaluation results.
- **Input:** `period: str`, identifying the requested scope. The tool must
  validate that the source can actually select that period; a label supplied
  by the user is not proof that data was filtered.
- **Source/mapping:** Compose only outputs from `get_dashboard_metrics()` and
  `get_model_comparison()` (and other validated tools added later). No report
  generator currently exists.
- **Proposed output:** `{requested_period, data_coverage, source_mode,
  application_metrics, model_comparison, limitations, summary}`. Each section
  may be explicitly unavailable; `summary` must be grounded in the returned
  sections.
- **Validation/errors:** If the source has no relevant timestamps or period
  filter, mark the period unsupported and offer an all-available-data summary
  only if the user accepts that scope. Do not claim trends, changes over time,
  or period comparisons without date-scoped records or evaluation artifacts.
  Preserve partial results and their missing sections; do not fill gaps.
- **Status:** Not implementable as a period report with current APIs. Whole
  dataset summaries can be composed after the report wrapper is implemented;
  period filtering and trend claims depend on timestamped, filterable source
  data and evaluation coverage from Person 1.

## 6. Response, safety, and unavailable-data rules

The agent must apply these rules consistently, regardless of phrasing:

| Condition | Required behavior |
|---|---|
| Invalid application ID | State that the ID is invalid and provide the accepted positive-integer format. Do not call it not found. |
| Application not found | State that no matching record was returned. Do not infer why it is absent. |
| Duplicate application ID | State that the ID is ambiguous and do not select a record. |
| Missing application fields | Show available fields only. Distinguish source nulls from fields not present; never guess or fill values. |
| Prediction unavailable | Say no prediction provider/result is available. Do not use `RISK_PROBABILITY`, `TARGET`, or an LLM estimate as a substitute. |
| Model unavailable | Identify the model status as unavailable; do not imply it was evaluated or that it performed poorly. |
| Missing evaluation metrics | Mark the specific metric unavailable. For partial results, state which metrics/models are present and absent. Never interpolate or supply example scores as real results. |
| Explanation unavailable | State that no supported explanation was returned. Do not generate feature-importance claims from the model score or record fields. |
| Data provider unavailable | Report the configured source/provider failure. Do not silently switch to mock or present empty data as a successful query. |
| Malformed provider response | Report a data/tool error and do not interpret malformed values as valid. |
| Partial results | Answer only the supported portion and list relevant limitations or unavailable fields. |
| Mock-data mode | Prominently label results synthetic/demo. Never describe mock records, risk categories, probabilities, predictions, or metrics as real applicants or real model output. |

Use "repayment difficulty" for `TARGET=1` and "no defined repayment
difficulty" for `TARGET=0`. Explain that target labels are historical outcomes,
not lending decisions. The agent must not make automatic loan approval or
rejection recommendations. It must not claim financial causality or
unsupported conclusions. Factual answers may use only retrieved project
information and clearly identified calculations derived from it.

## 7. Framework selection

**Recommendation: LangGraph for the initial agent workflow.** LangChain offers
useful model, prompt, and tool abstractions; LangGraph adds explicit state and
control over the sequence and branches of a tool-using workflow. The initial
requirements prioritize validated inputs, allowlisted tool execution, visible
unavailable/error paths, and grounded responses over open-ended autonomous
planning.

A single agent and a small explicit graph are sufficient:

1. Receive a user question and identify the requested operation.
2. Validate required arguments and reject unsupported scopes.
3. Select from the four allowlisted tools and execute only the necessary call.
4. Validate tool status/payloads, retaining provenance and unavailable fields.
5. Produce a concise response grounded in those outputs and enforce the
   response rules in Section 6.

This workflow is implemented in `src/agent/graph.py` as
`build_agent_graph(model=None)` and `run_agent(question, *, model=None)`.
`AgentState` in `src/agent/state.py` stores the current question, only the
current user/assistant messages, selected tool and arguments, current tool
result, response, and status/error. It does not retain multi-turn history.

The router uses a constrained structured response that can select only one of
the four tools, `general`, or `unsupported`. It cannot supply tool arguments.
Application IDs are extracted from the user's text and validated as positive
integers before execution. If an application-specific request omits an ID,
the workflow asks for clarification and does not call a tool. Invalid or
ambiguous IDs are rejected. Only the selected tool is executed; dynamic tool
names, arbitrary SQL, filesystem, shell, and API access are not exposed.

Unavailable, invalid, and partial tool results are communicated
deterministically without asking the LLM to fill gaps. Tool exceptions are
logged and returned as safe error states. Other tool results and general
platform questions are answered with a grounded response prompt; mock-backed
tool answers receive an explicit mock/demo label. The initial workflow uses a
fresh single-turn state and
doesn't persist conversation history.

No chat interface is implemented. No LangGraph checkpoint store or long-term
conversation memory is configured.

Multiple agents are unnecessary for the first version: these use cases share
one data-access boundary, have a small fixed tool set, and require consistent
validation and response policy. Multiple agents would add routing and
coordination complexity without a demonstrated independent responsibility.
LangChain may be used for compatible model/tool components if useful, but a
second orchestration framework is not a requirement.

### LLM configuration

The LangGraph package is the orchestrator and `langchain-openai` is the
initial chat-model adapter. Configuration is read at request time:

- `AGENT_LLM_PROVIDER`: defaults to `openai`; unsupported provider names raise
  `AgentConfigurationError`.
- `OPENAI_API_KEY`: required only when the default model is first requested;
  it is read from the environment, never embedded in code.
- `AGENT_LLM_MODEL`: optional model name; defaults to `gpt-4o-mini`.
- `OPENAI_BASE_URL`: optional OpenAI-compatible endpoint.

`build_agent_graph(model=...)` and `run_agent(..., model=...)` accept an
injected LangChain chat model, so tests and custom integrations do not need an
API key or external calls. A missing key without an injected model raises a
clear `AgentConfigurationError`; no model is initialized merely by importing
the module or building the graph.

## 8. Intended architecture

```text
Streamlit chat interface (future)
        |
        v
LangGraph single-agent workflow (implemented)
        |
        v
Validated, allowlisted agent tools
        |
        v
dashboard.data_access
        |
        v
Project data and ML interfaces
        |
        v
Retrieved results + provenance/status
        |
        v
Grounded agent response
```

Only the orchestration layer, prompts, typed state, and four tools are
implemented. The Streamlit chat interface remains future work. The agent must
not independently read datasets, model artifacts, or storage, and must not
duplicate dashboard retrieval or metric logic. Any new source should be
connected through a provider/data-access contract, then exposed through a
narrow validated agent tool.

## 9. Person 1 integration requirements

Before real-data use, Person 1 should agree and provide:

- **Application data:** Stable provider availability, identifier uniqueness,
  application field/schema documentation, null semantics, and provenance for
  each source. A stable real provider must implement the existing
  `DashboardDataProvider` contract or an agreed compatible adapter.
- **Predictions:** A callable inference boundary by application ID; an agreed
  response schema with probability meaning (`P(TARGET=1)`), model name/version,
  timestamp, explicit no-prediction/model-unavailable states, and malformed
  response behavior. Probability must be finite and in `[0, 1]`.
- **Model evaluation:** The current API reads the saved XGBoost
  `evaluation_results.json` artifact; it has no LSTM evaluation artifact or
  LSTM evaluation API output. Future model results require an evaluation
  artifact/adapter, stable model and metric identifiers, finite scores,
  evaluation dataset/split and timestamp, positive-class convention, and
  explicit absent/partial-result semantics. Provide LSTM metrics only when an
  evaluation was actually performed and its genuine results are available.
- **Explanations:** The current API supplies TreeSHAP output in log-odds
  space. Changes to that output convention or schema must be versioned and
  validated at the agent boundary; the agent must not infer contributions.
- **Periods and reports:** Timestamp fields, timezone/coverage semantics, and
  provider-side filtering or aggregation if reports are to describe a period or
  historical change. Current overview/evaluation functions accept no period.
- **Source provenance:** A way to identify mock/demo versus real source and,
  for real outputs, the relevant model/data version where available.

Person 2 owns the implemented wrappers and LangGraph workflow, and remains
responsible for their maintenance, a future chat UI, and report generation.
The agent should integrate only through agreed interfaces, not ML internals.

## 10. Explicit out of scope

- Chatbot UI, long-term conversation memory, and report-generation tools.
- Causal interpretation of SHAP contributions or use of explanations as
  lending recommendations.
- Training, modifying, or deploying XGBoost or LSTM.
- Generating predictions, model metrics, feature explanations, or application
  data with an LLM.
- Automatic credit approval/rejection or lending recommendations.
- AWS, database, Kafka, or external-service integration work.
- Period trends, causal analysis, or unsupported financial advice.
- Multi-agent orchestration, unrestricted query tools, or unnecessary
  autonomous actions.

## 11. Unresolved interfaces

The following decisions must be settled before the dependent tools are treated
as production-ready: the real provider adapter and provenance schema; model
version and timestamp guarantees; LSTM sequence/evaluation availability; and
whether application/evaluation data supports period filtering. The current
XGBoost/TreeSHAP and XGBoost evaluation contracts are implemented for the
configured API; LSTM evaluation remains unavailable until actual results and
protocol metadata are supplied.
