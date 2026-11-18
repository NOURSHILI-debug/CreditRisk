# Agent Evaluation Framework

This document describes the deterministic evaluation framework for the
LangGraph/LangChain AI agent in the Credit Risk Intelligence Platform. It lives
under `tests/agent_evaluation/` and is runnable with `pytest`.

## Purpose

Evaluate whether the agent:

1. Selects the correct tool for a user request (**tool selection**).
2. Passes valid arguments to its tools (**argument validation**).
3. Produces answers grounded in tool results without fabrication
   (**groundedness**).
4. Protects applicant privacy (**privacy**).
5. Resists prompt injection (**prompt injection**).
6. Handles missing data, invalid inputs, and tool failures safely
   (**failure handling**).

## Scope and limitations

- The framework uses **controlled fake tool outputs** and a **fake chat model**.
  It does not require Person 1's real dataset, XGBoost model artifacts, AWS
  credentials, SageMaker, Kafka, Redpanda, or a live Ollama/OpenAI provider.
- Mock data is clearly labeled and used only inside the isolated evaluation
  environment. It never replaces real project data in production code paths.
- The evaluation is deterministic: every check is a plain Python assertion on
  the agent state, final response, and LLM inputs. No LLM-based qualitative
  judgment is used or treated as authoritative.
- Person 1's modules (`src/ingestion/`, `src/storage/`, `src/training/`,
  `src/inference/`) were **not modified**.

## Directory structure

```
tests/agent_evaluation/
├── __init__.py
├── datasets/
│   ├── __init__.py
│   └── cases.json        # 30 structured evaluation cases
├── fixtures.py           # Fake chat model and call-tracking fake tools
├── runner.py             # Deterministic evaluation runner + report formatter
└── test_evaluation.py    # Pytest wrapper with metric assertions
```

## Evaluation dataset

`tests/agent_evaluation/datasets/cases.json` contains 30 cases across 6
categories:

| Category | Cases | Description |
|---|---|---|
| `tool_selection` | 6 | Correct tool routing, no-argument requests, general questions |
| `argument_validation` | 5 | Invalid IDs, missing IDs, unsupported report periods |
| `groundedness` | 5 | Actual probabilities/metrics reflected, no fabrication, no ranking |
| `privacy` | 4 | ID redaction, minimized rendering, no sensitive leakage |
| `prompt_injection` | 5 | User and retrieved-data injection, lending-refusal, allowlist enforcement |
| `failure_handling` | 5 | Tool exceptions, missing records, missing predictions, unavailable model |

Each case specifies the question, expected tool (or `null`), expected tool
arguments, a fake tool result, expected final status, and a list of
deterministic checks.

## Metrics

Each category reports:

| Metric | Definition |
|---|---|
| Tool-selection accuracy | Correct tool selected / applicable routing cases |
| Argument-validation pass rate | Correct validation behavior / argument cases |
| Groundedness pass rate | Responses supported by tool results / groundedness cases |
| Privacy pass rate | No prohibited information exposure / privacy cases |
| Injection-resistance pass rate | Malicious instructions safely handled / injection cases |
| Failure-handling pass rate | Cases handled safely / failure cases |

Each metric reports its numerator, denominator, and percentage. Privacy and
prompt-injection failures are treated as **critical** and fail the evaluation
even if other metrics pass.

## How to run

From the repository root, with the virtual environment activated:

```powershell
# Run the full evaluation suite
.venv\Scripts\python.exe -m pytest tests/agent_evaluation -v

# Run with the full test suite
.venv\Scripts\python.exe -m pytest -q

# Run only the agent evaluation tests
.venv\Scripts\python.exe -m pytest tests/agent_evaluation/test_evaluation.py -v
```

### Expected output

A successful run prints something like:

```
Agent Evaluation Report
============================================================

Total cases: 30
Passed: 30 / 30
Overall pass rate: 100.0%

Critical privacy/security failures:
  0

Results by category:
------------------------------------------------------------
  argument_validation:
    Passed: 5 / 5 (100.0%)
  failure_handling:
    Passed: 5 / 5 (100.0%)
  groundedness:
    Passed: 5 / 5 (100.0%)
  privacy:
    Passed: 4 / 4 (100.0%)
  prompt_injection:
    Passed: 5 / 5 (100.0%)
  tool_selection:
    Passed: 6 / 6 (100.0%)
  ...
```

## Interpreting results

- **Overall pass rate** is the aggregate across all 30 cases.
- **Per-category percentages** show where the agent is strong or weak.
- **Critical failures** (privacy or injection) fail the evaluation immediately,
  regardless of other category scores.
- **Failed cases** are listed with the specific check that failed, the expected
  value, and the actual detail.

## Adding new cases

Add a new entry to `tests/agent_evaluation/datasets/cases.json` following the
existing schema:

```json
{
  "id": "new-001",
  "category": "tool_selection",
  "description": "Human-readable description.",
  "question": "User question string.",
  "expected_tool": "get_prediction",
  "expected_arguments": {"application_id": 1001},
  "tool_result": { ... fake tool output ... },
  "llm_response": "Fake LLM response (used only if LLM is invoked).",
  "expected_status": "answered",
  "critical": false,
  "checks": [
    {"type": "tool_selected", "expected": "get_prediction", "description": "..."},
    {"type": "final_status", "expected": "answered", "description": "..."}
  ]
}
```

Supported check types: `tool_selected`, `tool_not_called`, `tool_was_called`,
`tool_arguments`, `final_status`, `response_contains`,
`response_not_contains`, `response_starts_with`, `llm_response_calls`,
`route_inputs_contain`, `route_inputs_not_contains`,
`response_inputs_contain`, `response_inputs_not_contains`.

Set `"critical": true` for privacy or injection cases where a failure must
fail the entire evaluation.

## Files created

- `tests/agent_evaluation/__init__.py`
- `tests/agent_evaluation/datasets/__init__.py`
- `tests/agent_evaluation/datasets/cases.json`
- `tests/agent_evaluation/fixtures.py`
- `tests/agent_evaluation/runner.py`
- `tests/agent_evaluation/test_evaluation.py`
- `docs/AGENT_EVALUATION.md`

## Files modified

None. No production code, test code, or configuration was modified.
