"""Deterministic evaluation runner for the LangGraph AI agent.

Loads evaluation cases from ``datasets/cases.json``, runs each case against the
real agent graph using controlled fake tool outputs and a fake chat model, and
applies deterministic assertions. No real LLM provider, model artifacts,
dataset, or cloud credentials are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent import graph
from agent_evaluation.fixtures import CallTrackingFakeTool, FakeChatModel

CASES_PATH = Path(__file__).resolve().parent / "datasets" / "cases.json"


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """Load evaluation cases from the JSON dataset."""
    cases_path = path or CASES_PATH
    with cases_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_check(
    state: dict[str, Any],
    check: dict[str, Any],
    fake_model: FakeChatModel,
    fake_tool: CallTrackingFakeTool | None,
) -> tuple[bool, str]:
    """Apply a single deterministic check to an agent run result.

    Returns ``(passed, detail)`` where ``detail`` explains the outcome.
    """
    check_type = check["type"]
    expected = check["expected"]

    if check_type == "tool_selected":
        actual = state.get("selected_tool")
        if expected is None:
            return (actual is None, f"selected_tool is {actual!r}, expected None")
        return (actual == expected, f"selected_tool is {actual!r}, expected {expected!r}")

    if check_type == "tool_not_called":
        called = fake_tool.called if fake_tool is not None else False
        return (not called, f"tool was called: {called}")

    if check_type == "tool_was_called":
        called = fake_tool.called if fake_tool is not None else False
        return (called, f"tool was called: {called}")

    if check_type == "tool_arguments":
        actual = state.get("tool_arguments")
        return (
            actual == expected,
            f"tool_arguments is {actual!r}, expected {expected!r}",
        )

    if check_type == "final_status":
        actual = state.get("status")
        return (
            actual == expected,
            f"status is {actual!r}, expected {expected!r}",
        )

    if check_type == "response_contains":
        response = state.get("final_response") or ""
        return (
            expected in response,
            f"'{expected}' not in final response",
        )

    if check_type == "response_not_contains":
        response = state.get("final_response") or ""
        return (
            expected not in response,
            f"'{expected}' unexpectedly present in final response",
        )

    if check_type == "response_starts_with":
        response = state.get("final_response") or ""
        return (
            response.startswith(expected),
            f"final response does not start with {expected!r}",
        )

    if check_type == "llm_response_calls":
        actual = fake_model.response_calls
        return (
            actual == expected,
            f"response LLM called {actual} time(s), expected {expected}",
        )

    if check_type == "route_inputs_contain":
        return (
            expected in str(fake_model.route_inputs),
            f"'{expected}' not in routing LLM inputs",
        )

    if check_type == "route_inputs_not_contains":
        return (
            expected not in str(fake_model.route_inputs),
            f"'{expected}' unexpectedly present in routing LLM inputs",
        )

    if check_type == "response_inputs_contain":
        return (
            expected in str(fake_model.response_inputs),
            f"'{expected}' not in response LLM inputs",
        )

    if check_type == "response_inputs_not_contains":
        return (
            expected not in str(fake_model.response_inputs),
            f"'{expected}' unexpectedly present in response LLM inputs",
        )

    return (False, f"unknown check type: {check_type}")


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run a single evaluation case and return the result.

    The real agent graph is executed with a fake chat model and, when the case
    expects a tool, a call-tracking fake tool temporarily replaces the real
    entry in ``graph._ALLOWED_TOOLS``. The original entry is always restored
    afterwards.
    """
    tool_name = case.get("expected_tool")
    fake_model = FakeChatModel(
        tool=tool_name or "general",
        response=case.get("llm_response", "Grounded response."),
        router_return=case.get("router_return"),
    )

    fake_tool: CallTrackingFakeTool | None = None
    original_tool: Any = None

    try:
        if tool_name is not None:
            original_tool = graph._ALLOWED_TOOLS.get(tool_name)
            expected_args = case.get("expected_arguments", {})
            tool_result = case.get("tool_result") or {}
            fake_tool = CallTrackingFakeTool(
                tool_result,
                expected_args,
                raise_exception=bool(case.get("raise_exception", False)),
            )
            graph._ALLOWED_TOOLS[tool_name] = fake_tool

        state = graph.run_agent(case["question"], model=fake_model)
    finally:
        if tool_name is not None:
            if original_tool is not None:
                graph._ALLOWED_TOOLS[tool_name] = original_tool
            else:
                graph._ALLOWED_TOOLS.pop(tool_name, None)

    check_results: list[dict[str, Any]] = []
    for check in case.get("checks", []):
        passed, detail = _apply_check(state, check, fake_model, fake_tool)
        check_results.append(
            {
                "type": check["type"],
                "expected": check["expected"],
                "passed": passed,
                "detail": detail,
                "description": check.get("description", ""),
            }
        )

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "description": case.get("description", ""),
        "critical": bool(case.get("critical", False)),
        "passed": all(cr["passed"] for cr in check_results),
        "check_results": check_results,
        "final_status": state.get("status"),
        "final_response": state.get("final_response"),
        "tool_called": fake_tool.called if fake_tool is not None else None,
        "llm_response_calls": fake_model.response_calls,
    }


def run_evaluation(cases: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Run all evaluation cases and return per-case results."""
    if cases is None:
        cases = load_cases()
    return [run_case(case) for case in cases]


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute evaluation metrics from case results.

    Returns a dict with per-category numerators/denominators/percentages, an
    overall metric, and a list of critical failures.
    """
    by_category: dict[str, dict[str, Any]] = {}
    for result in results:
        cat = result["category"]
        entry = by_category.setdefault(
            cat, {"passed": 0, "total": 0, "cases": [], "critical_failures": []}
        )
        entry["total"] += 1
        if result["passed"]:
            entry["passed"] += 1
        entry["cases"].append(result)
        if result["critical"] and not result["passed"]:
            entry["critical_failures"].append(result)

    categories: dict[str, Any] = {}
    overall_passed = 0
    overall_total = 0
    all_critical_failures: list[dict[str, Any]] = []

    for cat, data in sorted(by_category.items()):
        rate = data["passed"] / data["total"] if data["total"] else 0.0
        categories[cat] = {
            "passed": data["passed"],
            "total": data["total"],
            "rate": round(rate, 4),
            "percentage": round(rate * 100, 2),
            "critical_failures": len(data["critical_failures"]),
        }
        overall_passed += data["passed"]
        overall_total += data["total"]
        all_critical_failures.extend(data["critical_failures"])

    overall_rate = overall_passed / overall_total if overall_total else 0.0

    return {
        "categories": categories,
        "overall": {
            "passed": overall_passed,
            "total": overall_total,
            "rate": round(overall_rate, 4),
            "percentage": round(overall_rate * 100, 2),
        },
        "critical_failures": all_critical_failures,
        "critical_failure_count": len(all_critical_failures),
    }


def format_report(results: list[dict[str, Any]]) -> str:
    """Produce a human-readable evaluation report."""
    metrics = compute_metrics(results)

    lines: list[str] = []
    lines.append("Agent Evaluation Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"Total cases: {metrics['overall']['total']}"
    )
    lines.append(
        f"Passed: {metrics['overall']['passed']} / {metrics['overall']['total']}"
    )
    lines.append(
        f"Overall pass rate: {metrics['overall']['percentage']}%"
    )
    lines.append("")
    lines.append("Critical privacy/security failures:")
    lines.append(f"  {metrics['critical_failure_count']}")
    lines.append("")
    lines.append("Results by category:")
    lines.append("-" * 60)
    for cat, data in metrics["categories"].items():
        lines.append(f"  {cat}:")
        lines.append(
            f"    Passed: {data['passed']} / {data['total']} "
            f"({data['percentage']}%)"
        )
        if data["critical_failures"]:
            lines.append(
                f"    Critical failures: {data['critical_failures']}"
            )
        lines.append("")

    lines.append("Failed cases:")
    lines.append("-" * 60)
    failed = [r for r in results if not r["passed"]]
    if not failed:
        lines.append("  None")
    else:
        for result in failed:
            lines.append(f"  {result['id']} ({result['category']}):")
            lines.append(f"    Question: {result['question']}")
            for check in result["check_results"]:
                if not check["passed"]:
                    lines.append(
                        f"    FAILED check [{check['type']}]: "
                        f"{check.get('description', '')}"
                    )
                    lines.append(f"      Expected: {check['expected']!r}")
                    lines.append(f"      Detail: {check['detail']}")
            lines.append("")

    lines.append("Execution environment:")
    lines.append("  - Deterministic fake LLM (no live OpenAI/Ollama provider)")
    lines.append("  - Controlled fake tool outputs (no real API, model artifacts, or dataset)")
    lines.append("  - No AWS credentials, SageMaker, Kafka, or Redpanda required")
    lines.append("")
    return "\n".join(lines)
