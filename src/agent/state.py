from __future__ import annotations

from typing import Any, Literal, TypedDict

AgentTool = Literal[
    "get_application_details",
    "get_dashboard_metrics",
    "get_model_comparison",
    "get_prediction",
]

AgentStatus = Literal[
    "pending",
    "answered",
    "needs_clarification",
    "invalid_input",
    "unsupported_request",
    "partial",
    "success",
    "invalid",
    "not_found",
    "no_prediction",
    "model_unavailable",
    "unavailable",
    "error",
]


class AgentMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class AgentState(TypedDict, total=False):
    """Single-turn agent state; messages retain only the question and response."""

    question: str
    messages: list[AgentMessage]
    selected_tool: AgentTool | None
    tool_arguments: dict[str, int | str]
    tool_result: dict[str, Any] | None
    final_response: str
    status: AgentStatus
    error: str | None