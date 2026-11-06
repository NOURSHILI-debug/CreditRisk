from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

AgentTool = Literal[
    "get_application_details",
    "get_dashboard_metrics",
    "get_model_comparison",
    "get_prediction",
    "explain_prediction",
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
    "explanation_unavailable",
    "unavailable",
    "error",
]


class AgentMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass
class AgentConversationContext:
    """Caller-owned, session-scoped context; stores no applicant records or results."""

    current_application_id: int | None = None
    last_intent: str | None = None


class AgentState(TypedDict, total=False):
    """One graph turn plus minimal caller-provided session context."""

    question: str
    messages: list[AgentMessage]
    selected_tool: AgentTool | None
    selected_tools: list[AgentTool]
    tool_arguments: dict[str, int | str]
    tool_result: dict[str, Any] | None
    tool_results: dict[str, dict[str, Any]]
    tool_statuses: dict[str, str]
    current_application_id: int | None
    last_intent: str | None
    final_response: str
    status: AgentStatus
    error: str | None