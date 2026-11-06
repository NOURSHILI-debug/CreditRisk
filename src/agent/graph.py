from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent import prompts, tools
from agent.state import AgentConversationContext, AgentMessage, AgentState, AgentTool

logger = logging.getLogger(__name__)

_APP_ID_PATTERN = re.compile(
    r"\b(?:application|app|applicant)"
    r"(?:\s+(?:details?|information|record|data))?"
    r"(?:\s+(?:is|for))?"
    r"(?:\s+id)?\s*(?:number\s*|no\.?\s*)?[:#]?\s*"
    r"([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?!\w|\.\d)",
    re.IGNORECASE,
)
_CONTEXT_REFERENCE_PATTERN = re.compile(
    r"\b(?:this|that|same|it|previous|last|former|why)\b",
    re.IGNORECASE,
)
_DECISION_REQUEST_PATTERN = re.compile(
    r"\b(?:approve|approving|reject|rejecting|decline|declining|accept|accepting)\b"
    r".*\b(?:application|loan|credit)\b"
    r"|\b(?:loan|credit)\s+(?:approval|rejection|decision)\b"
    r"|\b(?:should|can|may)\s+i\s+(?:lend|approve|reject)\b"
    r"|\b(?:application|loan|credit)\b.*\b(?:eligib(?:le|ility)|approve|reject)\b",
    re.IGNORECASE,
)
_ALLOWED_TOOLS: dict[AgentTool, Callable[..., dict[str, Any]]] = {
    "get_application_details": tools.get_application_details,
    "get_dashboard_metrics": tools.get_dashboard_metrics,
    "get_model_comparison": tools.get_model_comparison,
    "get_prediction": tools.get_prediction,
    "explain_prediction": tools.explain_prediction,
    "generate_report": tools.generate_report,
}
_APPLICATION_TOOLS = {
    "get_application_details",
    "get_prediction",
    "explain_prediction",
}
_ROUTABLE_TOOLS = Literal[
    "get_application_details",
    "get_dashboard_metrics",
    "get_model_comparison",
    "get_prediction",
    "explain_prediction",
    "generate_report",
    "general",
    "unsupported",
]


class AgentConfigurationError(RuntimeError):
    """Raised when an LLM request cannot be configured."""


class RouteDecision(BaseModel):
    """Structured intent decision; tool arguments are always extracted in Python."""

    model_config = ConfigDict(extra="forbid")

    tool: _ROUTABLE_TOOLS | None = None
    tools: list[AgentTool] = Field(default_factory=list)
    needs_clarification: bool = False

    @model_validator(mode="after")
    def validate_route(self) -> "RouteDecision":
        if self.tools and self.tool is not None:
            raise ValueError("Choose either one legacy tool or a tools list.")
        if self.tool in {"general", "unsupported"} and self.tools:
            raise ValueError("General and unsupported routes cannot select tools.")
        return self


def _create_chat_model() -> BaseChatModel:
    provider = os.environ.get("AGENT_LLM_PROVIDER", "openai").strip().lower()
    if provider != "openai":
        raise AgentConfigurationError(
            f"Unsupported agent LLM provider {provider!r}; supported provider: 'openai'."
        )
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AgentConfigurationError(
            "OPENAI_API_KEY is required to run the agent. Set it in the environment "
            "or inject a chat model when building the graph."
        )

    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("AGENT_LLM_MODEL", "gpt-4o-mini").strip()
    if not model_name:
        raise AgentConfigurationError("AGENT_LLM_MODEL must not be empty.")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    options: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": 0,
    }
    if base_url:
        options["base_url"] = base_url
    return ChatOpenAI(**options)


def _extract_application_id(question: str) -> tuple[int | None, str | None]:
    matches = [match.group(1) for match in _APP_ID_PATTERN.finditer(question)]
    if not matches:
        return None, None
    if len(matches) != 1:
        return None, "Provide exactly one application ID."

    raw_id = matches[0]
    if not raw_id.isascii() or not raw_id.isdigit():
        return None, "Application ID must be a positive integer."
    try:
        application_id = int(raw_id)
    except ValueError:
        return None, "Application ID must be a positive integer."
    if application_id <= 0:
        return None, "Application ID must be a positive integer."
    return application_id, None


def _decision_from_model(
    model: BaseChatModel,
    question: str,
    *,
    current_application_id: int | None = None,
    last_intent: str | None = None,
) -> RouteDecision:
    router = model.with_structured_output(RouteDecision)
    messages = [
        SystemMessage(
            content=prompts.ROUTER_SYSTEM_PROMPT.format(
                platform_facts=prompts.PLATFORM_FACTS
            )
        )
    ]
    if current_application_id is not None:
        messages.append(
            SystemMessage(
                content=(
                    "Session context only: current applicant ID is "
                    f"{current_application_id}; previous intent was "
                    f"{last_intent or 'not recorded'}. Use this applicant only "
                    "when the new question clearly refers to the current applicant. "
                    "Previous tool outputs are not fresh evidence; retrieve again."
                )
            )
        )
    messages.append(HumanMessage(content=question))
    decision = router.invoke(messages)
    if isinstance(decision, RouteDecision):
        return decision
    if isinstance(decision, Mapping):
        return RouteDecision.model_validate(decision)
    raise ValueError("The LLM returned an invalid routing response.")


def _message_content(response: Any) -> str:
    if isinstance(response, Mapping):
        content = response.get("content")
    else:
        content = getattr(response, "content", response)
    if isinstance(content, str) and content.strip():
        return content.strip()
    raise ValueError("The LLM returned an empty or non-text response.")


def _append_assistant_message(state: AgentState, content: str) -> list[AgentMessage]:
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": content})
    return messages


def _selected_tools(decision: RouteDecision) -> list[AgentTool]:
    if decision.tools:
        unique_tools = list(dict.fromkeys(decision.tools))
        order = {
            "get_application_details": 0,
            "get_prediction": 1,
            "explain_prediction": 2,
            "get_model_comparison": 3,
            "get_dashboard_metrics": 4,
            "generate_report": 5,
        }
        return sorted(unique_tools, key=order.__getitem__)
    if decision.tool in _ALLOWED_TOOLS:
        return [decision.tool]
    return []


def _result_status(result: Mapping[str, Any]) -> str:
    status = result.get("status")
    if status in {
        "invalid",
        "not_found",
        "no_prediction",
        "model_unavailable",
        "explanation_unavailable",
        "unavailable",
        "error",
    }:
        return str(status)
    if status == "invalid_id":
        return "invalid_input"
    if status == "duplicate":
        return "unavailable"
    if status == "partial":
        return "partial"
    if status == "success":
        return "success"
    if status == "complete":
        return "success"
    if status in {"found", "available"}:
        return "answered"
    return "invalid"


def build_agent_graph(
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    """Build the single-agent workflow, optionally injecting a test/custom model."""
    selected_model = model

    def resolve_model() -> BaseChatModel:
        nonlocal selected_model
        if selected_model is None:
            selected_model = _create_chat_model()
        return selected_model

    def route_request(state: AgentState) -> dict[str, Any]:
        question = state.get("question", "").strip()
        if not question:
            return {
                "status": "invalid_input",
                "selected_tool": None,
                "tool_arguments": {},
                "error": "A non-empty question is required.",
            }
        if _DECISION_REQUEST_PATTERN.search(question):
            return {
                "status": "unsupported_request",
                "selected_tool": None,
                "selected_tools": [],
                "tool_arguments": {},
                "last_intent": "unsupported",
                "current_application_id": None,
            }
        current_application_id = state.get("current_application_id")
        last_intent = state.get("last_intent")
        try:
            decision = _decision_from_model(
                resolve_model(),
                question,
                current_application_id=current_application_id,
                last_intent=last_intent,
            )
        except AgentConfigurationError:
            raise
        except Exception as error:
            logger.error("Agent request routing failed (%s).", type(error).__name__)
            return {
                "status": "error",
                "selected_tool": None,
                "selected_tools": [],
                "tool_arguments": {},
                "error": f"routing_failed:{type(error).__name__}",
            }

        if decision.needs_clarification:
            return {
                "status": "needs_clarification",
                "selected_tool": None,
                "selected_tools": [],
                "tool_arguments": {},
                "last_intent": "clarification",
            }
        if decision.tool == "unsupported":
            return {
                "status": "unsupported_request",
                "selected_tool": None,
                "selected_tools": [],
                "tool_arguments": {},
                "last_intent": "unsupported",
                "current_application_id": None,
            }
        if decision.tool == "general":
            return {
                "status": "pending",
                "selected_tool": None,
                "selected_tools": [],
                "tool_arguments": {},
                "last_intent": "general",
                "current_application_id": None,
            }

        selected_tools = _selected_tools(decision)
        if not selected_tools:
            return {
                "status": "needs_clarification",
                "selected_tool": None,
                "selected_tools": [],
                "tool_arguments": {},
                "last_intent": "clarification",
            }
        if any(tool_name not in _ALLOWED_TOOLS for tool_name in selected_tools):
            return {
                "status": "error",
                "selected_tool": None,
                "selected_tools": [],
                "tool_arguments": {},
                "error": "invalid_tool_selection",
            }

        application_tools = [
            tool_name for tool_name in selected_tools
            if tool_name in _APPLICATION_TOOLS
        ]
        application_id: int | None = None
        if application_tools:
            application_id, validation_error = _extract_application_id(question)
            if validation_error is not None:
                return {
                    "status": "invalid_input",
                    "selected_tool": None,
                    "selected_tools": [],
                    "tool_arguments": {},
                    "error": validation_error,
                    "current_application_id": None,
                    "last_intent": "invalid_input",
                }
            if application_id is None:
                has_context_reference = bool(
                    _CONTEXT_REFERENCE_PATTERN.search(question)
                )
                is_explanation_followup = (
                    "explain_prediction" in application_tools
                    and bool(
                        {"get_prediction", "explain_prediction"}.intersection(
                            (last_intent or "").split("+")
                        )
                    )
                )
                if (
                    current_application_id is not None
                    and (has_context_reference or is_explanation_followup)
                ):
                    application_id = current_application_id
                else:
                    return {
                        "status": "needs_clarification",
                        "selected_tool": None,
                        "selected_tools": [],
                        "tool_arguments": {},
                        "last_intent": "clarification",
                    }

        if application_id is not None:
            if (
                isinstance(application_id, bool)
                or not isinstance(application_id, int)
                or application_id <= 0
            ):
                return {
                    "status": "invalid_input",
                    "selected_tool": None,
                    "selected_tools": [],
                    "tool_arguments": {},
                    "error": "Application ID must be a positive integer.",
                }
            arguments: dict[str, int | str] = {"application_id": application_id}
            intent = "+".join(selected_tools)
            retained_id = application_id
        else:
            arguments = {}
            if "generate_report" in selected_tools:
                report_periods = re.findall(
                    r"\b(\d+\s*(?:d|days?)|all)\b", question, re.IGNORECASE
                )
                if report_periods:
                    period = re.sub(r"\s+", "", report_periods[0]).lower()
                    if period.endswith("days"):
                        period = period[:-4] + "d"
                    elif period.endswith("day"):
                        period = period[:-3] + "d"
                    arguments["period"] = period
                else:
                    arguments["period"] = "all"
            intent = "+".join(selected_tools)
            retained_id = None
        return {
            "status": "pending",
            "selected_tool": selected_tools[0],
            "selected_tools": selected_tools,
            "tool_arguments": arguments,
            "current_application_id": retained_id,
            "last_intent": intent,
        }

    def execute_tool(state: AgentState) -> dict[str, Any]:
        selected_tools = state.get("selected_tools") or (
            [state["selected_tool"]]
            if state.get("selected_tool") is not None
            else []
        )
        if not selected_tools or any(
            selected_tool not in _ALLOWED_TOOLS
            for selected_tool in selected_tools
        ):
            return {
                "status": "error",
                "error": "invalid_tool_selection",
                "tool_result": None,
                "tool_results": {},
                "tool_statuses": {},
            }

        arguments = state.get("tool_arguments", {})
        question_id, validation_error = _extract_application_id(
            state.get("question", "")
        )
        application_id = arguments.get("application_id")
        has_application_tool = any(
            selected_tool in _APPLICATION_TOOLS
            for selected_tool in selected_tools
        )
        if has_application_tool and (
            isinstance(application_id, bool)
            or not isinstance(application_id, int)
            or application_id <= 0
            or (
                question_id is not None
                and question_id != application_id
            )
            or validation_error is not None
        ):
            return {
                "status": "invalid_input",
                "error": validation_error or "Application ID validation failed.",
                "tool_result": None,
                "tool_results": {},
                "tool_statuses": {},
            }
        if not has_application_tool and set(arguments) - {"period"}:
            return {
                "status": "invalid_input",
                "error": "Unexpected tool arguments were rejected.",
                "tool_result": None,
                "tool_results": {},
                "tool_statuses": {},
            }

        results: dict[str, dict[str, Any]] = {}
        statuses: dict[str, str] = {}
        execution_errors: list[str] = []
        for selected_tool in selected_tools:
            tool_arguments: dict[str, int | str] = {}
            if selected_tool in _APPLICATION_TOOLS:
                # The ID comes from Python extraction or the session context, never the router.
                tool_arguments["application_id"] = int(application_id)
            elif selected_tool == "generate_report":
                period = arguments.get("period", "all")
                if not isinstance(period, str):
                    return {
                        "status": "invalid_input",
                        "error": "Report period must be one of 7d, 30d, 90d, or all.",
                        "tool_result": None,
                        "tool_results": {},
                        "tool_statuses": {},
                    }
                tool_arguments["period"] = period
            try:
                result = _ALLOWED_TOOLS[selected_tool](**tool_arguments)
                if not isinstance(result, dict):
                    raise TypeError("Agent tools must return a dictionary.")
                json.dumps(result, allow_nan=False)
                if selected_tool in {"get_prediction", "explain_prediction"}:
                    result_application_id = result.get("application_id")
                    if (
                        result_application_id is not None
                        and (
                            isinstance(result_application_id, bool)
                            or result_application_id != application_id
                        )
                    ):
                        raise ValueError("Prediction result application ID mismatch.")
                    if (
                        result.get("status") in {"available", "success"}
                        and result_application_id != application_id
                    ):
                        raise ValueError("Prediction result application ID is missing.")
                if result.get("status") not in {
                    "complete",
                    "available",
                    "success",
                    "found",
                    "partial",
                    "unavailable",
                    "invalid",
                    "invalid_id",
                    "duplicate",
                    "not_found",
                    "no_prediction",
                    "model_unavailable",
                    "explanation_unavailable",
                    "error",
                }:
                    result = {
                        "status": "invalid",
                        "message": "The tool returned an unexpected or malformed result.",
                    }
                status = _result_status(result)
            except Exception as error:
                logger.error(
                    "Agent tool %s failed (%s).",
                    selected_tool,
                    type(error).__name__,
                )
                result = {
                    "status": "error",
                    "message": "The requested data could not be retrieved safely.",
                }
                status = "error"
                execution_errors.append(f"tool_failed:{type(error).__name__}")
            results[selected_tool] = result
            statuses[selected_tool] = status

        if len(results) == 1:
            tool_result = next(iter(results.values()))
            overall_status = next(iter(statuses.values()))
        else:
            successful = sum(
                status in {"answered", "success"} for status in statuses.values()
            )
            if successful == len(statuses):
                overall_status = "answered"
            elif successful:
                overall_status = "partial"
            elif len(set(statuses.values())) == 1:
                overall_status = next(iter(statuses.values()))
            else:
                overall_status = "error"
            tool_result = {
                "tool_results": results,
                "tool_statuses": statuses,
                "source_mode": "configured",
            }

        try:
            json.dumps(tool_result, allow_nan=False)
        except (TypeError, ValueError):
            return {
                "status": "invalid",
                "error": "The tool result was not JSON serializable.",
                "tool_result": None,
                "tool_results": results,
                "tool_statuses": statuses,
            }
        return {
            "tool_result": tool_result,
            "tool_results": results,
            "tool_statuses": statuses,
            "status": overall_status,
            "error": (
                execution_errors[0]
                if len(execution_errors) == 1
                else "multiple_tool_failures"
                if execution_errors
                else None
            ),
        }

    def generate_response(state: AgentState) -> dict[str, Any]:
        status = state.get("status", "pending")
        error = state.get("error")
        tool_result = state.get("tool_result")

        if status == "needs_clarification":
            answer = "Which application ID should I use? Please provide a positive integer ID."
            return {
                "final_response": answer,
                "messages": _append_assistant_message(state, answer),
            }
        if status in {"invalid_input", "invalid"}:
            answer = error or "The request contains invalid input."
            return {
                "final_response": answer,
                "messages": _append_assistant_message(state, answer),
            }
        comparison_result = None
        if state.get("selected_tool") == "get_model_comparison":
            comparison_result = tool_result
        elif isinstance(tool_result, Mapping):
            tool_results = tool_result.get("tool_results")
            if isinstance(tool_results, Mapping):
                candidate = tool_results.get("get_model_comparison")
                if isinstance(candidate, Mapping):
                    comparison_result = candidate
        if (
            comparison_result is not None
            and comparison_result.get("comparison_available") is False
        ):
            note = comparison_result.get("comparison_note")
            answer = (
                "A direct model comparison is not justified. "
                f"{note if isinstance(note, str) and note else 'Evaluation protocol comparability is unavailable.'}\n"
                f"{json.dumps(tool_result, ensure_ascii=False, indent=2, allow_nan=False)}"
            )
            return {
                "final_response": answer,
                "messages": _append_assistant_message(state, answer),
            }
        if status in {"not_found", "no_prediction", "model_unavailable"}:
            messages = {
                "not_found": "No application was found for that ID.",
                "no_prediction": "No prediction is available for that application.",
                "model_unavailable": "The prediction model is unavailable.",
            }
            answer = messages[status]
            return {
                "final_response": answer,
                "messages": _append_assistant_message(state, answer),
            }
        if status == "explanation_unavailable":
            answer = (
                "The prediction is available, but its SHAP explanation is "
                "unavailable. No feature contributions can be reported."
            )
            return {
                "final_response": answer,
                "messages": _append_assistant_message(state, answer),
            }
        if status == "unsupported_request":
            answer = (
                "I can provide application information and retrieved model outputs, "
                "but I cannot approve or reject applications or make lending decisions."
            )
            return {
                "final_response": answer,
                "messages": _append_assistant_message(state, answer),
            }
        if status == "error":
            answer = "I couldn't safely complete that request. Please try again later."
            return {
                "final_response": answer,
                "messages": _append_assistant_message(state, answer),
            }
        if status == "partial" and tool_result is not None:
            source_mode = tool_result.get("source_mode")
            prefix = (
                "Partial results were returned. Missing values are unavailable; "
                "the data below contains only retrieved values."
            )
            if comparison_result is not None and comparison_result.get(
                "comparison_available"
            ) is False:
                note = comparison_result.get("comparison_note")
                prefix += (
                    " A direct model comparison is not justified. "
                    f"{note if isinstance(note, str) and note else ''}"
                )
            if source_mode == "mock":
                prefix += " The configured source is labeled mock/demo data."
            answer = (
                f"{prefix}\n"
                f"{json.dumps(tool_result, ensure_ascii=False, indent=2, allow_nan=False)}"
            )
            return {
                "final_response": answer,
                "messages": _append_assistant_message(state, answer),
            }
        if tool_result is not None and status in {"unavailable", "invalid"}:
            source_mode = tool_result.get("source_mode")
            prefix = "The retrieved result is unavailable."
            if status == "invalid":
                prefix = "The retrieved result was invalid and cannot be interpreted."
            if source_mode == "mock":
                prefix += " The configured source is labeled mock/demo data."
            message = tool_result.get("message")
            answer = f"{prefix} {message}" if isinstance(message, str) and message else prefix
            return {
                "final_response": answer,
                "messages": _append_assistant_message(state, answer),
            }

        try:
            model_response = resolve_model().invoke(
                [
                    SystemMessage(
                        content=prompts.RESPONSE_SYSTEM_PROMPT.format(
                            platform_facts=prompts.PLATFORM_FACTS
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "question": state.get("question", ""),
                                "tool_result": tool_result,
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    ),
                ]
            )
            answer = _message_content(model_response)
            if (
                tool_result is not None
                and tool_result.get("source_mode") == "mock"
                and not answer.lower().startswith(("mock", "demo"))
            ):
                answer = f"Mock/demo data: {answer}"
        except AgentConfigurationError:
            raise
        except Exception as error:
            logger.error(
                "Agent response generation failed (%s).",
                type(error).__name__,
            )
            return {
                "status": "error",
                "error": f"response_failed:{type(error).__name__}",
                "final_response": "I couldn't safely generate a response. Please try again later.",
                "messages": _append_assistant_message(
                    state,
                    "I couldn't safely generate a response. Please try again later.",
                ),
            }

        return {
            "status": "success" if status == "success" else "answered",
            "final_response": answer,
            "messages": _append_assistant_message(state, answer),
        }

    def route_after_selection(state: AgentState) -> str:
        if state.get("status") in {
            "needs_clarification",
            "invalid_input",
            "unsupported_request",
            "error",
        }:
            return "generate_response"
        if state.get("selected_tool") is not None:
            return "execute_tool"
        return "generate_response"

    workflow = StateGraph(AgentState)
    workflow.add_node("route_request", route_request)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("generate_response", generate_response)
    workflow.add_edge(START, "route_request")
    workflow.add_conditional_edges(
        "route_request",
        route_after_selection,
        {
            "execute_tool": "execute_tool",
            "generate_response": "generate_response",
        },
    )
    workflow.add_edge("execute_tool", "generate_response")
    workflow.add_edge("generate_response", END)
    return workflow.compile()


def run_agent(
    question: str,
    *,
    model: BaseChatModel | None = None,
    context: AgentConversationContext | None = None,
) -> AgentState:
    """Run one turn, optionally updating caller-owned session context."""
    if not isinstance(question, str) or not question.strip():
        return AgentState(
            question=question if isinstance(question, str) else "",
            messages=[],
            selected_tool=None,
            selected_tools=[],
            tool_arguments={},
            tool_result=None,
            tool_results={},
            tool_statuses={},
            final_response="A non-empty question is required.",
            status="invalid_input",
            error="A non-empty question is required.",
        )
    graph = build_agent_graph(model)
    initial_state = AgentState(
        question=question.strip(),
        messages=[{"role": "user", "content": question.strip()}],
        selected_tool=None,
        selected_tools=[],
        tool_arguments={},
        tool_result=None,
        tool_results={},
        tool_statuses={},
        current_application_id=(
            context.current_application_id if context is not None else None
        ),
        last_intent=context.last_intent if context is not None else None,
        final_response="",
        status="pending",
        error=None,
    )
    result = graph.invoke(initial_state)
    if context is not None:
        context.current_application_id = result.get("current_application_id")
        context.last_intent = result.get("last_intent")
    return result
