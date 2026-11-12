"""Streamlit chat page for the existing LangGraph credit-risk agent."""
from __future__ import annotations

import logging
from collections.abc import Mapping
import streamlit as st

from agent.state import AgentConversationContext
from dashboard.components.ui import render_page_header

LOGGER = logging.getLogger(__name__)

_HISTORY_KEY = "credit_risk_assistant_history"
_CONTEXT_KEY = "credit_risk_assistant_context"
_CHAT_INPUT_KEY = "credit_risk_assistant_input"


def _clear_conversation() -> None:
    st.session_state[_HISTORY_KEY] = []
    st.session_state[_CONTEXT_KEY] = AgentConversationContext()


def _conversation_context() -> AgentConversationContext:
    context = st.session_state.get(_CONTEXT_KEY)
    if not isinstance(context, AgentConversationContext):
        context = AgentConversationContext()
        st.session_state[_CONTEXT_KEY] = context
    return context


def _request_agent_response(question: str) -> tuple[str, str]:
    """Call only the existing agent entry point and return safe display text."""
    try:
        from agent.graph import AgentConfigurationError, run_agent

        result = run_agent(question, context=_conversation_context())
    except AgentConfigurationError:
        LOGGER.warning("dashboard_agent_configuration_unavailable")
        return (
            "The assistant is not configured. Check the selected language model "
            "provider settings and try again.",
            "error",
        )
    except Exception as error:
        # Keep credentials, prompts, applicant data, and provider diagnostics out
        # of both the dashboard and logs.
        LOGGER.warning(
            "dashboard_agent_request_failed error_type=%s", type(error).__name__
        )
        return (
            "The assistant could not complete that request. Check that the "
            "configured language model is available, then try again.",
            "error",
        )

    if not isinstance(result, Mapping):
        return (
            "The assistant returned an invalid response. Please try again.",
            "error",
        )
    response = result.get("final_response")
    if not isinstance(response, str) or not response.strip():
        return (
            "The assistant did not return a response. Please try again.",
            "error",
        )
    status = result.get("status")
    return response.strip(), status if isinstance(status, str) else "answered"


def _render_assistant_message(content: str, status: str = "answered") -> None:
    if status == "error":
        st.error(content)
    elif status in {"unavailable", "partial", "invalid", "invalid_input"}:
        st.warning(content)
    else:
        st.markdown(content)


def render_assistant() -> None:
    """Render the AI Assistant page inside the dashboard navigation."""
    render_page_header(
        "Credit Risk Assistant",
        "Ask about platform metrics, model results, reports, or a specific application. "
        "Applicant-specific answers use the existing local privacy controls.",
    )

    history_state = st.session_state.get(_HISTORY_KEY)
    if not isinstance(history_state, list):
        st.session_state[_HISTORY_KEY] = []
    _conversation_context()
    _, clear_column = st.columns([5, 1])
    with clear_column:
        st.button(
            "Clear conversation",
            key="clear_credit_risk_assistant",
            on_click=_clear_conversation,
            use_container_width=True,
        )

    st.divider()

    history = st.session_state[_HISTORY_KEY]
    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        with st.chat_message(role):
            if role == "assistant":
                _render_assistant_message(content, message.get("status", "answered"))
            else:
                st.markdown(content)

    submitted = st.chat_input(
        "Ask the credit risk assistant",
        key=_CHAT_INPUT_KEY,
        max_chars=2000,
        submit_mode="disable",
    )
    if not isinstance(submitted, str) or not submitted.strip():
        return

    question = submitted.strip()
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Working on your request..."):
            response, status = _request_agent_response(question)
        _render_assistant_message(response, status)
    history.append({"role": "assistant", "content": response, "status": status})


if __name__ == "__main__":
    render_assistant()
