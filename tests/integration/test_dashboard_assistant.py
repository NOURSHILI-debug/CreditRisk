from __future__ import annotations

import streamlit as st
from streamlit.testing.v1 import AppTest

from agent import graph
from agent.state import AgentConversationContext
from dashboard import data_access


def _open_assistant(monkeypatch, project_root) -> AppTest:
    import dashboard.pages.overview as overview

    monkeypatch.setattr(
        data_access,
        "get_api_health",
        lambda: {"status": "ok", "model_name": "XGBoost", "model_version": "test"},
    )
    monkeypatch.setattr(overview, "render_overview", lambda: st.title("Overview"))
    app = AppTest.from_file(str(project_root / "src" / "dashboard" / "app.py")).run()
    assert not app.exception
    app.switch_page("pages/assistant.py").run()
    return app


def _messages(app: AppTest) -> list[dict[str, str]]:
    return app.session_state["credit_risk_assistant_history"]


def test_assistant_page_is_registered_and_renders_native_chat(
    monkeypatch, project_root
) -> None:
    app = _open_assistant(monkeypatch, project_root)

    assert not app.exception
    assert [title.value for title in app.title] == ["Credit Risk Assistant"]
    assert len(app.chat_input) == 1
    assert app.button(key="clear_credit_risk_assistant").label == "Clear conversation"


def test_user_message_calls_existing_agent_and_renders_its_answer(
    monkeypatch, project_root
) -> None:
    calls: list[tuple[str, AgentConversationContext]] = []

    def fake_run_agent(
        question: str, *, context: AgentConversationContext
    ) -> dict[str, str]:
        calls.append((question, context))
        return {"status": "answered", "final_response": "Retrieved dashboard metrics."}

    monkeypatch.setattr(graph, "run_agent", fake_run_agent)
    app = _open_assistant(monkeypatch, project_root)
    app.chat_input[0].set_value("Show dashboard metrics.").run()

    assert not app.exception
    assert len(calls) == 1
    assert calls[0][0] == "Show dashboard metrics."
    assert isinstance(calls[0][1], AgentConversationContext)
    assert "Retrieved dashboard metrics." in [item.value for item in app.markdown]
    assert _messages(app) == [
        {"role": "user", "content": "Show dashboard metrics."},
        {
            "role": "assistant",
            "content": "Retrieved dashboard metrics.",
            "status": "answered",
        },
    ]


def test_history_and_agent_context_survive_reruns_without_duplicate_submission(
    monkeypatch, project_root
) -> None:
    calls: list[tuple[str, AgentConversationContext]] = []

    def fake_run_agent(
        question: str, *, context: AgentConversationContext
    ) -> dict[str, str]:
        calls.append((question, context))
        if question == "Get prediction for application 1001.":
            context.current_application_id = 1001
            context.last_intent = "get_prediction"
        return {"status": "answered", "final_response": f"Response to: {question}"}

    monkeypatch.setattr(graph, "run_agent", fake_run_agent)
    app = _open_assistant(monkeypatch, project_root)
    app.chat_input[0].set_value("Get prediction for application 1001.").run()
    app.run()

    assert len(calls) == 1
    assert len(_messages(app)) == 2
    app.chat_input[0].set_value("Explain that prediction.").run()

    assert not app.exception
    assert len(calls) == 2
    assert calls[0][1] is calls[1][1]
    assert calls[1][1].current_application_id == 1001
    assert len(_messages(app)) == 4
    assert [message["role"] for message in _messages(app)] == [
        "user", "assistant", "user", "assistant"
    ]


def test_clear_conversation_resets_visible_history_and_agent_context(
    monkeypatch, project_root
) -> None:
    def fake_run_agent(
        _question: str, *, context: AgentConversationContext
    ) -> dict[str, str]:
        context.current_application_id = 1001
        context.last_intent = "get_prediction"
        return {"status": "answered", "final_response": "Prediction retrieved."}

    monkeypatch.setattr(graph, "run_agent", fake_run_agent)
    app = _open_assistant(monkeypatch, project_root)
    app.chat_input[0].set_value("Get prediction for application 1001.").run()
    assert len(_messages(app)) == 2

    app.button(key="clear_credit_risk_assistant").click().run()

    assert not app.exception
    assert _messages(app) == []
    context = app.session_state["credit_risk_assistant_context"]
    assert context.current_application_id is None
    assert context.last_intent is None


def test_agent_exception_is_user_friendly_and_sensitive_details_stay_out_of_ui(
    monkeypatch, project_root, caplog
) -> None:
    def fail_agent(_question: str, *, context: AgentConversationContext) -> None:
        raise RuntimeError("secret-key and applicant-1001 internal traceback")

    monkeypatch.setattr(graph, "run_agent", fail_agent)
    app = _open_assistant(monkeypatch, project_root)
    app.chat_input[0].set_value("Ask a question.").run()

    assert not app.exception
    assert app.error
    visible = " ".join(item.value for item in app.error)
    assert "could not complete" in visible
    assert "secret-key" not in visible
    assert "applicant-1001" not in visible
    assert "secret-key" not in caplog.text
    assert _messages(app)[-1]["status"] == "error"


def test_agent_configuration_failure_does_not_expose_provider_details(
    monkeypatch, project_root
) -> None:
    def fail_agent(_question: str, *, context: AgentConversationContext) -> None:
        raise graph.AgentConfigurationError("secret API key is missing")

    monkeypatch.setattr(graph, "run_agent", fail_agent)
    app = _open_assistant(monkeypatch, project_root)
    app.chat_input[0].set_value("Ask a question.").run()

    assert not app.exception
    assert app.error
    visible = " ".join(item.value for item in app.error)
    assert "not configured" in visible
    assert "secret" not in visible
    assert "API key" not in visible


def test_unavailable_agent_result_is_shown_as_unavailable(
    monkeypatch, project_root
) -> None:
    monkeypatch.setattr(
        graph,
        "run_agent",
        lambda _question, *, context: {
            "status": "unavailable",
            "final_response": "Dashboard metrics are unavailable from the configured API.",
            "tool_result": {"metrics": None},
        },
    )
    app = _open_assistant(monkeypatch, project_root)
    app.chat_input[0].set_value("Show dashboard metrics.").run()

    assert not app.exception
    assert app.warning
    assert "unavailable" in app.warning[0].value.casefold()
    assert "tool_result" not in str(_messages(app))
    assert "'metrics': None" not in str(_messages(app))


def test_chat_keeps_only_agent_display_text_not_raw_tool_results(
    monkeypatch, project_root
) -> None:
    raw_result = {"application_id": 1001, "fields": {"AMT_INCOME_TOTAL": 75000}}
    monkeypatch.setattr(
        graph,
        "run_agent",
        lambda _question, *, context: {
            "status": "answered",
            "final_response": "The requested field is unavailable.",
            "tool_result": raw_result,
        },
    )
    app = _open_assistant(monkeypatch, project_root)
    app.chat_input[0].set_value("What is the income for application 1001?").run()

    assert not app.exception
    history_text = str(_messages(app))
    assert "75000" not in history_text
    assert "tool_result" not in history_text
    assert "The requested field is unavailable." in history_text


def test_assistant_exposes_the_existing_six_tool_agent(monkeypatch, project_root) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        graph,
        "run_agent",
        lambda question, *, context: calls.append(question)
        or {"status": "answered", "final_response": "Request handled."},
    )
    app = _open_assistant(monkeypatch, project_root)
    app.chat_input[0].set_value("Compare model results.").run()

    assert calls == ["Compare model results."]
    assert set(graph._ALLOWED_TOOLS) == {
        "get_application_details",
        "get_dashboard_metrics",
        "get_model_comparison",
        "get_prediction",
        "explain_prediction",
        "generate_report",
    }
