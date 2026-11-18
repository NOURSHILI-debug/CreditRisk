"""Deterministic test fixtures for the agent evaluation framework.

Provides a fake chat model and call-tracking fake tools that mirror the
fixtures used in ``tests/unit/test_agent.py`` so the evaluation runner can
exercise the real agent graph without a live LLM provider, real credentials,
or real model artifacts.
"""

from __future__ import annotations

from typing import Any


class FakeStructuredRouter:
    """Structured-output stand-in returned by ``FakeChatModel.with_structured_output``."""

    def __init__(
        self,
        tool: str,
        captured: list[object] | None = None,
        router_return: dict[str, Any] | None = None,
    ) -> None:
        self.tool = tool
        self.captured = captured
        self.router_return = router_return

    def invoke(self, messages: object) -> dict[str, str]:
        if self.captured is not None:
            self.captured.append(messages)
        if self.router_return is not None:
            return dict(self.router_return)
        return {"tool": self.tool}


class FakeChatModel:
    """Deterministic stand-in for a chat model.

    Captures every input passed to the routing LLM and the response LLM so the
    evaluation can assert that applicant identifiers and untrusted evidence are
    handled correctly without invoking a real model.
    """

    def __init__(
        self,
        tool: str = "general",
        response: str = "Grounded response.",
        router_return: dict[str, Any] | None = None,
    ) -> None:
        self.tool = tool
        self.response = response
        self.router_return = router_return
        self.response_calls = 0
        self.route_inputs: list[object] = []
        self.response_inputs: list[object] = []

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> FakeStructuredRouter:
        return FakeStructuredRouter(self.tool, self.route_inputs, self.router_return)

    def invoke(self, messages: object) -> dict[str, str]:
        self.response_calls += 1
        self.response_inputs.append(messages)
        return {"content": self.response}


class CallTrackingFakeTool:
    """Fake tool that records whether it was called and with what arguments.

    Used to verify that the agent passes the correct arguments to the selected
    tool and that it does not invoke a tool when validation fails.
    """

    def __init__(
        self,
        result: dict[str, Any],
        expected_args: dict[str, Any],
        raise_exception: bool = False,
    ) -> None:
        self.result = result
        self.expected_args = expected_args
        self.called = False
        self.actual_args: dict[str, Any] | None = None
        self.raise_exception = raise_exception

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.called = True
        self.actual_args = dict(kwargs)
        if self.raise_exception:
            raise RuntimeError("sensitive provider detail")
        if kwargs != self.expected_args:
            raise AssertionError(
                f"Tool called with unexpected arguments: "
                f"expected {self.expected_args!r}, got {kwargs!r}"
            )
        return dict(self.result)
