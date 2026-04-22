from types import SimpleNamespace
from unittest.mock import patch

from byte import Config
from byte.adapter.pipeline import _request_setup as request_setup


class _DummyCache:
    def __init__(self) -> None:
        self.config = Config(enable_token_counter=False)


def test_prepare_request_for_execution_populates_contract_and_reuse_policy() -> None:
    chat_cache = _DummyCache()
    request_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Say hi"}],
    }
    context = {}

    with patch.object(
        request_setup,
        "_compile_context_if_needed",
        side_effect=lambda chat_cache, kwargs, context, session=None: dict(kwargs),
    ), patch.object(request_setup, "_plan_workflow", return_value=None), patch.object(
        request_setup, "_maybe_reasoning_shortcut", return_value=None
    ), patch.object(request_setup, "_maybe_route_request", return_value=None):
        prepared = request_setup.prepare_request_for_execution(
            chat_cache,
            request_kwargs,
            context,
            session=None,
        )

    assert prepared.early_response is None
    assert prepared.context["_byte_request_kwargs"]["model"] == "gpt-4o-mini"
    assert isinstance(prepared.context["_byte_output_contract"], dict)
    assert prepared.context["_byte_reuse_policy"]["mode"] == "full_reuse"


def test_prepare_request_for_execution_returns_clarification_response_when_planner_requests_it() -> None:
    chat_cache = _DummyCache()
    request_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Help"}],
    }
    workflow_decision = SimpleNamespace(action="clarify", response_text="Need more detail.")

    with patch.object(
        request_setup,
        "_compile_context_if_needed",
        side_effect=lambda chat_cache, kwargs, context, session=None: dict(kwargs),
    ), patch.object(request_setup, "_plan_workflow", return_value=workflow_decision):
        prepared = request_setup.prepare_request_for_execution(
            chat_cache,
            request_kwargs,
            {},
            session=None,
        )

    assert prepared.early_response is not None
    assert prepared.early_response["byte_reason"] == "clarification_required"
    assert prepared.early_response["choices"][0]["message"]["content"] == "Need more detail."
