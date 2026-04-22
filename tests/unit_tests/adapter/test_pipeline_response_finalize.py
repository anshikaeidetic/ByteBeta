from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from byte.adapter.pipeline import (
    _response_assessment,
    _response_commit,
    _response_finalize,
    _response_streaming,
)


def test_run_sync_response_assessment_escalates_when_route_requires_it() -> None:
    route_decision = SimpleNamespace(
        tier="cheap",
        selected_model="gpt-4o-mini",
        route_key="chat",
        category="chat",
        signals={},
    )
    escalated_decision = SimpleNamespace(
        tier="balanced",
        selected_model="gpt-4.1",
        route_key="chat",
        category="chat",
        signals={},
    )
    first_assessment = SimpleNamespace(accepted=False, reason="needs_upgrade", constraint="freeform")
    second_assessment = SimpleNamespace(accepted=True, reason="accepted", constraint="freeform")
    quality_scorer = MagicMock()
    llm_handler = MagicMock(return_value={"id": "escalated"})
    request_kwargs = {"model": "gpt-4o-mini"}
    context = {"_byte_model_route": route_decision, "_byte_task_policy": {}}

    with patch.object(
        _response_assessment,
        "_assess_and_repair_response",
        side_effect=[
            ({"id": "initial"}, first_assessment),
            ({"id": "escalated"}, second_assessment),
        ],
    ), patch.object(
        _response_assessment,
        "_run_cheap_consensus_sync",
        side_effect=lambda *call_args, **call_kwargs: (call_args[5], call_args[6]),
    ), patch.object(
        _response_assessment,
        "_run_verifier_model_sync",
        side_effect=[
            ({"id": "initial"}, first_assessment),
            ({"id": "escalated"}, second_assessment),
        ],
    ), patch.object(
        _response_assessment,
        "_should_escalate_routed_response",
        side_effect=[True, False],
    ), patch.object(
        _response_assessment,
        "_resolve_escalation_target",
        return_value=("gpt-4.1", "balanced"),
    ), patch.object(
        _response_assessment,
        "_provider_request_kwargs",
        side_effect=lambda chat_cache, request_kwargs, context: dict(request_kwargs),
    ), patch.object(
        _response_assessment,
        "_make_escalated_decision",
        return_value=escalated_decision,
    ), patch.object(
        _response_assessment,
        "_record_failure_memory",
    ) as record_failure_memory, patch.object(
        _response_assessment,
        "get_quality_scorer",
        return_value=quality_scorer,
    ), patch.object(
        _response_assessment,
        "record_route_outcome",
    ):
        llm_data, response_assessment = _response_finalize.run_sync_response_assessment(
            chat_cache=SimpleNamespace(
                config=SimpleNamespace(),
                report=SimpleNamespace(llm=MagicMock()),
            ),
            llm_handler=llm_handler,
            args=(),
            request_kwargs=request_kwargs,
            context=context,
            llm_data={"id": "initial"},
            start_time=0.0,
            coalesce_key=None,
        )

    assert llm_data == {"id": "escalated"}
    assert response_assessment is second_assessment
    assert request_kwargs["model"] == "gpt-4.1"
    assert context["_byte_model_route"] is escalated_decision
    quality_scorer.record_escalation.assert_called_once()
    record_failure_memory.assert_called_once()
    llm_handler.assert_called_once()


def test_finalize_sync_llm_response_completes_coalescer_and_records_workflow_outcome() -> None:
    chat_cache = SimpleNamespace(config=SimpleNamespace())
    coalescer = MagicMock()
    llm_data = {
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    with patch.object(
        _response_streaming,
        "_record_route_completion",
        return_value=(None, True),
    ), patch.object(
        _response_streaming,
        "_record_ai_memory",
    ) as record_ai_memory, patch.object(
        _response_streaming,
        "_record_execution_memory",
    ) as record_execution_memory, patch.object(
        _response_streaming,
        "_record_reasoning_memory",
    ) as record_reasoning_memory, patch.object(
        _response_streaming,
        "_record_workflow_outcome",
    ) as record_workflow_outcome, patch.object(
        _response_commit,
        "get_coalescer",
        return_value=coalescer,
    ):
        response = _response_finalize.finalize_sync_llm_response(
            chat_cache=chat_cache,
            request_kwargs={"model": "gpt-4o-mini"},
            context={},
            llm_data=llm_data,
            response_assessment=SimpleNamespace(accepted=True),
            embedding_data="embedding",
            pre_store_data="prompt",
            session=None,
            start_time=0.0,
            cache_enable=False,
            coalesce_key="coalesce-key",
            update_cache_callback=lambda llm_data, update_cache_func, *args, **kwargs: llm_data,
            args=(),
        )

    assert response is llm_data
    coalescer.complete.assert_called_once_with("coalesce-key", llm_data)
    record_ai_memory.assert_called_once()
    record_execution_memory.assert_called_once()
    record_reasoning_memory.assert_called_once()
    record_workflow_outcome.assert_called_once_with(
        chat_cache,
        {"model": "gpt-4o-mini"},
        {},
        success=True,
        reason="completed",
    )
