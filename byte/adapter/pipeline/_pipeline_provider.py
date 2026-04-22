"""Provider miss-path execution and escalation stages for the adapter pipeline."""

from __future__ import annotations

import time
from typing import Any

from byte.processor.model_router import record_route_outcome
from byte.utils.time import time_cal

from ._pipeline_common import NO_RESULT, awith_coalescer_guard, with_coalescer_guard
from ._pipeline_state import PipelineRunState
from .consensus import _run_cheap_consensus_async, _run_cheap_consensus_sync
from .context import _provider_request_kwargs
from .escalation import (
    _escalation_action_for_tier,
    _make_escalated_decision,
    _resolve_escalation_target,
    _should_escalate_routed_response,
)
from .memory import _record_failure_memory
from .utils import _await_with_report, _try_record_budget, get_quality_scorer
from .verifier import (
    _assess_and_repair_response,
    _run_verifier_model_async,
    _run_verifier_model_sync,
)


def execute_provider_stage_sync(
    state: PipelineRunState, *, recursive_adapter: Any
) -> Any:
    """Execute the miss path for a sync pipeline run."""

    next_cache = state.chat_cache.next_cache
    if next_cache is not None:
        recursive_kwargs = dict(state.kwargs)
        recursive_kwargs["cache_obj"] = next_cache
        recursive_kwargs["cache_context"] = state.context
        recursive_kwargs["cache_skip"] = state.cache_skip
        recursive_kwargs["cache_factor"] = state.cache_factor
        recursive_kwargs["search_only"] = state.search_only_flag
        state.llm_data = with_coalescer_guard(
            state,
            lambda: recursive_adapter(
                state.llm_handler,
                state.cache_data_convert,
                state.update_cache_callback,
                *state.args,
                **recursive_kwargs,
            ),
            stage="next_cache",
        )
    else:
        if state.search_only_flag:
            return NO_RESULT
        state.llm_data = with_coalescer_guard(
            state,
            lambda: _provider_request_sync(state, state.kwargs),
            stage="provider_request",
        )

    if not state.llm_data:
        return NO_RESULT
    _try_record_budget(state.llm_data, state.kwargs.get("model", "unknown"), chat_cache=state.chat_cache)
    _assess_response_sync(state)
    _run_consensus_and_verifier_sync(state)
    _run_escalation_sync(state)
    return state.llm_data


async def execute_provider_stage_async(
    state: PipelineRunState, *, recursive_adapter: Any
) -> Any:
    """Execute the miss path for an async pipeline run."""

    next_cache = state.chat_cache.next_cache
    if next_cache is not None:
        recursive_kwargs = dict(state.kwargs)
        recursive_kwargs["cache_obj"] = next_cache
        recursive_kwargs["cache_context"] = state.context
        recursive_kwargs["cache_skip"] = state.cache_skip
        recursive_kwargs["cache_factor"] = state.cache_factor
        recursive_kwargs["search_only"] = state.search_only_flag
        state.llm_data = await awith_coalescer_guard(
            state,
            lambda: recursive_adapter(
                state.llm_handler,
                state.cache_data_convert,
                state.update_cache_callback,
                *state.args,
                **recursive_kwargs,
            ),
            stage="next_cache",
        )
    else:
        if state.search_only_flag:
            return NO_RESULT
        state.llm_data = await awith_coalescer_guard(
            state,
            lambda: _provider_request_async(state, state.kwargs),
            stage="provider_request",
        )

    if not state.llm_data:
        return NO_RESULT
    _try_record_budget(state.llm_data, state.kwargs.get("model", "unknown"), chat_cache=state.chat_cache)
    _assess_response_sync(state)
    await _run_consensus_and_verifier_async(state)
    await _run_escalation_async(state)
    return state.llm_data


def _inject_reference_hint(provider_kwargs: dict[str, Any], hint: str) -> dict[str, Any]:
    """Prepend a cached-answer reference hint to the system prompt (dual-threshold Stage 2)."""
    messages = provider_kwargs.get("messages")
    if not messages or not isinstance(messages, list):
        return provider_kwargs
    hint_text = (
        f"[Reference context from semantic cache — use as a guide, not verbatim]: {hint}"
    )
    updated = list(messages)
    if updated and updated[0].get("role") == "system":
        updated[0] = dict(updated[0])
        updated[0]["content"] = hint_text + "\n\n" + str(updated[0].get("content", ""))
    else:
        updated.insert(0, {"role": "system", "content": hint_text})
    result = dict(provider_kwargs)
    result["messages"] = updated
    return result


def _provider_request_sync(state: PipelineRunState, request_kwargs: dict[str, Any]) -> Any:
    provider_kwargs = _provider_request_kwargs(state.chat_cache, request_kwargs, state.context)
    hint = state.context.pop("_byte_reference_hint", None)
    if hint:
        provider_kwargs = _inject_reference_hint(provider_kwargs, hint)
    return time_cal(
        state.llm_handler,
        func_name="llm_request",
        report_func=state.chat_cache.report.llm,
    )(*state.args, **provider_kwargs)


async def _provider_request_async(state: PipelineRunState, request_kwargs: dict[str, Any]) -> Any:
    provider_kwargs = _provider_request_kwargs(state.chat_cache, request_kwargs, state.context)
    hint = state.context.pop("_byte_reference_hint", None)
    if hint:
        provider_kwargs = _inject_reference_hint(provider_kwargs, hint)
    return await _await_with_report(
        state.llm_handler(*state.args, **provider_kwargs),
        func_name="llm_request",
        report_func=state.chat_cache.report.llm,
        chat_cache=state.chat_cache,
        span_attributes={
            "model.name": request_kwargs.get("model", ""),
            "byteai.route.tier": getattr(state.context.get("_byte_model_route"), "tier", ""),
        },
    )


def _assess_response_sync(state: PipelineRunState) -> None:
    state.llm_data, state.response_assessment = _assess_and_repair_response(
        state.chat_cache,
        state.kwargs,
        state.llm_data,
        context=state.context,
        task_policy=state.context.get("_byte_task_policy") or {},
    )
    state.route_decision = state.context.get("_byte_model_route")
    state.task_policy = state.context.get("_byte_task_policy") or {}


def _run_consensus_and_verifier_sync(state: PipelineRunState) -> None:
    state.llm_data, state.response_assessment = _run_cheap_consensus_sync(
        state.chat_cache,
        state.llm_handler,
        state.args,
        state.kwargs,
        state.context,
        state.llm_data,
        state.response_assessment,
        task_policy=state.task_policy,
    )
    state.llm_data, state.response_assessment = _run_verifier_model_sync(
        state.chat_cache,
        state.llm_handler,
        state.args,
        state.kwargs,
        state.context,
        state.llm_data,
        state.response_assessment,
    )


async def _run_consensus_and_verifier_async(state: PipelineRunState) -> None:
    state.llm_data, state.response_assessment = await _run_cheap_consensus_async(
        state.chat_cache,
        state.llm_handler,
        state.args,
        state.kwargs,
        state.context,
        state.llm_data,
        state.response_assessment,
        task_policy=state.task_policy,
    )
    state.llm_data, state.response_assessment = await _run_verifier_model_async(
        state.chat_cache,
        state.llm_handler,
        state.args,
        state.kwargs,
        state.context,
        state.llm_data,
        state.response_assessment,
    )


def _run_escalation_sync(state: PipelineRunState) -> None:
    escalation_attempts = 0
    while _should_continue_escalation(state) and escalation_attempts < 2:
        escalated_model, escalated_tier = _resolve_escalation_target(
            state.chat_cache,
            state.route_decision,
            state.kwargs,
        )
        if not escalated_model:
            break
        escalation_reason = _escalation_reason(state)
        _prepare_escalation(state, escalated_tier, escalation_reason)
        escalated_kwargs = dict(state.kwargs)
        escalated_kwargs["model"] = escalated_model

        def run_escalated_request(
            request_kwargs: dict[str, Any] = escalated_kwargs,
        ) -> Any:
            return _provider_request_sync(state, request_kwargs)

        state.llm_data = with_coalescer_guard(
            state,
            run_escalated_request,
            stage="provider_escalation",
        )
        state.kwargs["model"] = escalated_model
        state.context["_byte_model_route"] = _make_escalated_decision(
            state.route_decision,
            escalated_model,
            escalated_tier,
            escalation_reason,
        )
        state.route_decision = state.context.get("_byte_model_route")
        _try_record_budget(state.llm_data, escalated_model, chat_cache=state.chat_cache)
        _assess_response_sync(state)
        state.llm_data, state.response_assessment = _run_verifier_model_sync(
            state.chat_cache,
            state.llm_handler,
            state.args,
            state.kwargs,
            state.context,
            state.llm_data,
            state.response_assessment,
        )
        escalation_attempts += 1


async def _run_escalation_async(state: PipelineRunState) -> None:
    escalation_attempts = 0
    while _should_continue_escalation(state) and escalation_attempts < 2:
        escalated_model, escalated_tier = _resolve_escalation_target(
            state.chat_cache,
            state.route_decision,
            state.kwargs,
        )
        if not escalated_model:
            break
        escalation_reason = _escalation_reason(state)
        _prepare_escalation(state, escalated_tier, escalation_reason)
        escalated_kwargs = dict(state.kwargs)
        escalated_kwargs["model"] = escalated_model

        async def run_escalated_request(
            request_kwargs: dict[str, Any] = escalated_kwargs,
        ) -> Any:
            return await _provider_request_async(state, request_kwargs)

        state.llm_data = await awith_coalescer_guard(
            state,
            run_escalated_request,
            stage="provider_escalation",
        )
        state.kwargs["model"] = escalated_model
        state.context["_byte_model_route"] = _make_escalated_decision(
            state.route_decision,
            escalated_model,
            escalated_tier,
            escalation_reason,
        )
        state.route_decision = state.context.get("_byte_model_route")
        _try_record_budget(state.llm_data, escalated_model, chat_cache=state.chat_cache)
        _assess_response_sync(state)
        state.llm_data, state.response_assessment = await _run_verifier_model_async(
            state.chat_cache,
            state.llm_handler,
            state.args,
            state.kwargs,
            state.context,
            state.llm_data,
            state.response_assessment,
        )
        escalation_attempts += 1


def _should_continue_escalation(state: PipelineRunState) -> bool:
    return bool(
        _should_escalate_routed_response(
        state.chat_cache,
        state.route_decision,
        state.response_assessment,
        state.context,
        task_policy=state.task_policy,
        )
    )


def _escalation_reason(state: PipelineRunState) -> str:
    return str(
        getattr(state.response_assessment, "reason", "")
        or f"{state.route_decision.tier}_response_rejected"
    )


def _prepare_escalation(state: PipelineRunState, escalated_tier: str, escalation_reason: str) -> None:
    if "_byte_counterfactual" not in state.context:
        state.context["_byte_counterfactual"] = {
            "action": _escalation_action_for_tier(escalated_tier),
            "reason": escalation_reason,
        }
    record_route_outcome(
        state.route_decision,
        accepted=False,
        latency_ms=(time.time() - state.start_time) * 1000,
    )
    get_quality_scorer(state.chat_cache).record_escalation()
    _record_failure_memory(
        state.chat_cache,
        state.kwargs,
        state.context,
        reason=escalation_reason,
        llm_data=state.llm_data,
    )


__all__ = ["execute_provider_stage_async", "execute_provider_stage_sync"]
