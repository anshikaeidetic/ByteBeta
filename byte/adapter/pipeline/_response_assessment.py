
"""Response assessment and escalation paths for cache misses."""

from __future__ import annotations

import time
from typing import Any

from byte.processor.model_router import record_route_outcome
from byte.utils.time import time_cal

from ._response_commit import _cancel_coalescer
from ._runtime_support import _await_with_report, _try_record_budget, get_quality_scorer
from .consensus import _run_cheap_consensus_async, _run_cheap_consensus_sync
from .context import _provider_request_kwargs
from .escalation import (
    _escalation_action_for_tier,
    _make_escalated_decision,
    _resolve_escalation_target,
    _should_escalate_routed_response,
)
from .memory import _record_failure_memory
from .verifier import (
    _assess_and_repair_response,
    _run_verifier_model_async,
    _run_verifier_model_sync,
)


def run_sync_response_assessment(
    *,
    chat_cache: Any,
    llm_handler: Any,
    args: tuple[Any, ...],
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    llm_data: Any,
    start_time: float,
    coalesce_key: str | None,
) -> tuple[Any, Any]:
    """Assess, verify, and escalate an upstream sync response when needed."""
    model_name = request_kwargs.get("model", "unknown")
    _try_record_budget(llm_data, model_name, chat_cache=chat_cache)
    llm_data, response_assessment = _assess_and_repair_response(
        chat_cache,
        request_kwargs,
        llm_data,
        context=context,
        task_policy=context.get("_byte_task_policy") or {},
    )
    route_decision = context.get("_byte_model_route")
    task_policy = context.get("_byte_task_policy") or {}
    llm_data, response_assessment = _run_cheap_consensus_sync(
        chat_cache,
        llm_handler,
        args,
        request_kwargs,
        context,
        llm_data,
        response_assessment,
        task_policy=task_policy,
    )
    llm_data, response_assessment = _run_verifier_model_sync(
        chat_cache,
        llm_handler,
        args,
        request_kwargs,
        context,
        llm_data,
        response_assessment,
    )
    escalation_attempts = 0
    while (
        _should_escalate_routed_response(
            chat_cache,
            route_decision,
            response_assessment,
            context,
            task_policy=task_policy,
        )
        and escalation_attempts < 2
    ):
        escalated_model, escalated_tier = _resolve_escalation_target(
            chat_cache, route_decision, request_kwargs
        )
        if not escalated_model:
            break
        escalation_reason = str(
            getattr(response_assessment, "reason", "")
            or f"{route_decision.tier}_response_rejected"
        )
        if "_byte_counterfactual" not in context:
            context["_byte_counterfactual"] = {
                "action": _escalation_action_for_tier(escalated_tier),
                "reason": escalation_reason,
            }
        record_route_outcome(
            route_decision,
            accepted=False,
            latency_ms=(time.time() - start_time) * 1000,
        )
        get_quality_scorer(chat_cache).record_escalation()
        _record_failure_memory(
            chat_cache, request_kwargs, context, reason=escalation_reason, llm_data=llm_data
        )
        escalated_kwargs = dict(request_kwargs)
        escalated_kwargs["model"] = escalated_model
        try:
            provider_escalated_kwargs = _provider_request_kwargs(
                chat_cache, escalated_kwargs, context
            )
            llm_data = time_cal(
                llm_handler,
                func_name="llm_request",
                report_func=chat_cache.report.llm,
            )(*args, **provider_escalated_kwargs)
            request_kwargs["model"] = escalated_model
            context["_byte_model_route"] = _make_escalated_decision(
                route_decision,
                escalated_model,
                escalated_tier,
                escalation_reason,
            )
            route_decision = context.get("_byte_model_route")
            _try_record_budget(llm_data, escalated_model, chat_cache=chat_cache)
            llm_data, response_assessment = _assess_and_repair_response(
                chat_cache,
                request_kwargs,
                llm_data,
                context=context,
                task_policy=context.get("_byte_task_policy") or {},
            )
            llm_data, response_assessment = _run_verifier_model_sync(
                chat_cache,
                llm_handler,
                args,
                request_kwargs,
                context,
                llm_data,
                response_assessment,
            )
        except Exception:
            _cancel_coalescer(chat_cache, coalesce_key)
            raise
        escalation_attempts += 1
    return llm_data, response_assessment


async def run_async_response_assessment(
    *,
    chat_cache: Any,
    llm_handler: Any,
    args: tuple[Any, ...],
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    llm_data: Any,
    start_time: float,
    coalesce_key: str | None,
) -> tuple[Any, Any]:
    """Assess, verify, and escalate an upstream async response when needed."""
    model_name = request_kwargs.get("model", "unknown")
    _try_record_budget(llm_data, model_name, chat_cache=chat_cache)
    llm_data, response_assessment = _assess_and_repair_response(
        chat_cache,
        request_kwargs,
        llm_data,
        context=context,
        task_policy=context.get("_byte_task_policy") or {},
    )
    route_decision = context.get("_byte_model_route")
    task_policy = context.get("_byte_task_policy") or {}
    llm_data, response_assessment = await _run_cheap_consensus_async(
        chat_cache,
        llm_handler,
        args,
        request_kwargs,
        context,
        llm_data,
        response_assessment,
        task_policy=task_policy,
    )
    llm_data, response_assessment = await _run_verifier_model_async(
        chat_cache,
        llm_handler,
        args,
        request_kwargs,
        context,
        llm_data,
        response_assessment,
    )
    escalation_attempts = 0
    while (
        _should_escalate_routed_response(
            chat_cache,
            route_decision,
            response_assessment,
            context,
            task_policy=task_policy,
        )
        and escalation_attempts < 2
    ):
        escalated_model, escalated_tier = _resolve_escalation_target(
            chat_cache, route_decision, request_kwargs
        )
        if not escalated_model:
            break
        escalation_reason = str(
            getattr(response_assessment, "reason", "")
            or f"{route_decision.tier}_response_rejected"
        )
        if "_byte_counterfactual" not in context:
            context["_byte_counterfactual"] = {
                "action": _escalation_action_for_tier(escalated_tier),
                "reason": escalation_reason,
            }
        record_route_outcome(
            route_decision,
            accepted=False,
            latency_ms=(time.time() - start_time) * 1000,
        )
        get_quality_scorer(chat_cache).record_escalation()
        _record_failure_memory(
            chat_cache, request_kwargs, context, reason=escalation_reason, llm_data=llm_data
        )
        escalated_kwargs = dict(request_kwargs)
        escalated_kwargs["model"] = escalated_model
        try:
            provider_escalated_kwargs = _provider_request_kwargs(
                chat_cache, escalated_kwargs, context
            )
            llm_data = await _await_with_report(
                llm_handler(*args, **provider_escalated_kwargs),
                func_name="llm_request",
                report_func=chat_cache.report.llm,
                chat_cache=chat_cache,
                span_attributes={
                    "model.name": escalated_model,
                    "byteai.route.tier": escalated_tier,
                },
            )
            request_kwargs["model"] = escalated_model
            context["_byte_model_route"] = _make_escalated_decision(
                route_decision,
                escalated_model,
                escalated_tier,
                escalation_reason,
            )
            route_decision = context.get("_byte_model_route")
            _try_record_budget(llm_data, escalated_model, chat_cache=chat_cache)
            llm_data, response_assessment = _assess_and_repair_response(
                chat_cache,
                request_kwargs,
                llm_data,
                context=context,
                task_policy=context.get("_byte_task_policy") or {},
            )
            llm_data, response_assessment = await _run_verifier_model_async(
                chat_cache,
                llm_handler,
                args,
                request_kwargs,
                context,
                llm_data,
                response_assessment,
            )
        except Exception:
            _cancel_coalescer(chat_cache, coalesce_key)
            raise
        escalation_attempts += 1
    return llm_data, response_assessment

__all__ = ["run_async_response_assessment", "run_sync_response_assessment"]
