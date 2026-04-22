"""Runtime support helpers shared by sync and async pipeline flows."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from byte import cache
from byte.adapter.runtime_state import (
    get_budget_tracker as _runtime_budget_tracker,
)
from byte.adapter.runtime_state import (
    get_quality_scorer as _runtime_quality_scorer,
)
from byte.processor.cache_latency import record_cache_stage_outcome, should_bypass_cache_stage
from byte.processor.intent import extract_request_intent
from byte.telemetry import get_log_time_func, telemetry_stage_span
from byte.utils.async_ops import run_sync
from byte.utils.log import byte_log

from .context import _repo_fingerprint_from_context


def get_budget_tracker(chat_cache: Any = None) -> Any:
    """Return the budget tracker bound to a cache instance."""
    return _runtime_budget_tracker(chat_cache)


def get_quality_scorer(chat_cache: Any = None) -> Any:
    """Return the quality scorer bound to a cache instance."""
    return _runtime_quality_scorer(chat_cache)


async def _time_cal_async(
    func: Any,
    *args: Any,
    func_name: str | None = None,
    report_func: Any = None,
    chat_cache: Any = None,
    span_attributes: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Async equivalent of ``time_cal`` for sync callables."""
    operation_name = func.__name__ if func_name is None else func_name
    time_start = time.time()
    with telemetry_stage_span(
        operation_name,
        chat_cache=chat_cache,
        report_func=report_func,
        fallback_cache=cache,
        attributes=span_attributes,
    ):
        result = await run_sync(func, *args, **kwargs)
    delta_time = time.time() - time_start
    log_time_func = get_log_time_func(
        chat_cache=chat_cache,
        report_func=report_func,
        fallback_cache=cache,
    )
    if log_time_func:
        log_time_func(operation_name, delta_time)
    if report_func is not None:
        report_func(delta_time)
    return result


async def _await_with_report(
    awaitable: Any,
    *,
    func_name: str,
    report_func: Any = None,
    chat_cache: Any = None,
    span_attributes: dict[str, Any] | None = None,
) -> Any:
    """Measure and report an awaited operation."""
    time_start = time.time()
    with telemetry_stage_span(
        func_name,
        chat_cache=chat_cache,
        report_func=report_func,
        fallback_cache=cache,
        attributes=span_attributes,
    ):
        result = await awaitable
    delta_time = time.time() - time_start
    log_time_func = get_log_time_func(
        chat_cache=chat_cache,
        report_func=report_func,
        fallback_cache=cache,
    )
    if log_time_func:
        log_time_func(func_name, delta_time)
    if report_func is not None:
        report_func(delta_time)
    return result


def _log_background_task_result(task: asyncio.Task[Any]) -> None:
    """Surface background task failures instead of dropping them silently."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:  # pylint: disable=W0703
        byte_log.error("background cache task failed", exc_info=True)


def _try_record_budget(llm_data: Any, model_name: str, chat_cache: Any = None) -> None:
    """Best-effort extraction of token usage from an LLM response."""
    try:
        usage = None
        if hasattr(llm_data, "usage") and llm_data.usage:
            usage = llm_data.usage
        elif isinstance(llm_data, dict) and "usage" in llm_data:
            usage = llm_data["usage"]

        if usage is None:
            return

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        get_budget_tracker(chat_cache).record_usage(
            model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        byte_log.debug("Skipping budget usage extraction for %s: %s", model_name, exc)


def _reuse_policy_payload(context: dict[str, Any]) -> dict[str, Any]:
    policy = context.get("_byte_reuse_policy", {}) or {}
    if hasattr(policy, "to_dict"):
        return policy.to_dict()
    if isinstance(policy, dict):
        return policy
    return {}


def _cache_lookup_allowed(context: dict[str, Any]) -> bool:
    policy = _reuse_policy_payload(context)
    return str(policy.get("mode", "full_reuse") or "full_reuse") == "full_reuse"


def _cache_save_allowed(context: dict[str, Any]) -> bool:
    policy = _reuse_policy_payload(context)
    return str(policy.get("mode", "full_reuse") or "full_reuse") == "full_reuse"


def _cache_stage_name(chat_cache: Any) -> str:
    stage = str(getattr(chat_cache, "byte_cache_stage", "") or "").strip().lower()
    if stage:
        return stage
    similarity = getattr(chat_cache, "similarity_evaluation", None)
    similarity_name = type(similarity).__name__.lower() if similarity is not None else ""
    if similarity_name == "exactmatchevaluation":
        pre_name = str(
            getattr(getattr(chat_cache, "pre_embedding_func", None), "__name__", "") or ""
        ).lower()
        return "normalized" if "normalized" in pre_name else "exact"
    return "semantic"


def _route_key_for_request(request_kwargs: dict[str, Any], context: dict[str, Any]) -> str:
    route_decision = context.get("_byte_model_route")
    route_key = str(getattr(route_decision, "route_key", "") or "")
    if route_key:
        return route_key
    return str(extract_request_intent(request_kwargs).route_key or "")


def _record_cache_stage_latency(
    chat_cache: Any,
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    *,
    started_at: float | None,
    hit: bool,
) -> None:
    if started_at in (None, 0):
        return
    route_key = _route_key_for_request(request_kwargs, context)
    stage = _cache_stage_name(chat_cache)
    if not route_key or not stage:
        return
    latency_ms = max(0.0, (time.time() - float(started_at)) * 1000.0)
    record_cache_stage_outcome(route_key, stage, latency_ms=latency_ms, hit=hit)


def _should_bypass_current_cache_stage(
    chat_cache: Any, request_kwargs: dict[str, Any], context: dict[str, Any]
) -> bool:
    route_key = _route_key_for_request(request_kwargs, context)
    stage = _cache_stage_name(chat_cache)
    if not route_key or not stage:
        return False
    return should_bypass_cache_stage(route_key, stage, chat_cache.config)


def _admission_allowed(chat_cache: Any, assessment: Any, task_policy: dict[str, Any] | None = None) -> bool:
    if assessment is None:
        return True
    if assessment.constraint != "freeform" and not assessment.accepted:
        return False
    min_score = float(
        (task_policy or {}).get("cache_admission_min_score")
        or getattr(chat_cache.config, "cache_admission_min_score", 0.0)
        or 0.0
    )
    return assessment.score >= min_score


def _answer_has_content(handled_llm_data: Any) -> bool:
    if handled_llm_data is None:
        return False
    text = getattr(handled_llm_data, "answer", handled_llm_data)
    if text is None:
        return False
    try:
        return bool(str(text).strip())
    except Exception:
        return False


def _requires_verified_reuse(chat_cache: Any, request_kwargs: dict[str, Any]) -> bool:
    intent = extract_request_intent(request_kwargs)
    if getattr(chat_cache.config, "verified_reuse_for_all", False):
        return True
    return getattr(chat_cache.config, "verified_reuse_for_coding", True) and intent.category in {
        "code_fix",
        "code_refactor",
        "test_generation",
        "code_explanation",
        "documentation",
    }


def _cache_reuse_allowed(
    chat_cache: Any,
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    answer: Any,
) -> bool:
    if answer in (None, ""):
        return True
    if not _cache_lookup_allowed(context):
        return False
    memory_context = context.get("_byte_memory", {}) or {}
    provider = str(memory_context.get("provider") or memory_context.get("byte_provider") or "")
    model_name = str(request_kwargs.get("model", "") or "")
    if getattr(chat_cache.config, "failure_memory", True):
        try:
            failure_hint = chat_cache.failure_memory_hint(
                request_kwargs,
                provider=provider,
                model=model_name,
            )
            if failure_hint.get("avoid_cache_reuse"):
                return False
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            byte_log.debug("Skipping failure-memory reuse hint lookup: %s", exc)
    if not getattr(chat_cache.config, "execution_memory", True):
        return True
    if not _requires_verified_reuse(chat_cache, request_kwargs):
        return True
    try:
        verification = chat_cache.lookup_execution_result(
            request_kwargs,
            answer=answer,
            repo_fingerprint=_repo_fingerprint_from_context(context),
            model=model_name,
            verified_only=True,
        )
        return verification is not None
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        byte_log.debug("Skipping execution-memory reuse lookup: %s", exc)
        return False


__all__ = [
    "_admission_allowed",
    "_answer_has_content",
    "_await_with_report",
    "_cache_lookup_allowed",
    "_cache_reuse_allowed",
    "_cache_save_allowed",
    "_cache_stage_name",
    "_log_background_task_result",
    "_record_cache_stage_latency",
    "_should_bypass_current_cache_stage",
    "_time_cal_async",
    "_try_record_budget",
    "get_budget_tracker",
    "get_quality_scorer",
]
