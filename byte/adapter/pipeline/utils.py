import asyncio
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from byte import cache
from byte.adapter.runtime_state import (
    get_budget_tracker as _runtime_budget_tracker,
)
from byte.adapter.runtime_state import (
    get_quality_scorer as _runtime_quality_scorer,
)
from byte.processor.cache_latency import record_cache_stage_outcome, should_bypass_cache_stage
from byte.processor.intent import extract_request_intent
from byte.processor.policy import record_policy_event
from byte.processor.reasoning_reuse import (
    resolve_reasoning_shortcut,
)
from byte.security import redact_text
from byte.similarity_evaluation.exact_match import ExactMatchEvaluation
from byte.telemetry import get_log_time_func, telemetry_stage_span
from byte.utils.async_ops import run_sync
from byte.utils.log import byte_log

from .context import _repo_fingerprint_from_context

_input_summarizer = None
_T = TypeVar("_T")


def _best_effort(default: _T, operation: Callable[[], _T], *, log_message: str | None = None) -> _T:
    """Run cache-adjacent bookkeeping as a best-effort boundary."""

    try:
        return operation()
    except Exception:  # best-effort cache boundary
        if log_message:
            byte_log.debug(log_message, exc_info=True)
        return default


def get_budget_tracker(chat_cache=None) -> Any:
    """Return the budget tracker bound to a cache instance."""
    return _runtime_budget_tracker(chat_cache)


def get_quality_scorer(chat_cache=None) -> Any:
    """Return the quality scorer bound to a cache instance."""
    return _runtime_quality_scorer(chat_cache)


async def _time_cal_async(
    func,
    *args,
    func_name=None,
    report_func=None,
    chat_cache=None,
    span_attributes=None,
    **kwargs,
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
        res = await run_sync(func, *args, **kwargs)
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
    return res


async def _await_with_report(
    awaitable,
    *,
    func_name,
    report_func=None,
    chat_cache=None,
    span_attributes=None,
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
        res = await awaitable
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
    return res


def _log_background_task_result(task: asyncio.Task) -> None:
    """Surface background task failures instead of dropping them silently."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:  # pylint: disable=W0703
        byte_log.error("background cache task failed", exc_info=True)


def _try_record_budget(llm_data, model_name: str, chat_cache=None) -> None:
    """Best-effort extraction of token usage from an LLM response."""
    def record_budget() -> None:
        usage = None
        if hasattr(llm_data, "usage") and llm_data.usage:
            usage = llm_data.usage
        elif isinstance(llm_data, dict) and "usage" in llm_data:
            usage = llm_data["usage"]

        if usage is None:
            return

        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
        else:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0

        get_budget_tracker(chat_cache).record_usage(
            model_name,
            prompt_tokens=pt,
            completion_tokens=ct,
        )

    _best_effort(None, record_budget)


def _build_synthetic_response(request_kwargs, content, *, byte_reason, model=None) -> dict[str, Any]:
    model_name = model or str(request_kwargs.get("model", "") or "")
    return {
        "byte": True,
        "byte_reason": byte_reason,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "created": int(time.time()),
        "model": model_name,
        "object": "chat.completion",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _safe_log_value(chat_cache, value) -> Any:
    if getattr(chat_cache.config, "security_redact_logs", False):
        return redact_text(value)
    return value


def _extract_llm_answer(llm_data) -> Any | None:
    if not isinstance(llm_data, dict):
        return None
    choices = llm_data.get("choices") or []
    if choices:
        first_choice = choices[0] or {}
        message = first_choice.get("message", {}) or {}
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "") or item.get("content", "") or ""))
                else:
                    parts.append(str(item or ""))
            joined = "".join(parts).strip()
            if joined:
                return joined
        if content is not None:
            return content
        if first_choice.get("text") is not None:
            return first_choice.get("text")
    if llm_data.get("text") is not None:
        return llm_data.get("text")
    return None


def _extract_llm_reasoning(llm_data) -> Any | None:
    if not isinstance(llm_data, dict):
        return None
    choices = llm_data.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    for value in (
        llm_data.get("reasoning"),
        llm_data.get("summary"),
        first_choice.get("reasoning") if isinstance(first_choice, dict) else None,
        message.get("reasoning"),
        message.get("reasoning_content"),
        message.get("thinking"),
    ):
        if value not in (None, "", [], {}):
            return value
    return None


def _extract_llm_tool_outputs(llm_data) -> Any | None:
    if not isinstance(llm_data, dict):
        return None
    choices = llm_data.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    for value in (
        message.get("tool_outputs"),
        message.get("tool_calls"),
        first_choice.get("tool_outputs") if isinstance(first_choice, dict) else None,
        llm_data.get("tool_outputs"),
        llm_data.get("tool_calls"),
    ):
        if value not in (None, "", [], {}):
            return value
    return None


def _set_llm_answer(llm_data, answer_text: str) -> Any:
    if not isinstance(llm_data, dict):
        return llm_data
    choices = llm_data.get("choices") or []
    if choices:
        first_choice = choices[0] or {}
        message = first_choice.get("message")
        if isinstance(message, dict):
            message["content"] = answer_text
        elif first_choice.get("text") is not None:
            first_choice["text"] = answer_text
        else:
            first_choice["message"] = {"role": "assistant", "content": answer_text}
        return llm_data
    if llm_data.get("text") is not None:
        llm_data["text"] = answer_text
    return llm_data


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _reuse_policy_payload(context) -> Any:
    policy = context.get("_byte_reuse_policy", {}) or {}
    if hasattr(policy, "to_dict"):
        return policy.to_dict()
    if isinstance(policy, dict):
        return policy
    return {}


def _cache_lookup_allowed(context) -> bool:
    policy = _reuse_policy_payload(context)
    return str(policy.get("mode", "full_reuse") or "full_reuse") == "full_reuse"


def _cache_save_allowed(context) -> bool:
    policy = _reuse_policy_payload(context)
    return str(policy.get("mode", "full_reuse") or "full_reuse") == "full_reuse"


def _cache_stage_name(chat_cache) -> str:
    stage = str(getattr(chat_cache, "byte_cache_stage", "") or "").strip().lower()
    if stage:
        return stage
    if isinstance(getattr(chat_cache, "similarity_evaluation", None), ExactMatchEvaluation):
        pre_name = str(
            getattr(getattr(chat_cache, "pre_embedding_func", None), "__name__", "") or ""
        ).lower()
        return "normalized" if "normalized" in pre_name else "exact"
    return "semantic"


def _route_key_for_request(request_kwargs, context) -> str:
    route_decision = context.get("_byte_model_route")
    route_key = str(getattr(route_decision, "route_key", "") or "")
    if route_key:
        return route_key
    return str(extract_request_intent(request_kwargs).route_key or "")


def _record_cache_stage_latency(
    chat_cache, request_kwargs, context, *, started_at, hit: bool
) -> None:
    if started_at in (None, 0):
        return
    route_key = _route_key_for_request(request_kwargs, context)
    stage = _cache_stage_name(chat_cache)
    if not route_key or not stage:
        return
    latency_ms = max(0.0, (time.time() - float(started_at)) * 1000.0)
    record_cache_stage_outcome(route_key, stage, latency_ms=latency_ms, hit=hit)


def _should_bypass_current_cache_stage(chat_cache, request_kwargs, context) -> bool:
    route_key = _route_key_for_request(request_kwargs, context)
    stage = _cache_stage_name(chat_cache)
    if not route_key or not stage:
        return False
    return should_bypass_cache_stage(route_key, stage, chat_cache.config)


def _admission_allowed(chat_cache, assessment, task_policy=None) -> bool:
    if assessment is None:
        return False
    if assessment.constraint != "freeform" and not assessment.accepted:
        return False
    min_score = float(
        (task_policy or {}).get("cache_admission_min_score")
        or getattr(chat_cache.config, "cache_admission_min_score", 0.0)
        or 0.0
    )
    return assessment.score >= min_score


def _requires_verified_reuse(chat_cache, request_kwargs) -> bool:
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


def _cache_reuse_allowed(chat_cache, request_kwargs, context, answer) -> bool:
    if answer in (None, ""):
        return True
    if not _cache_lookup_allowed(context):
        return False
    memory_context = context.get("_byte_memory", {}) or {}
    provider = str(memory_context.get("provider") or memory_context.get("byte_provider") or "")
    model_name = str(request_kwargs.get("model", "") or "")
    if getattr(chat_cache.config, "failure_memory", True):
        def failure_memory_hint() -> Any:
            failure_hint = chat_cache.failure_memory_hint(
                request_kwargs,
                provider=provider,
                model=model_name,
            )
            if failure_hint.get("avoid_cache_reuse"):
                return False
            return True

        if not _best_effort(True, failure_memory_hint):
            return False
    if not getattr(chat_cache.config, "execution_memory", True):
        return True
    if not _requires_verified_reuse(chat_cache, request_kwargs):
        return True
    def lookup_execution_result() -> Any:
        verification = chat_cache.lookup_execution_result(
            request_kwargs,
            answer=answer,
            repo_fingerprint=_repo_fingerprint_from_context(context),
            model=model_name,
            verified_only=True,
        )
        return verification is not None

    return _best_effort(False, lookup_execution_result)


def _return_reasoning_shortcut(chat_cache, request_kwargs, context, shortcut) -> Any:
    from .memory import (  # local import to avoid pipeline import cycles
        _record_ai_memory,
        _record_reasoning_memory,
        _record_workflow_outcome,
    )

    synthetic_response = _build_synthetic_response(
        request_kwargs,
        shortcut.answer,
        byte_reason=shortcut.byte_reason,
    )
    synthetic_response["byte_reasoning"] = shortcut.to_dict()
    context["_byte_reasoning_shortcut"] = shortcut.to_dict()
    if getattr(chat_cache.config, "intent_memory", False):
        _best_effort(None, lambda: chat_cache.record_intent(request_kwargs))
    _record_ai_memory(
        chat_cache,
        request_kwargs,
        context=context,
        answer=shortcut.answer,
        embedding_data=None,
        llm_data=synthetic_response,
        source="reasoning_shortcut",
    )
    _record_reasoning_memory(
        chat_cache,
        request_kwargs,
        answer=shortcut.answer,
        verified=True,
        source=shortcut.source,
    )
    def record_shortcut_policy_event() -> None:
        intent = extract_request_intent(request_kwargs)
        record_policy_event(
            intent.route_key,
            category=intent.category,
            event=shortcut.byte_reason,
        )

    _best_effort(None, record_shortcut_policy_event)
    _record_workflow_outcome(
        chat_cache, request_kwargs, context, success=True, reason=shortcut.reason
    )
    return synthetic_response


def _maybe_reasoning_shortcut(chat_cache, request_kwargs, context) -> Any | None:
    if request_kwargs.get("byte_disable_reasoning_shortcut"):
        return None
    if not getattr(chat_cache.config, "reasoning_reuse", True):
        return None
    def resolve_shortcut() -> Any:
        shortcut = resolve_reasoning_shortcut(
            request_kwargs,
            store=getattr(chat_cache, "reasoning_memory_store", None),
            config=chat_cache.config,
            context_hints=context,
        )
        return shortcut

    shortcut = _best_effort(None, resolve_shortcut)
    if shortcut is None:
        return None
    return _return_reasoning_shortcut(chat_cache, request_kwargs, context, shortcut)


def cache_health_check(vectordb, cache_dict) -> Any:
    """This function checks if the embedding
    from vector store matches one in cache store.
    If cache store and vector store are out of
    sync with each other, cache retrieval can
    be incorrect.
    If this happens, force the similary score
    to the lowerest possible value.
    """
    emb_in_cache = cache_dict["embedding"]
    _, data_id = cache_dict["search_result"]
    emb_in_vec = vectordb.get_embeddings(data_id)
    flag = np.all(emb_in_cache == emb_in_vec)
    if not flag:
        byte_log.critical("Cache Store and Vector Store are out of sync!!!")
        # 0: identical, inf: different
        cache_dict["search_result"] = (
            np.inf,
            data_id,
        )
        # self-healing by replacing entry
        # in the vec store with the one
        # from cache store by the same
        # entry_id.
        vectordb.update_embeddings(
            data_id,
            emb=cache_dict["embedding"],
        )
    return flag


async def acache_health_check(vectordb, cache_dict) -> Any:
    """Async version of ``cache_health_check`` for async adapter flow."""
    emb_in_cache = cache_dict["embedding"]
    _, data_id = cache_dict["search_result"]
    if hasattr(vectordb, "aget_embeddings"):
        emb_in_vec = await vectordb.aget_embeddings(data_id)
    else:
        emb_in_vec = await run_sync(vectordb.get_embeddings, data_id)
    flag = np.all(emb_in_cache == emb_in_vec)
    if not flag:
        byte_log.critical("Cache Store and Vector Store are out of sync!!!")
        cache_dict["search_result"] = (
            np.inf,
            data_id,
        )
        if hasattr(vectordb, "aupdate_embeddings"):
            await vectordb.aupdate_embeddings(
                data_id,
                emb=cache_dict["embedding"],
            )
        else:
            await run_sync(
                vectordb.update_embeddings,
                data_id,
                emb=cache_dict["embedding"],
            )
    return flag


def _summarize_input(text, text_length) -> Any:
    if len(text) <= text_length:
        return text

    # pylint: disable=import-outside-toplevel
    from byte.processor.context.summarization_context import (
        SummarizationContextProcess,
    )

    global _input_summarizer
    def summarize_text() -> Any:
        global _input_summarizer
        if _input_summarizer is None:
            _input_summarizer = SummarizationContextProcess()
        return _input_summarizer.summarize_to_sentence([text], text_length)

    summarized = _best_effort(None, summarize_text, log_message="input summarizer unavailable")
    if summarized is not None:
        return summarized
    # Fall back to a cheap truncation path when the summarizer backend
    # is unavailable. This keeps input shortening predictable in CI and
    # avoids adding cold-start latency to normal cache usage.
    token_budget = max(int(text_length) - 1, 1)
    return " ".join(str(text).split()[:token_budget])


def _build_coalesced_retry_kwargs(
    kwargs,
    *,
    chat_cache,
    context,
    cache_factor,
    session,
    require_object_store,
    user_temperature,
    temperature,
    requested_top_k,
) -> Any:
    retry_kwargs = dict(kwargs)
    retry_kwargs["cache_obj"] = chat_cache
    retry_kwargs["cache_context"] = context
    retry_kwargs["cache_skip"] = False
    retry_kwargs["cache_factor"] = cache_factor
    retry_kwargs["search_only"] = True
    if session is not None:
        retry_kwargs["session"] = session
    if require_object_store:
        retry_kwargs["require_object_store"] = True
    if user_temperature:
        retry_kwargs["temperature"] = temperature
    if requested_top_k is not None:
        retry_kwargs["top_k"] = requested_top_k
    return retry_kwargs


__all__ = [name for name in globals() if not name.startswith("__")]
