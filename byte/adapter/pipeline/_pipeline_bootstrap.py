"""Shared request bootstrap and cache-input preparation for the adapter pipeline."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from byte import cache
from byte.adapter.runtime_state import get_coalescer
from byte.processor.coalesce import _SENTINEL
from byte.processor.intent import extract_request_intent
from byte.processor.policy import record_policy_event
from byte.processor.post import temperature_softmax
from byte.processor.quality import extract_output_contract
from byte.processor.reuse_policy import detect_reuse_policy
from byte.security import sanitize_outbound_overrides
from byte.utils.error import NotInitError
from byte.utils.log import byte_log
from byte.utils.time import time_cal

from ._pipeline_common import NO_RESULT, build_recursive_kwargs
from ._pipeline_state import PipelineRunState
from .context import (
    _apply_request_namespaces,
    _compile_context_if_needed,
    _maybe_route_request,
    _plan_workflow,
    _resolve_memory_context,
    _semantic_cache_allowed,
)
from .memory import (
    _aembed_request,
    _embed_request,
    _make_coalesce_key,
    _record_failure_memory,
    _record_workflow_outcome,
)
from .utils import (
    _build_coalesced_retry_kwargs,
    _build_synthetic_response,
    _cache_lookup_allowed,
    _cache_stage_name,
    _maybe_reasoning_shortcut,
    _should_bypass_current_cache_stage,
    _summarize_input,
    _time_cal_async,
)


def _resolve_cache_skip(*, temperature: float, kwargs: dict[str, Any]) -> bool:
    if 0 < temperature < 2:
        return bool(
            kwargs.pop(
                "cache_skip",
                temperature_softmax(
                    messages=[True, False],
                    scores=[0, 1],
                    temperature=temperature,
                ),
            )
        )
    if temperature >= 2:
        return bool(kwargs.pop("cache_skip", True))
    return bool(kwargs.pop("cache_skip", False))


def initialize_run_state(
    llm_handler: Any,
    cache_data_convert: Any,
    update_cache_callback: Any,
    *args: Any,
    event_loop: asyncio.AbstractEventLoop | None = None,
    **kwargs: Any,
) -> tuple[PipelineRunState, Any]:
    """Normalize a pipeline request before cache or provider execution begins."""

    start_time = time.time()
    search_only_flag = kwargs.pop("search_only", False)
    user_temperature = "temperature" in kwargs
    user_top_k = "top_k" in kwargs
    requested_top_k = kwargs.get("top_k") if user_top_k else None
    temperature = kwargs.pop("temperature", 0.0)
    chat_cache = kwargs.pop("cache_obj", cache)
    session = kwargs.pop("session", None)
    require_object_store = kwargs.pop("require_object_store", False)
    if require_object_store and getattr(chat_cache.data_manager, "o", None) is None:
        raise ValueError("Object store is required for adapter.")
    if not chat_cache.has_init:
        raise NotInitError()

    context = kwargs.pop("cache_context", {}) or {}
    _resolve_memory_context(kwargs, context)
    kwargs = _compile_context_if_needed(chat_cache, kwargs, context, session=session)
    context["_byte_request_kwargs"] = dict(kwargs)
    context["_byte_output_contract"] = extract_output_contract(kwargs).to_dict()
    reuse_policy = detect_reuse_policy(kwargs, config=chat_cache.config, context=context)
    context["_byte_reuse_policy"] = reuse_policy.to_dict()
    if reuse_policy.mode != "full_reuse":
        intent = extract_request_intent(kwargs)
        record_policy_event(
            intent.route_key,
            category=intent.category,
            event=(
                "context_only_reuse" if reuse_policy.mode == "context_only" else "direct_only_reuse"
            ),
        )
    kwargs = sanitize_outbound_overrides(kwargs, chat_cache.config)
    workflow_decision = _plan_workflow(chat_cache, kwargs, context)
    if workflow_decision is not None:
        if workflow_decision.action == "clarify":
            _record_failure_memory(chat_cache, kwargs, context, reason="ambiguous_request")
            _record_workflow_outcome(chat_cache, kwargs, context, success=False, reason="clarify")
            return _build_run_state(
                llm_handler,
                cache_data_convert,
                update_cache_callback,
                args=args,
                kwargs=kwargs,
                chat_cache=chat_cache,
                context=context,
                start_time=start_time,
                search_only_flag=search_only_flag,
                user_temperature=user_temperature,
                user_top_k=user_top_k,
                requested_top_k=requested_top_k,
                temperature=temperature,
                session=session,
                require_object_store=require_object_store,
                cache_enable=False,
                cache_skip=False,
                cache_factor=1.0,
                event_loop=event_loop,
            ), _build_synthetic_response(
                kwargs,
                workflow_decision.response_text,
                byte_reason="clarification_required",
            )
        if workflow_decision.action == "reuse_verified_patch" and workflow_decision.response_text:
            _record_workflow_outcome(
                chat_cache,
                kwargs,
                context,
                success=True,
                reason="reuse_verified_patch",
            )
            return _build_run_state(
                llm_handler,
                cache_data_convert,
                update_cache_callback,
                args=args,
                kwargs=kwargs,
                chat_cache=chat_cache,
                context=context,
                start_time=start_time,
                search_only_flag=search_only_flag,
                user_temperature=user_temperature,
                user_top_k=user_top_k,
                requested_top_k=requested_top_k,
                temperature=temperature,
                session=session,
                require_object_store=require_object_store,
                cache_enable=False,
                cache_skip=False,
                cache_factor=1.0,
                event_loop=event_loop,
            ), _build_synthetic_response(
                kwargs,
                workflow_decision.response_text,
                byte_reason="verified_patch_reuse",
            )

    reasoning_response = _maybe_reasoning_shortcut(chat_cache, kwargs, context)
    if reasoning_response is not None:
        return _build_run_state(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            args=args,
            kwargs=kwargs,
            chat_cache=chat_cache,
            context=context,
            start_time=start_time,
            search_only_flag=search_only_flag,
            user_temperature=user_temperature,
            user_top_k=user_top_k,
            requested_top_k=requested_top_k,
            temperature=temperature,
            session=session,
            require_object_store=require_object_store,
            cache_enable=False,
            cache_skip=False,
            cache_factor=1.0,
            event_loop=event_loop,
        ), reasoning_response

    _maybe_route_request(chat_cache, kwargs, context)
    context["_byte_request_kwargs"] = dict(kwargs)
    cache_enable = chat_cache.cache_enable_func(*args, **kwargs)
    if getattr(chat_cache.config, "intent_memory", False):
        try:
            chat_cache.record_intent(kwargs, session_id=session.name if session else None)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            byte_log.debug("Skipping intent memory capture: %s", exc)

    state = _build_run_state(
        llm_handler,
        cache_data_convert,
        update_cache_callback,
        args=args,
        kwargs=kwargs,
        chat_cache=chat_cache,
        context=context,
        start_time=start_time,
        search_only_flag=search_only_flag,
        user_temperature=user_temperature,
        user_top_k=user_top_k,
        requested_top_k=requested_top_k,
        temperature=temperature,
        session=session,
        require_object_store=require_object_store,
        cache_enable=cache_enable,
        cache_skip=_resolve_cache_skip(temperature=temperature, kwargs=kwargs),
        cache_factor=kwargs.pop("cache_factor", 1.0),
        event_loop=event_loop,
    )
    return state, NO_RESULT


def _build_run_state(
    llm_handler: Any,
    cache_data_convert: Any,
    update_cache_callback: Any,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    chat_cache: Any,
    context: dict[str, Any],
    start_time: float,
    search_only_flag: bool,
    user_temperature: bool,
    user_top_k: bool,
    requested_top_k: Any,
    temperature: float,
    session: Any,
    require_object_store: bool,
    cache_enable: bool,
    cache_skip: bool,
    cache_factor: float,
    event_loop: asyncio.AbstractEventLoop | None,
) -> PipelineRunState:
    return PipelineRunState(
        llm_handler=llm_handler,
        cache_data_convert=cache_data_convert,
        update_cache_callback=update_cache_callback,
        args=tuple(args),
        kwargs=dict(kwargs),
        chat_cache=chat_cache,
        context=context,
        start_time=start_time,
        search_only_flag=search_only_flag,
        user_temperature=user_temperature,
        user_top_k=user_top_k,
        requested_top_k=requested_top_k,
        temperature=temperature,
        session=session,
        require_object_store=require_object_store,
        cache_enable=cache_enable,
        cache_skip=cache_skip,
        cache_factor=cache_factor,
        event_loop=event_loop,
    )


def _apply_intent_filter_if_enabled(state: PipelineRunState) -> None:
    """Apply IntentDrivenContextFilter when intent_context_filtering_enabled=True ."""
    cfg = getattr(state.chat_cache, "config", None)
    if cfg is None or not getattr(cfg, "intent_context_filtering_enabled", False):
        return
    try:
        from byte.processor.intent_context import (
            IntentDrivenContextFilter,  # pylint: disable=import-outside-toplevel
        )
        intent_filter = IntentDrivenContextFilter(
            budget_ratio=getattr(cfg, "intent_context_budget_ratio", 0.6),
            cache_intent_labels=getattr(cfg, "intent_cache_intent_labels", True),
        )
        updated_kwargs, tokens_removed = intent_filter.apply(state.kwargs, state.context)
        if tokens_removed > 0:
            state.kwargs = updated_kwargs
            existing = state.context.get("_byte_intent_tokens_saved", 0)
            state.context["_byte_intent_tokens_saved"] = existing + tokens_removed
    except Exception as exc:  # pylint: disable=W0703
        byte_log.debug("IntentDrivenContextFilter failed (non-fatal): %s", exc)


def prepare_cache_inputs_sync(state: PipelineRunState) -> None:
    """Compute pre-embedding data and gating for sync cache lookup."""

    _apply_intent_filter_if_enabled(state)
    pre_embedding_res = time_cal(
        state.chat_cache.pre_embedding_func,
        func_name="pre_process",
        report_func=state.chat_cache.report.pre,
    )(
        state.kwargs,
        extra_param=state.context.get("pre_embedding_func", None),
        prompts=state.chat_cache.config.prompts,
        cache_config=state.chat_cache.config,
    )
    if isinstance(pre_embedding_res, tuple):
        state.pre_store_data, state.pre_embedding_data = pre_embedding_res
    else:
        state.pre_store_data = pre_embedding_res
        state.pre_embedding_data = pre_embedding_res
    _finalize_cache_inputs(state)


async def prepare_cache_inputs_async(state: PipelineRunState) -> None:
    """Compute pre-embedding data and gating for async cache lookup."""

    _apply_intent_filter_if_enabled(state)
    pre_embedding_res = await _time_cal_async(
        state.chat_cache.pre_embedding_func,
        state.kwargs,
        extra_param=state.context.get("pre_embedding_func", None),
        prompts=state.chat_cache.config.prompts,
        cache_config=state.chat_cache.config,
        func_name="pre_process",
        report_func=state.chat_cache.report.pre,
    )
    if isinstance(pre_embedding_res, tuple):
        state.pre_store_data, state.pre_embedding_data = pre_embedding_res
    else:
        state.pre_store_data = pre_embedding_res
        state.pre_embedding_data = pre_embedding_res
    _finalize_cache_inputs(state)


def _finalize_cache_inputs(state: PipelineRunState) -> None:
    if state.chat_cache.config.input_summary_len is not None:
        state.pre_embedding_data = _summarize_input(
            state.pre_embedding_data, state.chat_cache.config.input_summary_len
        )
    state.pre_embedding_data = _apply_request_namespaces(
        state.pre_embedding_data,
        state.kwargs,
        state.chat_cache,
        context=state.context,
    )
    semantic_cache_allowed = _semantic_cache_allowed(state.chat_cache, state.kwargs, state.context)
    state.cache_enable = (
        state.cache_enable and semantic_cache_allowed and _cache_lookup_allowed(state.context)
    )


def maybe_delegate_bypass_sync(
    state: PipelineRunState,
    *,
    recursive_adapter: Any,
) -> Any:
    """Delegate directly to the next cache stage when the current stage should be bypassed."""

    if not (
        state.cache_enable
        and not state.cache_skip
        and _should_bypass_current_cache_stage(state.chat_cache, state.kwargs, state.context)
    ):
        return NO_RESULT
    state.context.setdefault("_byte_cache_stage_bypass", []).append(
        _cache_stage_name(state.chat_cache)
    )
    next_cache = state.chat_cache.next_cache
    if next_cache is None:
        state.cache_enable = False
        return NO_RESULT
    recursive_kwargs = build_recursive_kwargs(state)
    recursive_kwargs["cache_obj"] = next_cache
    return recursive_adapter(
        state.llm_handler,
        state.cache_data_convert,
        state.update_cache_callback,
        *state.args,
        **recursive_kwargs,
    )


async def maybe_delegate_bypass_async(
    state: PipelineRunState,
    *,
    recursive_adapter: Any,
) -> Any:
    """Delegate directly to the next cache stage when the current stage should be bypassed."""

    if not (
        state.cache_enable
        and not state.cache_skip
        and _should_bypass_current_cache_stage(state.chat_cache, state.kwargs, state.context)
    ):
        return NO_RESULT
    state.context.setdefault("_byte_cache_stage_bypass", []).append(
        _cache_stage_name(state.chat_cache)
    )
    next_cache = state.chat_cache.next_cache
    if next_cache is None:
        state.cache_enable = False
        return NO_RESULT
    recursive_kwargs = build_recursive_kwargs(state)
    recursive_kwargs["cache_obj"] = next_cache
    return await recursive_adapter(
        state.llm_handler,
        state.cache_data_convert,
        state.update_cache_callback,
        *state.args,
        **recursive_kwargs,
    )


def maybe_return_coalesced_sync(
    state: PipelineRunState,
    *,
    recursive_adapter: Any,
) -> Any:
    """Short-circuit with an inflight coalesced request when available."""

    if state.cache_enable and not state.search_only_flag and not state.kwargs.get("stream", False):
        state.coalesce_key = _make_coalesce_key(state.chat_cache, state.pre_embedding_data)
    if not state.coalesce_key:
        return NO_RESULT
    coalesced_result = get_coalescer(state.chat_cache).get_if_inflight(state.coalesce_key)
    if coalesced_result is _SENTINEL:
        return NO_RESULT
    byte_log.debug("request coalesced early for key %s", state.coalesce_key[:12])
    if not state.cache_enable:
        return coalesced_result
    retry_kwargs = _build_coalesced_retry_kwargs(
        state.kwargs,
        chat_cache=state.chat_cache,
        context=state.context,
        cache_factor=state.cache_factor,
        session=state.session,
        require_object_store=state.require_object_store,
        user_temperature=state.user_temperature,
        temperature=state.temperature,
        requested_top_k=state.requested_top_k,
    )
    retry_result = recursive_adapter(
        state.llm_handler,
        state.cache_data_convert,
        state.update_cache_callback,
        *state.args,
        **retry_kwargs,
    )
    return retry_result if retry_result is not None else coalesced_result


async def maybe_return_coalesced_async(
    state: PipelineRunState,
    *,
    recursive_adapter: Any,
) -> Any:
    """Short-circuit with an inflight coalesced request when available."""

    if state.cache_enable and not state.search_only_flag and not state.kwargs.get("stream", False):
        state.coalesce_key = _make_coalesce_key(state.chat_cache, state.pre_embedding_data)
    if not state.coalesce_key:
        return NO_RESULT
    coalesced_result = get_coalescer(state.chat_cache).get_if_inflight(state.coalesce_key)
    if coalesced_result is _SENTINEL:
        return NO_RESULT
    byte_log.debug("request coalesced early for key %s", state.coalesce_key[:12])
    if not state.cache_enable:
        return coalesced_result
    retry_kwargs = _build_coalesced_retry_kwargs(
        state.kwargs,
        chat_cache=state.chat_cache,
        context=state.context,
        cache_factor=state.cache_factor,
        session=state.session,
        require_object_store=state.require_object_store,
        user_temperature=state.user_temperature,
        temperature=state.temperature,
        requested_top_k=state.requested_top_k,
    )
    retry_result = await recursive_adapter(
        state.llm_handler,
        state.cache_data_convert,
        state.update_cache_callback,
        *state.args,
        **retry_kwargs,
    )
    return retry_result if retry_result is not None else coalesced_result


def maybe_embed_request_sync(state: PipelineRunState) -> None:
    """Populate sync embedding state when cache lookup is enabled."""

    if state.cache_enable:
        state.cache_stage_started_at = time.time()
        state.embedding_data = _embed_request(
            state.chat_cache,
            state.pre_embedding_data,
            state.context,
        )


async def maybe_embed_request_async(state: PipelineRunState) -> None:
    """Populate async embedding state when cache lookup is enabled."""

    if state.cache_enable:
        state.cache_stage_started_at = time.time()
        state.embedding_data = await _aembed_request(
            state.chat_cache,
            state.pre_embedding_data,
            state.context,
        )


__all__ = [
    "initialize_run_state",
    "maybe_delegate_bypass_async",
    "maybe_delegate_bypass_sync",
    "maybe_embed_request_async",
    "maybe_embed_request_sync",
    "maybe_return_coalesced_async",
    "maybe_return_coalesced_sync",
    "prepare_cache_inputs_async",
    "prepare_cache_inputs_sync",
]
