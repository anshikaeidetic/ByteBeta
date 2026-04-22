import hashlib
from collections.abc import Callable
from typing import Any, TypeVar

from byte.adapter.runtime_state import (
    get_adaptive_threshold,
    get_embedding_cache,
)
from byte.processor.intent import extract_request_intent
from byte.processor.policy import record_policy_event
from byte.processor.reasoning_reuse import (
    derive_reasoning_memory_record,
)
from byte.utils.time import time_cal

from .context import _negative_context_metadata, _repo_fingerprint_from_context
from .utils import (
    _extract_llm_reasoning,
    _extract_llm_tool_outputs,
    _time_cal_async,
)

_T = TypeVar("_T")


def _best_effort_memory(default: _T, operation: Callable[[], _T]) -> _T:
    """Run a memory-write boundary without surfacing cache bookkeeping failures."""

    try:
        return operation()
    except Exception:  # memory boundary bookkeeping
        return default


def _build_memory_metadata(llm_data, route_decision, memory_context, *, cache_hit) -> Any:
    metadata = dict(memory_context.get("metadata", {}) or {})
    metadata["cache_hit"] = bool(cache_hit)
    if route_decision is not None:
        metadata["model_route"] = route_decision.to_dict()
    if isinstance(llm_data, dict):
        if llm_data.get("usage") is not None:
            metadata["usage"] = llm_data.get("usage")
        if llm_data.get("byte_runtime") is not None:
            metadata["byte_runtime"] = llm_data.get("byte_runtime")
        if llm_data.get("id"):
            metadata["response_id"] = llm_data.get("id")
        if llm_data.get("finish_reason"):
            metadata["finish_reason"] = llm_data.get("finish_reason")
        choices = llm_data.get("choices") or []
        if choices:
            first_choice = choices[0] or {}
            if first_choice.get("finish_reason") is not None:
                metadata["finish_reason"] = first_choice.get("finish_reason")
    return metadata


def _record_ai_memory(
    chat_cache,
    request_kwargs,
    *,
    context,
    answer,
    embedding_data,
    llm_data=None,
    source="llm",
) -> None:
    if answer in (None, ""):
        return
    memory_context = context.get("_byte_memory", {}) or {}
    route_decision = context.get("_byte_model_route")
    provider = str(memory_context.get("provider") or "")
    if not provider and isinstance(llm_data, dict):
        provider = str(llm_data.get("byte_provider", "") or "")
    metadata = _build_memory_metadata(
        llm_data,
        route_decision,
        memory_context,
        cache_hit=(source == "cache"),
    )
    reasoning = memory_context.get("reasoning")
    if reasoning is None:
        reasoning = _extract_llm_reasoning(llm_data)
    tool_outputs = memory_context.get("tool_outputs")
    if tool_outputs is None:
        tool_outputs = _extract_llm_tool_outputs(llm_data)
    model_name = request_kwargs.get("model", "") or ""
    if source == "cache":
        chat_cache.touch_interaction(
            request_kwargs,
            answer=answer,
            reasoning=reasoning,
            tool_outputs=tool_outputs,
            embedding_data=embedding_data,
            model=model_name,
            provider=provider,
            metadata=metadata,
            source=source,
        )
        return
    chat_cache.remember_interaction(
        request_kwargs,
        answer=answer,
        reasoning=reasoning,
        tool_outputs=tool_outputs,
        embedding_data=embedding_data,
        model=model_name,
        provider=provider,
        metadata=metadata,
        source=source,
    )


def _record_reasoning_memory(chat_cache, request_kwargs, *, answer, verified, source="llm") -> None:
    if answer in (None, "") or not getattr(chat_cache.config, "reasoning_memory", True):
        return
    def write_reasoning() -> None:
        record = derive_reasoning_memory_record(request_kwargs, answer)
        if record is None:
            return
        chat_cache.remember_reasoning_result(
            kind=str(record.get("kind", "") or ""),
            key=str(record.get("key", "") or ""),
            answer=record.get("answer"),
            verified=bool(verified),
            metadata=dict(record.get("metadata", {}) or {}),
            source=source,
        )

    _best_effort_memory(None, write_reasoning)


def _record_failure_memory(chat_cache, request_kwargs, context, *, reason, llm_data=None) -> None:
    if not getattr(chat_cache.config, "failure_memory", True):
        return
    memory_context = context.get("_byte_memory", {}) or {}
    provider = str(memory_context.get("provider") or memory_context.get("byte_provider") or "")
    if not provider and isinstance(llm_data, dict):
        provider = str(llm_data.get("byte_provider", "") or "")
    model_name = str(request_kwargs.get("model", "") or "")
    metadata = {"source": "adapter", "reason": reason}
    if getattr(chat_cache.config, "negative_context_memory", True) and reason in {
        "cheap_response_rejected",
        "verification_failed",
        "schema_invalid",
        "unverified_code_answer",
    }:
        metadata.update(_negative_context_metadata(context))
    def write_failure_memory() -> None:
        chat_cache.remember_failure_pattern(
            request_kwargs,
            reason=reason,
            provider=provider,
            model=model_name,
            metadata=metadata,
        )

    _best_effort_memory(None, write_failure_memory)
    intent = extract_request_intent(request_kwargs)
    record_policy_event(
        intent.route_key,
        category=intent.category,
        event="clarify"
        if reason == "ambiguous_request"
        else "tool_first"
        if reason == "missing_tool_context"
        else "cheap_failure",
    )


def _record_execution_memory(chat_cache, request_kwargs, context, *, answer, llm_data=None) -> None:
    if answer in (None, "") or not getattr(chat_cache.config, "execution_memory", True):
        return
    memory_context = context.get("_byte_memory", {}) or {}
    provider = str(memory_context.get("provider") or memory_context.get("byte_provider") or "")
    if not provider and isinstance(llm_data, dict):
        provider = str(llm_data.get("byte_provider", "") or "")
    model_name = str(request_kwargs.get("model", "") or "")
    repo_fingerprint = _repo_fingerprint_from_context(context)
    verification = memory_context.get("verification")
    test_result = memory_context.get("test_result") or memory_context.get("byte_test_result")
    lint_result = memory_context.get("lint_result") or memory_context.get("byte_lint_result")
    schema_validation = memory_context.get("schema_validation") or memory_context.get(
        "byte_schema_validation"
    )
    tool_checks = memory_context.get("tool_checks") or memory_context.get("byte_tool_checks")
    test_command = memory_context.get("test_command") or memory_context.get("byte_test_command")
    patch = memory_context.get("patch") or memory_context.get("byte_patch")
    def write_execution_memory() -> None:
        execution_entry = chat_cache.remember_execution_result(
            request_kwargs,
            answer=answer,
            verification=verification,
            patch=patch,
            test_command=test_command,
            test_result=test_result,
            lint_result=lint_result,
            schema_validation=schema_validation,
            tool_checks=tool_checks,
            repo_fingerprint=repo_fingerprint,
            model=model_name,
            provider=provider,
            metadata={
                "workflow": getattr(context.get("_byte_workflow_plan"), "to_dict", dict)()
            },
        )
        if patch not in (None, ""):
            chat_cache.remember_patch_pattern(
                request_kwargs,
                patch=patch,
                repo_fingerprint=repo_fingerprint,
                verified=bool(execution_entry.get("verified", False)),
                model=model_name,
                provider=provider,
            )
            intent = extract_request_intent(request_kwargs)
            record_policy_event(
                intent.route_key,
                category=intent.category,
                event="verified_patch_reuse"
                if execution_entry.get("verified")
                else "cheap_success",
            )

    _best_effort_memory(None, write_execution_memory)


def _record_workflow_outcome(chat_cache, request_kwargs, context, *, success, reason="") -> None:
    workflow_decision = context.get("_byte_workflow_plan")
    if workflow_decision is None:
        return
    counterfactual = dict(context.get("_byte_counterfactual", {}) or {})
    def write_workflow_outcome() -> None:
        chat_cache.remember_workflow_plan(
            request_kwargs,
            action=str(getattr(workflow_decision, "action", "") or ""),
            route_preference=str(getattr(workflow_decision, "route_preference", "") or ""),
            counterfactual_action=str(counterfactual.get("action", "") or ""),
            counterfactual_reason=str(counterfactual.get("reason", "") or ""),
            repo_fingerprint=_repo_fingerprint_from_context(context),
            artifact_fingerprint=str(context.get("_byte_artifact_fingerprint", "") or ""),
            success=bool(success),
            metadata={
                "reason": reason or str(getattr(workflow_decision, "reason", "") or ""),
                "planner_hints": getattr(workflow_decision, "planner_hints", {}) or {},
                "uncertainty": dict(context.get("_byte_uncertainty", {}) or {}),
            },
        )

    _best_effort_memory(None, write_workflow_outcome)


def _get_embedding_cache_key(chat_cache, pre_embedding_data) -> tuple[Any, ...] | None:
    """Namespace cached embeddings by cache instance and embedding function."""
    if not isinstance(pre_embedding_data, str):
        return None
    return (
        id(chat_cache),
        id(chat_cache.embedding_func),
        hashlib.sha256(str(pre_embedding_data).encode()).hexdigest(),
    )


def _embed_request(chat_cache, pre_embedding_data, context) -> Any:
    """Compute embeddings with an instance-safe in-memory LRU."""
    emb_cache = get_embedding_cache(chat_cache, chat_cache.config.embedding_cache_size)
    emb_key = _get_embedding_cache_key(chat_cache, pre_embedding_data)

    if emb_key and emb_key in emb_cache:
        return emb_cache[emb_key]

    embedding_data = time_cal(
        chat_cache.embedding_func,
        func_name="embedding",
        report_func=chat_cache.report.embedding,
    )(pre_embedding_data, extra_param=context.get("embedding_func", None))
    if emb_key is not None:
        emb_cache[emb_key] = embedding_data
    return embedding_data
async def _aembed_request(chat_cache, pre_embedding_data, context) -> Any:
    """Async variant of embedding generation for async adapter flow."""
    emb_cache = get_embedding_cache(chat_cache, chat_cache.config.embedding_cache_size)
    emb_key = _get_embedding_cache_key(chat_cache, pre_embedding_data)

    if emb_key and emb_key in emb_cache:
        return emb_cache[emb_key]

    embedding_data = await _time_cal_async(
        chat_cache.embedding_func,
        pre_embedding_data,
        extra_param=context.get("embedding_func", None),
        func_name="embedding",
        report_func=chat_cache.report.embedding,
    )
    if emb_key is not None:
        emb_cache[emb_key] = embedding_data
    return embedding_data


def _resolve_similarity_threshold(chat_cache) -> Any:
    """Resolve the current similarity threshold, including adaptive mode."""
    if chat_cache.config.adaptive_threshold:
        return get_adaptive_threshold(chat_cache).current_threshold
    return chat_cache.config.similarity_threshold


def _make_coalesce_key(chat_cache, pre_embedding_data) -> Any | None:
    if not isinstance(pre_embedding_data, str):
        return None
    raw = f"{id(chat_cache)}::{id(chat_cache.pre_embedding_func)}::{pre_embedding_data}"
    return hashlib.sha256(raw.encode()).hexdigest()


__all__ = [name for name in globals() if not name.startswith("__")]
