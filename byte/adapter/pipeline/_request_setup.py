"""Shared request normalization and early-exit handling for pipeline flows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from byte.processor.intent import extract_request_intent
from byte.processor.policy import record_policy_event
from byte.processor.quality import extract_output_contract
from byte.processor.reasoning_reuse import resolve_reasoning_shortcut
from byte.processor.reuse_policy import detect_reuse_policy
from byte.security import redact_text, sanitize_outbound_overrides
from byte.utils.log import byte_log

from .context import (
    _compile_context_if_needed,
    _maybe_route_request,
    _plan_workflow,
    _resolve_memory_context,
)

_input_summarizer = None


@dataclass(slots=True)
class PreparedPipelineRequest:
    """Normalized request state before cache lookup or upstream execution."""

    request_kwargs: dict[str, Any]
    context: dict[str, Any]
    early_response: Any | None = None


def _build_synthetic_response(
    request_kwargs: dict[str, Any],
    content: Any,
    *,
    byte_reason: str,
    model: str | None = None,
) -> dict[str, Any]:
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


def _safe_log_value(chat_cache: Any, value: Any) -> Any:
    if getattr(chat_cache.config, "security_redact_logs", False):
        return redact_text(value)
    return value


def _return_reasoning_shortcut(
    chat_cache: Any,
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    shortcut: Any,
) -> dict[str, Any]:
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
        try:
            chat_cache.record_intent(request_kwargs)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            byte_log.debug("Skipping intent memory capture for reasoning shortcut: %s", exc)
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
    try:
        intent = extract_request_intent(request_kwargs)
        record_policy_event(
            intent.route_key,
            category=intent.category,
            event=shortcut.byte_reason,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        byte_log.debug("Skipping reasoning shortcut policy event: %s", exc)
    _record_workflow_outcome(
        chat_cache, request_kwargs, context, success=True, reason=shortcut.reason
    )
    return synthetic_response


def _maybe_reasoning_shortcut(
    chat_cache: Any, request_kwargs: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any] | None:
    if request_kwargs.get("byte_disable_reasoning_shortcut"):
        return None
    if not getattr(chat_cache.config, "reasoning_reuse", True):
        return None
    try:
        shortcut = resolve_reasoning_shortcut(
            request_kwargs,
            store=getattr(chat_cache, "reasoning_memory_store", None),
            config=chat_cache.config,
            context_hints=context,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        byte_log.debug("Skipping reasoning shortcut lookup: %s", exc)
        shortcut = None
    if shortcut is None:
        return None
    return _return_reasoning_shortcut(chat_cache, request_kwargs, context, shortcut)


def _summarize_input(text: str, text_length: int) -> str:
    if len(text) <= text_length:
        return text

    # pylint: disable=import-outside-toplevel
    from byte.processor.context.summarization_context import SummarizationContextProcess

    global _input_summarizer
    try:
        if _input_summarizer is None:
            _input_summarizer = SummarizationContextProcess()
        return _input_summarizer.summarize_to_sentence([text], text_length)
    except Exception as exc:  # pylint: disable=W0703
        byte_log.debug("Falling back to truncation-based input summary: %s", exc)
        token_budget = max(int(text_length) - 1, 1)
        return " ".join(str(text).split()[:token_budget])


def _build_coalesced_retry_kwargs(
    request_kwargs: dict[str, Any],
    *,
    chat_cache: Any,
    context: dict[str, Any],
    cache_factor: float,
    session: Any,
    require_object_store: bool,
    user_temperature: bool,
    temperature: float,
    requested_top_k: Any,
) -> dict[str, Any]:
    retry_kwargs = dict(request_kwargs)
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


def maybe_record_intent_memory(
    chat_cache: Any, request_kwargs: dict[str, Any], *, session: Any, mode_label: str
) -> None:
    """Best-effort intent memory capture with bounded failure modes."""
    if not getattr(chat_cache.config, "intent_memory", False):
        return
    try:
        chat_cache.record_intent(request_kwargs, session_id=session.name if session else None)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        byte_log.debug("Skipping intent memory capture for %s request: %s", mode_label, exc)


def prepare_request_for_execution(
    chat_cache: Any,
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    *,
    session: Any,
) -> PreparedPipelineRequest:
    """Normalize request state before cache lookup or upstream execution."""
    _resolve_memory_context(request_kwargs, context)
    normalized_kwargs = _compile_context_if_needed(
        chat_cache, request_kwargs, context, session=session
    )
    context["_byte_request_kwargs"] = dict(normalized_kwargs)
    context["_byte_output_contract"] = extract_output_contract(normalized_kwargs).to_dict()
    reuse_policy = detect_reuse_policy(
        normalized_kwargs,
        config=chat_cache.config,
        context=context,
    )
    context["_byte_reuse_policy"] = reuse_policy.to_dict()
    if reuse_policy.mode != "full_reuse":
        intent = extract_request_intent(normalized_kwargs)
        record_policy_event(
            intent.route_key,
            category=intent.category,
            event=(
                "context_only_reuse" if reuse_policy.mode == "context_only" else "direct_only_reuse"
            ),
        )
    normalized_kwargs = sanitize_outbound_overrides(normalized_kwargs, chat_cache.config)
    workflow_decision = _plan_workflow(chat_cache, normalized_kwargs, context)
    if workflow_decision is not None:
        from .memory import _record_failure_memory, _record_workflow_outcome

        if workflow_decision.action == "clarify":
            _record_failure_memory(
                chat_cache, normalized_kwargs, context, reason="ambiguous_request"
            )
            _record_workflow_outcome(
                chat_cache, normalized_kwargs, context, success=False, reason="clarify"
            )
            return PreparedPipelineRequest(
                request_kwargs=normalized_kwargs,
                context=context,
                early_response=_build_synthetic_response(
                    normalized_kwargs,
                    workflow_decision.response_text,
                    byte_reason="clarification_required",
                ),
            )
        if workflow_decision.action == "reuse_verified_patch" and workflow_decision.response_text:
            _record_workflow_outcome(
                chat_cache,
                normalized_kwargs,
                context,
                success=True,
                reason="reuse_verified_patch",
            )
            return PreparedPipelineRequest(
                request_kwargs=normalized_kwargs,
                context=context,
                early_response=_build_synthetic_response(
                    normalized_kwargs,
                    workflow_decision.response_text,
                    byte_reason="verified_patch_reuse",
                ),
            )
    reasoning_response = _maybe_reasoning_shortcut(chat_cache, normalized_kwargs, context)
    if reasoning_response is not None:
        return PreparedPipelineRequest(
            request_kwargs=normalized_kwargs,
            context=context,
            early_response=reasoning_response,
        )
    _maybe_route_request(chat_cache, normalized_kwargs, context)
    context["_byte_request_kwargs"] = dict(normalized_kwargs)
    return PreparedPipelineRequest(request_kwargs=normalized_kwargs, context=context)


__all__ = [
    "PreparedPipelineRequest",
    "_build_coalesced_retry_kwargs",
    "_build_synthetic_response",
    "_maybe_reasoning_shortcut",
    "_safe_log_value",
    "_summarize_input",
    "maybe_record_intent_memory",
    "prepare_request_for_execution",
]
