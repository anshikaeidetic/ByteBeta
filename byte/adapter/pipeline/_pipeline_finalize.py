"""Final response shaping and memory side effects for the adapter pipeline."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator as AsyncIteratorABC
from collections.abc import Iterator as IteratorABC
from typing import Any

from byte.processor.model_router import record_route_outcome
from byte.processor.policy import record_policy_event

from ._pipeline_common import complete_coalesced_request
from ._pipeline_state import PipelineRunState
from .memory import (
    _record_ai_memory,
    _record_execution_memory,
    _record_failure_memory,
    _record_reasoning_memory,
    _record_workflow_outcome,
)
from .stream import _wrap_async_stream_with_memory, _wrap_sync_stream_with_memory
from .utils import _extract_llm_answer


def finalize_pipeline_response(state: PipelineRunState) -> Any:
    """Apply the common post-provider bookkeeping and return the final response."""

    final_route_decision = state.context.get("_byte_model_route")
    accepted = state.response_assessment.accepted if state.response_assessment else True
    record_route_outcome(
        final_route_decision,
        accepted=accepted,
        latency_ms=(time.time() - state.start_time) * 1000,
    )
    if final_route_decision is not None and getattr(
        state.chat_cache.config,
        "tenant_policy_learning",
        True,
    ):
        record_policy_event(
            final_route_decision.route_key,
            category=final_route_decision.category,
            event=_policy_event_name(final_route_decision.tier, accepted),
            latency_ms=(time.time() - state.start_time) * 1000,
        )
    if isinstance(state.llm_data, AsyncIteratorABC):
        return _wrap_async_stream_with_memory(
            state.chat_cache,
            dict(state.kwargs),
            state.context,
            state.llm_data,
            embedding_data=state.embedding_data,
            pending_cache_tasks=state.pending_cache_tasks,
        )
    if isinstance(state.llm_data, IteratorABC) and not isinstance(
        state.llm_data,
        (dict, list, tuple, str, bytes),
    ):
        return _wrap_sync_stream_with_memory(
            state.chat_cache,
            dict(state.kwargs),
            state.context,
            state.llm_data,
            embedding_data=state.embedding_data,
        )
    if state.response_assessment is not None and not state.response_assessment.accepted:
        _record_failure_memory(
            state.chat_cache,
            state.kwargs,
            state.context,
            reason="verification_failed",
            llm_data=state.llm_data,
        )
        _record_workflow_outcome(
            state.chat_cache,
            state.kwargs,
            state.context,
            success=False,
            reason="verification_failed",
        )
    answer_text = _extract_llm_answer(state.llm_data)
    if answer_text not in (None, ""):
        _record_ai_memory(
            state.chat_cache,
            state.kwargs,
            context=state.context,
            answer=answer_text,
            embedding_data=state.embedding_data,
            llm_data=state.llm_data,
            source="llm",
        )
        _record_execution_memory(
            state.chat_cache,
            state.kwargs,
            state.context,
            answer=answer_text,
            llm_data=state.llm_data,
        )
        _record_reasoning_memory(
            state.chat_cache,
            state.kwargs,
            answer=answer_text,
            verified=accepted,
            source="llm",
        )
        _record_workflow_outcome(
            state.chat_cache,
            state.kwargs,
            state.context,
            success=accepted,
            reason="completed",
        )
    complete_coalesced_request(state, state.llm_data)
    return state.llm_data


def _policy_event_name(tier: str, accepted: bool) -> str:
    if tier == "cheap" and accepted:
        return "cheap_success"
    if tier == "cheap":
        return "cheap_failure"
    return "request_complete"


__all__ = ["finalize_pipeline_response"]
