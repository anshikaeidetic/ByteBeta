"""Common helpers shared across sync and async pipeline stages."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from byte.adapter.runtime_state import get_coalescer
from byte.utils.error import ByteErrorCode
from byte.utils.log import byte_log, log_byte_error

from ._pipeline_state import PipelineRunState

NO_RESULT = object()


def build_recursive_kwargs(state: PipelineRunState) -> dict[str, Any]:
    """Build recursive adapter kwargs while preserving user-specified controls."""

    kwargs = dict(state.kwargs)
    kwargs["cache_context"] = state.context
    kwargs["cache_skip"] = state.cache_skip
    kwargs["cache_factor"] = state.cache_factor
    kwargs["search_only"] = state.search_only_flag
    if state.session is not None:
        kwargs["session"] = state.session
    if state.require_object_store:
        kwargs["require_object_store"] = True
    if state.user_temperature:
        kwargs["temperature"] = state.temperature
    if state.requested_top_k is not None:
        kwargs["top_k"] = state.requested_top_k
    return kwargs


def cancel_coalesced_request(state: PipelineRunState) -> None:
    """Cancel an inflight coalesced request if the current pipeline aborts."""

    if state.coalesce_key:
        get_coalescer(state.chat_cache).cancel(state.coalesce_key)


def complete_coalesced_request(state: PipelineRunState, result: Any) -> None:
    """Complete an inflight coalesced request with the final pipeline result."""

    if state.coalesce_key:
        get_coalescer(state.chat_cache).complete(state.coalesce_key, result)


def best_effort_log(
    message: str,
    *,
    error: Exception,
    code: ByteErrorCode | str,
    boundary: str,
    stage: str,
    level: int = logging.DEBUG,
) -> None:
    """Emit a structured Byte log for a tolerated boundary failure."""

    log_byte_error(
        byte_log,
        level,
        message,
        error=error,
        code=code,
        boundary=boundary,
        stage=stage,
        exc_info=level >= logging.WARNING,
    )


def with_coalescer_guard(state: PipelineRunState, callback: Callable[[], Any], *, stage: str) -> Any:
    """Execute a sync callback and cancel coalescing on boundary failures."""

    try:
        return callback()
    except Exception as exc:  # pragma: no cover - boundary cleanup
        cancel_coalesced_request(state)
        log_byte_error(
            byte_log,
            logging.ERROR,
            "pipeline stage failed",
            error=exc,
            code=ByteErrorCode.PIPELINE_ESCALATION,
            boundary="pipeline.coalescer",
            stage=stage,
        )
        raise


async def awith_coalescer_guard(
    state: PipelineRunState,
    callback: Callable[[], Awaitable[Any]],
    *,
    stage: str,
) -> Any:
    """Execute an async callback and cancel coalescing on boundary failures."""

    try:
        return await callback()
    except Exception as exc:  # pragma: no cover - boundary cleanup
        cancel_coalesced_request(state)
        log_byte_error(
            byte_log,
            logging.ERROR,
            "pipeline stage failed",
            error=exc,
            code=ByteErrorCode.PIPELINE_ESCALATION,
            boundary="pipeline.coalescer",
            stage=stage,
        )
        raise


__all__ = [
    "NO_RESULT",
    "awith_coalescer_guard",
    "best_effort_log",
    "build_recursive_kwargs",
    "cancel_coalesced_request",
    "complete_coalesced_request",
    "with_coalescer_guard",
]
