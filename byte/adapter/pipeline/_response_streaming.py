
"""Persistence and stream wrapping for finalized upstream responses."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator as AsyncIteratorABC
from collections.abc import Iterator as IteratorABC
from typing import Any

from byte.utils.log import byte_log
from byte.utils.time import time_cal

from ._response_commit import _complete_coalescer, _record_route_completion
from ._runtime_support import (
    _admission_allowed,
    _answer_has_content,
    _await_with_report,
    _cache_save_allowed,
    _log_background_task_result,
)
from .memory import (
    _record_ai_memory,
    _record_execution_memory,
    _record_failure_memory,
    _record_reasoning_memory,
    _record_workflow_outcome,
)
from .stream import _wrap_async_stream_with_memory, _wrap_sync_stream_with_memory
from .utils import _extract_llm_answer


def finalize_sync_llm_response(
    *,
    chat_cache: Any,
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    llm_data: Any,
    response_assessment: Any,
    embedding_data: Any,
    pre_store_data: Any,
    session: Any,
    start_time: float,
    cache_enable: bool,
    coalesce_key: str | None,
    update_cache_callback: Any,
    args: tuple[Any, ...],
) -> Any:
    """Persist and materialize a finalized sync response."""
    task_policy = context.get("_byte_task_policy") or {}
    if cache_enable and not request_kwargs.get("cache_skip", False):
        try:

            def update_cache_func(handled_llm_data: Any, question: Any = None) -> None:
                cache_question = question if question is not None else pre_store_data
                if question is not None:
                    cache_question.content = pre_store_data

                if not _cache_save_allowed(context):
                    return
                if not _admission_allowed(chat_cache, response_assessment, task_policy=task_policy):
                    return
                if not _answer_has_content(handled_llm_data):
                    byte_log.debug("skip cache save: empty answer content (stream sync)")
                    return

                dedup_thresh = chat_cache.config.dedup_threshold
                if dedup_thresh > 0.0 and embedding_data is not None:
                    try:
                        existing = chat_cache.data_manager.search(embedding_data, top_k=1)
                        if existing:
                            existing_score = existing[0][0]
                            max_rank = chat_cache.similarity_evaluation.range()[1]
                            if max_rank > 0 and (existing_score / max_rank) >= dedup_thresh:
                                byte_log.debug(
                                    "semantic dedup: skipping save (existing score %.4f >= thresh %.2f)",
                                    existing_score / max_rank,
                                    dedup_thresh,
                                )
                                return
                    except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as exc:
                        byte_log.debug("Skipping sync semantic dedup check: %s", exc)

                time_cal(
                    chat_cache.data_manager.save,
                    func_name="save",
                    report_func=chat_cache.report.save,
                )(
                    cache_question,
                    handled_llm_data,
                    embedding_data,
                    extra_param=context.get("save_func"),
                    session=session,
                )
                if (
                    chat_cache.report.op_save.count > 0
                    and chat_cache.report.op_save.count % chat_cache.config.auto_flush == 0
                ):
                    chat_cache.flush()

            llm_data = update_cache_callback(llm_data, update_cache_func, *args, **request_kwargs)
        except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
            byte_log.warning("failed to save the data to cache, error: %s", exc)

    _, accepted = _record_route_completion(
        chat_cache,
        context,
        response_assessment,
        start_time=start_time,
    )
    if isinstance(llm_data, AsyncIteratorABC):
        return _wrap_async_stream_with_memory(
            chat_cache,
            dict(request_kwargs),
            context,
            llm_data,
            embedding_data=embedding_data,
        )
    if isinstance(llm_data, IteratorABC) and not isinstance(
        llm_data, (dict, list, tuple, str, bytes)
    ):
        return _wrap_sync_stream_with_memory(
            chat_cache,
            dict(request_kwargs),
            context,
            llm_data,
            embedding_data=embedding_data,
        )
    if response_assessment is not None and not response_assessment.accepted:
        _record_failure_memory(
            chat_cache, request_kwargs, context, reason="verification_failed", llm_data=llm_data
        )
        _record_workflow_outcome(
            chat_cache, request_kwargs, context, success=False, reason="verification_failed"
        )
    answer_text = _extract_llm_answer(llm_data)
    if answer_text not in (None, ""):
        _record_ai_memory(
            chat_cache,
            request_kwargs,
            context=context,
            answer=answer_text,
            embedding_data=embedding_data,
            llm_data=llm_data,
            source="llm",
        )
        _record_execution_memory(
            chat_cache,
            request_kwargs,
            context,
            answer=answer_text,
            llm_data=llm_data,
        )
        _record_reasoning_memory(
            chat_cache,
            request_kwargs,
            answer=answer_text,
            verified=accepted,
            source="llm",
        )
        _record_workflow_outcome(
            chat_cache,
            request_kwargs,
            context,
            success=accepted,
            reason="completed",
        )
    _complete_coalescer(chat_cache, coalesce_key, llm_data)
    return llm_data


async def finalize_async_llm_response(
    *,
    chat_cache: Any,
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    llm_data: Any,
    response_assessment: Any,
    embedding_data: Any,
    pre_store_data: Any,
    session: Any,
    start_time: float,
    cache_enable: bool,
    coalesce_key: str | None,
    update_cache_callback: Any,
    args: tuple[Any, ...],
    event_loop: asyncio.AbstractEventLoop,
) -> Any:
    """Persist and materialize a finalized async response."""
    task_policy = context.get("_byte_task_policy") or {}
    pending_cache_tasks: list[asyncio.Task[Any]] = []
    if cache_enable and not request_kwargs.get("cache_skip", False):
        try:

            def update_cache_func(handled_llm_data: Any, question: Any = None) -> None:
                cache_question = question if question is not None else pre_store_data
                if question is not None:
                    cache_question.content = pre_store_data

                if not _cache_save_allowed(context):
                    return
                if not _admission_allowed(chat_cache, response_assessment, task_policy=task_policy):
                    return
                if not _answer_has_content(handled_llm_data):
                    byte_log.debug("skip cache save: empty answer content (stream async)")
                    return

                async def save_to_cache() -> None:
                    dedup_thresh = chat_cache.config.dedup_threshold
                    if dedup_thresh > 0.0 and embedding_data is not None:
                        try:
                            existing = await chat_cache.data_manager.asearch(
                                embedding_data, top_k=1
                            )
                            if existing:
                                existing_score = existing[0][0]
                                max_rank = chat_cache.similarity_evaluation.range()[1]
                                if max_rank > 0 and (existing_score / max_rank) >= dedup_thresh:
                                    byte_log.debug(
                                        "semantic dedup: skipping save (existing score %.4f >= thresh %.2f)",
                                        existing_score / max_rank,
                                        dedup_thresh,
                                    )
                                    return
                        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as exc:
                            byte_log.debug("Skipping async semantic dedup check: %s", exc)

                    await _await_with_report(
                        chat_cache.data_manager.asave(
                            cache_question,
                            handled_llm_data,
                            embedding_data,
                            extra_param=context.get("save_func"),
                            session=session,
                        ),
                        func_name="save",
                        report_func=chat_cache.report.save,
                    )
                    if (
                        chat_cache.report.op_save.count > 0
                        and chat_cache.report.op_save.count % chat_cache.config.auto_flush == 0
                    ):
                        await chat_cache.aflush()

                task = event_loop.create_task(save_to_cache())
                task.add_done_callback(_log_background_task_result)
                pending_cache_tasks.append(task)

            llm_data = update_cache_callback(llm_data, update_cache_func, *args, **request_kwargs)
            if not isinstance(llm_data, (AsyncIteratorABC, IteratorABC)) and pending_cache_tasks:
                await asyncio.gather(*pending_cache_tasks)
                pending_cache_tasks.clear()
        except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError):
            byte_log.error("failed to save the data to cache", exc_info=True)

    _, accepted = _record_route_completion(
        chat_cache,
        context,
        response_assessment,
        start_time=start_time,
    )
    if isinstance(llm_data, AsyncIteratorABC):
        return _wrap_async_stream_with_memory(
            chat_cache,
            dict(request_kwargs),
            context,
            llm_data,
            embedding_data=embedding_data,
            pending_cache_tasks=pending_cache_tasks,
        )
    if isinstance(llm_data, IteratorABC) and not isinstance(
        llm_data, (dict, list, tuple, str, bytes)
    ):
        return _wrap_sync_stream_with_memory(
            chat_cache,
            dict(request_kwargs),
            context,
            llm_data,
            embedding_data=embedding_data,
        )
    if response_assessment is not None and not response_assessment.accepted:
        _record_failure_memory(
            chat_cache, request_kwargs, context, reason="verification_failed", llm_data=llm_data
        )
        _record_workflow_outcome(
            chat_cache, request_kwargs, context, success=False, reason="verification_failed"
        )
    answer_text = _extract_llm_answer(llm_data)
    if answer_text not in (None, ""):
        _record_ai_memory(
            chat_cache,
            request_kwargs,
            context=context,
            answer=answer_text,
            embedding_data=embedding_data,
            llm_data=llm_data,
            source="llm",
        )
        _record_execution_memory(
            chat_cache,
            request_kwargs,
            context,
            answer=answer_text,
            llm_data=llm_data,
        )
        _record_reasoning_memory(
            chat_cache,
            request_kwargs,
            answer=answer_text,
            verified=accepted,
            source="llm",
        )
        _record_workflow_outcome(
            chat_cache,
            request_kwargs,
            context,
            success=accepted,
            reason="completed",
        )
    _complete_coalescer(chat_cache, coalesce_key, llm_data)
    return llm_data

__all__ = ["finalize_async_llm_response", "finalize_sync_llm_response"]
