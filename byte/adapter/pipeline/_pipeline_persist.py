"""Cache persistence stages for sync and async adapter flows."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from byte.utils.error import ByteErrorCode, CacheError
from byte.utils.log import byte_log, log_byte_error
from byte.utils.time import time_cal

from ._pipeline_state import PipelineRunState
from .utils import (
    _admission_allowed,
    _answer_has_content,
    _await_with_report,
    _cache_save_allowed,
    _log_background_task_result,
)


def persist_response_sync(state: PipelineRunState) -> None:
    """Save a sync provider response into cache when policy allows it."""

    if not state.cache_enable or state.cache_skip:
        return
    try:

        def update_cache_func(handled_llm_data: Any, question: Any = None) -> None:
            resolved_question = state.pre_store_data if question is None else question
            if question is not None:
                resolved_question.content = state.pre_store_data
            if not _cache_save_allowed(state.context):
                return
            if not _admission_allowed(
                state.chat_cache,
                state.response_assessment,
                task_policy=state.task_policy,
            ):
                return
            if not _answer_has_content(handled_llm_data):
                byte_log.debug("skip cache save: empty answer content (sync)")
                return
            if _should_skip_dedup_sync(state):
                return
            time_cal(
                state.chat_cache.data_manager.save,
                func_name="save",
                report_func=state.chat_cache.report.save,
            )(
                resolved_question,
                handled_llm_data,
                state.embedding_data,
                extra_param=state.context.get("save_func", None),
                session=state.session,
            )
            if (
                state.chat_cache.report.op_save.count > 0
                and state.chat_cache.report.op_save.count % state.chat_cache.config.auto_flush == 0
            ):
                state.chat_cache.flush()

        state.llm_data = state.update_cache_callback(
            state.llm_data,
            update_cache_func,
            *state.args,
            **state.kwargs,
        )
    except Exception as exc:  # noqa: BLE001, RUF100 - callback boundary
        log_byte_error(
            byte_log,
            logging.WARNING,
            "failed to save the data to cache",
            error=exc,
            code=ByteErrorCode.PIPELINE_CACHE_SAVE,
            boundary="pipeline.cache_save",
            stage="sync_save",
        )


async def persist_response_async(state: PipelineRunState) -> None:
    """Save an async provider response into cache when policy allows it."""

    if not state.cache_enable or state.cache_skip:
        return
    try:

        def update_cache_func(handled_llm_data: Any, question: Any = None) -> None:
            resolved_question = state.pre_store_data if question is None else question
            if question is not None:
                resolved_question.content = state.pre_store_data
            if not _cache_save_allowed(state.context):
                return
            if not _admission_allowed(
                state.chat_cache,
                state.response_assessment,
                task_policy=state.task_policy,
            ):
                return
            if not _answer_has_content(handled_llm_data):
                byte_log.debug("skip cache save: empty answer content (async)")
                return

            async def save_to_cache() -> None:
                if await _should_skip_dedup_async(state):
                    return
                await _await_with_report(
                    state.chat_cache.data_manager.asave(
                        resolved_question,
                        handled_llm_data,
                        state.embedding_data,
                        extra_param=state.context.get("save_func", None),
                        session=state.session,
                    ),
                    func_name="save",
                    report_func=state.chat_cache.report.save,
                )
                if (
                    state.chat_cache.report.op_save.count > 0
                    and state.chat_cache.report.op_save.count % state.chat_cache.config.auto_flush == 0
                ):
                    await state.chat_cache.aflush()

            task = state.event_loop.create_task(save_to_cache())
            task.add_done_callback(_log_background_task_result)
            state.pending_cache_tasks.append(task)

        state.llm_data = state.update_cache_callback(
            state.llm_data,
            update_cache_func,
            *state.args,
            **state.kwargs,
        )
        if not _is_stream_like(state.llm_data) and state.pending_cache_tasks:
            await asyncio.gather(*state.pending_cache_tasks)
            state.pending_cache_tasks.clear()
    except Exception as exc:  # noqa: BLE001, RUF100 - callback boundary
        log_byte_error(
            byte_log,
            logging.ERROR,
            "failed to save the data to cache",
            error=exc,
            code=ByteErrorCode.PIPELINE_CACHE_SAVE,
            boundary="pipeline.cache_save",
            stage="async_save",
        )


def _should_skip_dedup_sync(state: PipelineRunState) -> bool:
    dedup_thresh = state.chat_cache.config.dedup_threshold
    if dedup_thresh <= 0.0 or state.embedding_data is None:
        return False
    try:
        existing = state.chat_cache.data_manager.search(state.embedding_data, top_k=1)
        return _existing_matches_threshold(state, existing, dedup_thresh)
    except (AttributeError, CacheError, IndexError, KeyError, TypeError, ValueError) as exc:
        log_byte_error(
            byte_log,
            logging.DEBUG,
            "semantic dedup check failed",
            error=exc,
            code=ByteErrorCode.PIPELINE_CACHE_SAVE,
            boundary="pipeline.cache_save",
            stage="sync_dedup",
            exc_info=False,
        )
        return False


async def _should_skip_dedup_async(state: PipelineRunState) -> bool:
    dedup_thresh = state.chat_cache.config.dedup_threshold
    if dedup_thresh <= 0.0 or state.embedding_data is None:
        return False
    try:
        existing = await state.chat_cache.data_manager.asearch(state.embedding_data, top_k=1)
        return _existing_matches_threshold(state, existing, dedup_thresh)
    except (AttributeError, CacheError, IndexError, KeyError, TypeError, ValueError) as exc:
        log_byte_error(
            byte_log,
            logging.DEBUG,
            "semantic dedup check failed",
            error=exc,
            code=ByteErrorCode.PIPELINE_CACHE_SAVE,
            boundary="pipeline.cache_save",
            stage="async_dedup",
            exc_info=False,
        )
        return False


def _existing_matches_threshold(
    state: PipelineRunState, existing: Any, dedup_thresh: float
) -> bool:
    if not existing:
        return False
    existing_score = existing[0][0]
    max_rank = state.chat_cache.similarity_evaluation.range()[1]
    if max_rank <= 0:
        return False
    ratio = existing_score / max_rank
    if ratio < dedup_thresh:
        return False
    byte_log.debug(
        "semantic dedup: skipping save (existing score %.4f >= thresh %.2f)",
        ratio,
        dedup_thresh,
    )
    return True


def _is_stream_like(value: Any) -> bool:
    from collections.abc import AsyncIterator as AsyncIteratorABC
    from collections.abc import Iterator as IteratorABC

    return isinstance(value, (AsyncIteratorABC, IteratorABC))


__all__ = ["persist_response_async", "persist_response_sync"]
