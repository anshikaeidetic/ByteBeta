"""Cache-hit verification and response materialization helpers."""

from __future__ import annotations

import time
from typing import Any

from byte.adapter.runtime_state import get_adaptive_threshold, get_coalescer
from byte.processor.post import LlmVerifier, temperature_softmax
from byte.utils.log import byte_log
from byte.utils.time import time_cal

from ._runtime_support import (
    _cache_reuse_allowed,
    _record_cache_stage_latency,
    _time_cal_async,
    get_budget_tracker,
    get_quality_scorer,
)
from .memory import _record_ai_memory, _record_failure_memory
from .verifier import _repair_cached_answer


def _post_process_cache_answers(
    chat_cache: Any,
    cache_answers: list[tuple[float, Any, Any, Any]],
    *,
    original_question: Any,
    temperature: float,
) -> Any:
    if chat_cache.post_process_messages_func is temperature_softmax:
        return chat_cache.post_process_messages_func(
            messages=[item[1] for item in cache_answers],
            scores=[item[0] for item in cache_answers],
            temperature=temperature,
        )
    if chat_cache.post_process_messages_func is LlmVerifier:
        return chat_cache.post_process_messages_func(
            messages=[item[1] for item in cache_answers],
            scores=[item[0] for item in cache_answers],
            original_question=original_question,
        )
    return chat_cache.post_process_messages_func([item[1] for item in cache_answers])


def _score_cache_hit_quality(
    chat_cache: Any,
    cache_answers: list[tuple[float, Any, Any, Any]],
    *,
    query: Any,
    cached_answer: Any,
    min_rank: float,
    max_rank: float,
) -> None:
    try:
        best_rank = cache_answers[0][0] if cache_answers else 0.0
        factor = max_rank - min_rank
        similarity_score = best_rank / factor if factor else best_rank
        get_quality_scorer(chat_cache).score(
            query=str(query),
            cached_answer=str(cached_answer),
            similarity_score=similarity_score,
        )
    except Exception as exc:  # pylint: disable=W0703
        byte_log.debug("Skipping cache-hit quality scoring: %s", exc)


def _report_sync_cache_hit(
    chat_cache: Any,
    cache_whole_data: tuple[float, Any, Any, Any] | None,
    *,
    session: Any,
    pre_embedding_data: Any,
    pre_store_data: Any,
    start_time: float,
) -> None:
    if session and cache_whole_data:
        chat_cache.data_manager.add_session(
            cache_whole_data[2], session.name, pre_embedding_data
        )
    if cache_whole_data and not chat_cache.config.disable_report:
        report_cache_data = cache_whole_data[3]
        report_search_data = cache_whole_data[2]
        chat_cache.data_manager.report_cache(
            pre_store_data if isinstance(pre_store_data, str) else "",
            report_cache_data.question if isinstance(report_cache_data.question, str) else "",
            report_search_data[1],
            (
                report_cache_data.answers[0].answer
                if isinstance(report_cache_data.answers[0].answer, str)
                else ""
            ),
            cache_whole_data[0],
            round(time.time() - start_time, 6),
        )


async def _report_async_cache_hit(
    chat_cache: Any,
    cache_whole_data: tuple[float, Any, Any, Any] | None,
    *,
    session: Any,
    pre_embedding_data: Any,
    pre_store_data: Any,
    start_time: float,
) -> None:
    if session and cache_whole_data:
        await chat_cache.data_manager.aadd_session(
            cache_whole_data[2], session.name, pre_embedding_data
        )
    if cache_whole_data and not chat_cache.config.disable_report:
        report_cache_data = cache_whole_data[3]
        report_search_data = cache_whole_data[2]
        await chat_cache.data_manager.areport_cache(
            pre_store_data if isinstance(pre_store_data, str) else "",
            report_cache_data.question if isinstance(report_cache_data.question, str) else "",
            report_search_data[1],
            (
                report_cache_data.answers[0].answer
                if isinstance(report_cache_data.answers[0].answer, str)
                else ""
            ),
            cache_whole_data[0],
            round(time.time() - start_time, 6),
        )


def resolve_sync_cache_hit(
    *,
    chat_cache: Any,
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    cache_answers: list[tuple[float, Any, Any, Any]],
    temperature: float,
    pre_store_data: Any,
    pre_embedding_data: Any,
    embedding_data: Any,
    session: Any,
    start_time: float,
    cache_stage_started_at: float | None,
    coalesce_key: str | None,
    cache_data_convert: Any,
) -> Any | None:
    """Return a converted cache hit response when a verified hit is usable."""
    if not cache_answers:
        return None
    min_rank, max_rank = chat_cache.similarity_evaluation.range()
    return_message = time_cal(
        lambda: _post_process_cache_answers(
            chat_cache,
            cache_answers,
            original_question=pre_embedding_data,
            temperature=temperature,
        ),
        func_name="post_process",
        report_func=chat_cache.report.post,
    )()
    if return_message is not None:
        return_message, cache_assessment = _repair_cached_answer(
            chat_cache,
            request_kwargs,
            return_message,
            context=context,
            task_policy=context.get("_byte_task_policy") or {},
        )
        if (
            cache_assessment is not None
            and cache_assessment.constraint != "freeform"
            and not cache_assessment.accepted
        ):
            _record_failure_memory(
                chat_cache, request_kwargs, context, reason="cache_revalidation_failed"
            )
            return_message = None
        if not _cache_reuse_allowed(chat_cache, request_kwargs, context, return_message):
            _record_failure_memory(
                chat_cache, request_kwargs, context, reason="unverified_code_answer"
            )
            return_message = None
    if return_message is None:
        return None

    chat_cache.report.hint_cache()
    if chat_cache.config.adaptive_threshold:
        get_adaptive_threshold(chat_cache).record(hit=True)
    model_name = request_kwargs.get("model", "unknown")
    get_budget_tracker(chat_cache).record_cache_hit(
        model_name,
        prompt_tokens=len(str(pre_embedding_data).split()) * 2,
        completion_tokens=len(str(return_message).split()) * 2,
    )
    _score_cache_hit_quality(
        chat_cache,
        cache_answers,
        query=pre_store_data,
        cached_answer=return_message,
        min_rank=min_rank,
        max_rank=max_rank,
    )
    cache_whole_data = next((item for item in cache_answers if item[1] == return_message), None)
    _report_sync_cache_hit(
        chat_cache,
        cache_whole_data,
        session=session,
        pre_embedding_data=pre_embedding_data,
        pre_store_data=pre_store_data,
        start_time=start_time,
    )
    _record_ai_memory(
        chat_cache,
        request_kwargs,
        context=context,
        answer=return_message,
        embedding_data=cache_whole_data[3].embedding_data if cache_whole_data else embedding_data,
        source="cache",
    )
    converted_response = cache_data_convert(return_message)
    if coalesce_key:
        get_coalescer(chat_cache).complete(coalesce_key, converted_response)
    _record_cache_stage_latency(
        chat_cache,
        request_kwargs,
        context,
        started_at=cache_stage_started_at,
        hit=True,
    )
    return converted_response


async def resolve_async_cache_hit(
    *,
    chat_cache: Any,
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    cache_answers: list[tuple[float, Any, Any, Any]],
    temperature: float,
    pre_store_data: Any,
    pre_embedding_data: Any,
    embedding_data: Any,
    session: Any,
    start_time: float,
    cache_stage_started_at: float | None,
    coalesce_key: str | None,
    cache_data_convert: Any,
) -> Any | None:
    """Return a converted async cache hit response when a verified hit is usable."""
    if not cache_answers:
        return None
    min_rank, max_rank = chat_cache.similarity_evaluation.range()
    return_message = await _time_cal_async(
        lambda: _post_process_cache_answers(
            chat_cache,
            cache_answers,
            original_question=pre_embedding_data,
            temperature=temperature,
        ),
        func_name="post_process",
        report_func=chat_cache.report.post,
    )
    if return_message is not None:
        return_message, cache_assessment = _repair_cached_answer(
            chat_cache,
            request_kwargs,
            return_message,
            context=context,
            task_policy=context.get("_byte_task_policy") or {},
        )
        if (
            cache_assessment is not None
            and cache_assessment.constraint != "freeform"
            and not cache_assessment.accepted
        ):
            _record_failure_memory(
                chat_cache, request_kwargs, context, reason="cache_revalidation_failed"
            )
            return_message = None
        if not _cache_reuse_allowed(chat_cache, request_kwargs, context, return_message):
            _record_failure_memory(
                chat_cache, request_kwargs, context, reason="unverified_code_answer"
            )
            return_message = None
    if return_message is None:
        return None

    chat_cache.report.hint_cache()
    if chat_cache.config.adaptive_threshold:
        get_adaptive_threshold(chat_cache).record(hit=True)
    model_name = request_kwargs.get("model", "unknown")
    get_budget_tracker(chat_cache).record_cache_hit(
        model_name,
        prompt_tokens=len(str(pre_embedding_data).split()) * 2,
        completion_tokens=len(str(return_message).split()) * 2,
    )
    _score_cache_hit_quality(
        chat_cache,
        cache_answers,
        query=pre_store_data,
        cached_answer=return_message,
        min_rank=min_rank,
        max_rank=max_rank,
    )
    cache_whole_data = next((item for item in cache_answers if item[1] == return_message), None)
    await _report_async_cache_hit(
        chat_cache,
        cache_whole_data,
        session=session,
        pre_embedding_data=pre_embedding_data,
        pre_store_data=pre_store_data,
        start_time=start_time,
    )
    _record_ai_memory(
        chat_cache,
        request_kwargs,
        context=context,
        answer=return_message,
        embedding_data=cache_whole_data[3].embedding_data if cache_whole_data else embedding_data,
        source="cache",
    )
    converted_response = cache_data_convert(return_message)
    if coalesce_key:
        get_coalescer(chat_cache).complete(coalesce_key, converted_response)
    _record_cache_stage_latency(
        chat_cache,
        request_kwargs,
        context,
        started_at=cache_stage_started_at,
        hit=True,
    )
    return converted_response


__all__ = ["resolve_async_cache_hit", "resolve_sync_cache_hit"]
