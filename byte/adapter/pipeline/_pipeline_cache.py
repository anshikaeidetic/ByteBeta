"""Cache lookup and cache-hit reuse stages for sync and async adapter flows."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from byte.adapter.runtime_state import get_adaptive_threshold
from byte.processor.post import LlmVerifier, temperature_softmax
from byte.utils.error import ByteErrorCode, CacheError
from byte.utils.log import byte_log
from byte.utils.time import time_cal

from ._pipeline_common import NO_RESULT, best_effort_log, complete_coalesced_request
from ._pipeline_state import CacheAnswerMatch, PipelineRunState
from .memory import (
    _record_ai_memory,
    _record_failure_memory,
    _resolve_similarity_threshold,
)
from .utils import (
    _await_with_report,
    _cache_reuse_allowed,
    _record_cache_stage_latency,
    _safe_log_value,
    _time_cal_async,
    acache_health_check,
    cache_health_check,
    get_budget_tracker,
    get_quality_scorer,
)
from .verifier import _repair_cached_answer

# ---------------------------------------------------------------------------
# New-paper helpers (lazy imports to avoid hard dependencies)
# ---------------------------------------------------------------------------

def _get_vcache_evaluator(state: PipelineRunState) -> Any:
    """Return the VCacheEvaluation instance if vcache is enabled, else None."""
    cfg = getattr(state.chat_cache, "config", None)
    if cfg is None or not getattr(cfg, "vcache_enabled", False):
        return None
    evaluator = getattr(state.chat_cache, "similarity_evaluation", None)
    if evaluator is None:
        return None
    # Support both direct VCacheEvaluation and wrapped evaluators
    if type(evaluator).__name__ == "VCacheEvaluation":
        return evaluator
    return None


def _vcache_update(state: PipelineRunState, search_data: Any, similarity: float, was_correct: bool) -> None:
    """Propagate correctness signal to the VCacheEvaluation if active."""
    vcache = _get_vcache_evaluator(state)
    if vcache is None:
        return
    try:
        question_id = int(search_data[1]) if search_data is not None else None
        if question_id is not None:
            vcache.update(question_id, similarity, was_correct)
    except Exception:  # pylint: disable=W0703
        pass


def lookup_cache_sync(state: PipelineRunState) -> Any:
    """Attempt a sync semantic-cache lookup and return a converted response on hit."""

    if not (state.cache_enable and not state.cache_skip):
        return NO_RESULT
    # Pass the raw query text so the LSH prefilter (arXiv 2503.05530) can probe
    # for near-duplicates before the full vector search.
    _lsh_text = str(state.pre_embedding_data or "") if isinstance(state.pre_embedding_data, str) else ""
    search_data_list = time_cal(
        state.chat_cache.data_manager.search,
        func_name="search",
        report_func=state.chat_cache.report.search,
    )(
        state.embedding_data,
        extra_param=state.context.get("search_func", None),
        top_k=_lookup_top_k(state),
        question_text=_lsh_text,
    )
    cache_answers = _rank_cache_answers_sync(state, search_data_list or [])
    if not cache_answers:
        return NO_RESULT
    return _materialize_cache_hit_sync(state, cache_answers)


async def lookup_cache_async(state: PipelineRunState) -> Any:
    """Attempt an async semantic-cache lookup and return a converted response on hit."""

    if not (state.cache_enable and not state.cache_skip):
        return NO_RESULT
    _lsh_text = str(state.pre_embedding_data or "") if isinstance(state.pre_embedding_data, str) else ""
    search_data_list = await _await_with_report(
        state.chat_cache.data_manager.asearch(
            state.embedding_data,
            extra_param=state.context.get("search_func", None),
            top_k=_lookup_top_k(state),
            question_text=_lsh_text,
        ),
        func_name="search",
        report_func=state.chat_cache.report.search,
    )
    cache_answers = await _rank_cache_answers_async(state, search_data_list or [])
    if not cache_answers:
        return NO_RESULT
    return await _materialize_cache_hit_async(state, cache_answers)


def record_cache_miss(state: PipelineRunState) -> None:
    """Record miss-path cache metrics after lookup completes."""

    _record_cache_stage_latency(
        state.chat_cache,
        state.kwargs,
        state.context,
        started_at=state.cache_stage_started_at,
        hit=False,
    )
    if state.chat_cache.config.adaptive_threshold:
        get_adaptive_threshold(state.chat_cache).record(hit=False)


def _lookup_top_k(state: PipelineRunState) -> int:
    if state.user_temperature and not state.user_top_k:
        return int(state.kwargs.pop("top_k", 5))
    return int(state.kwargs.pop("top_k", -1))


def _rank_cache_answers_sync(
    state: PipelineRunState,
    search_data_list: list[Any],
) -> list[CacheAnswerMatch]:
    similarity_threshold = _resolve_similarity_threshold(state.chat_cache)
    if similarity_threshold == 0.0:
        return []
    cfg = state.chat_cache.config
    min_rank, max_rank = state.chat_cache.similarity_evaluation.range()
    rank_threshold = _rank_threshold(min_rank, max_rank, similarity_threshold, state.cache_factor)

    # Dual-threshold reference lane bounds (arXiv 2601.11687)
    dual_mode = getattr(cfg, "dual_threshold_reference_mode", False)
    band_low = getattr(cfg, "llm_equivalence_ambiguity_band_low", 0.70)
    band_high = getattr(cfg, "llm_equivalence_ambiguity_band_high", 0.85)
    ref_low = _rank_threshold(min_rank, max_rank, band_low, state.cache_factor)
    ref_high = _rank_threshold(min_rank, max_rank, band_high, state.cache_factor)

    cache_answers: list[CacheAnswerMatch] = []
    for search_data in search_data_list:
        cache_data = time_cal(
            state.chat_cache.data_manager.get_scalar_data,
            func_name="get_data",
            report_func=state.chat_cache.report.data,
        )(
            search_data,
            extra_param=state.context.get("get_scalar_data", None),
            session=state.session,
        )
        if cache_data is None or _ttl_expired(cache_data, cfg.ttl):
            continue
        if cfg.data_check and not _is_cache_healthy_sync(state, cache_data, search_data):
            continue
        rank = time_cal(
            state.chat_cache.similarity_evaluation.evaluation,
            func_name="evaluation",
            report_func=state.chat_cache.report.evaluation,
        )(
            *_evaluation_payloads(state, cache_data, search_data),
            extra_param=state.context.get("evaluation_func", None),
        )
        _log_similarity(state, cache_data, rank)

        # Stage 1: Exact lane — above threshold, return from cache directly
        if rank_threshold <= rank:
            cache_answers.append(
                CacheAnswerMatch(
                    rank=float(rank),
                    answer=cache_data.answers[0].answer,
                    search_data=search_data,
                    cache_data=cache_data,
                )
            )
            state.chat_cache.data_manager.hit_cache_callback(search_data)
            _vcache_update(state, search_data, rank / max(max_rank - min_rank, 1.0), was_correct=True)

        # Stage 2: Reference lane — in ambiguity band, store hint for fresh model call
        elif dual_mode and ref_low <= rank < ref_high and "_byte_reference_hint" not in state.context:
            answer_text = cache_data.answers[0].answer if cache_data.answers else ""
            state.context["_byte_reference_hint"] = str(answer_text)[:800]
            byte_log.debug(
                "dual_threshold: reference lane activated (rank=%.3f), storing hint", rank
            )
            try:
                from byte.telemetry import (
                    bump_research_counter as _bump,  # pylint: disable=import-outside-toplevel
                )
                _bump("dual_threshold_reference_hits")
            except Exception:  # pragma: no cover - defensive
                pass
    return sorted(cache_answers, key=lambda item: item.rank, reverse=True)


async def _rank_cache_answers_async(
    state: PipelineRunState,
    search_data_list: list[Any],
) -> list[CacheAnswerMatch]:
    similarity_threshold = _resolve_similarity_threshold(state.chat_cache)
    if similarity_threshold == 0.0:
        return []
    cfg = state.chat_cache.config
    min_rank, max_rank = state.chat_cache.similarity_evaluation.range()
    rank_threshold = _rank_threshold(min_rank, max_rank, similarity_threshold, state.cache_factor)

    # Dual-threshold reference lane bounds (arXiv 2601.11687)
    dual_mode = getattr(cfg, "dual_threshold_reference_mode", False)
    band_low = getattr(cfg, "llm_equivalence_ambiguity_band_low", 0.70)
    band_high = getattr(cfg, "llm_equivalence_ambiguity_band_high", 0.85)
    ref_low = _rank_threshold(min_rank, max_rank, band_low, state.cache_factor)
    ref_high = _rank_threshold(min_rank, max_rank, band_high, state.cache_factor)

    cache_answers: list[CacheAnswerMatch] = []
    for search_data in search_data_list:
        cache_data = await _await_with_report(
            state.chat_cache.data_manager.aget_scalar_data(
                search_data,
                extra_param=state.context.get("get_scalar_data", None),
                session=state.session,
            ),
            func_name="get_data",
            report_func=state.chat_cache.report.data,
        )
        if cache_data is None or _ttl_expired(cache_data, cfg.ttl):
            continue
        if cfg.data_check and not await _is_cache_healthy_async(
            state,
            cache_data,
            search_data,
        ):
            continue
        rank = await _await_with_report(
            state.chat_cache.similarity_evaluation.aevaluation(
                *_evaluation_payloads(state, cache_data, search_data),
                extra_param=state.context.get("evaluation_func", None),
            ),
            func_name="evaluation",
            report_func=state.chat_cache.report.evaluation,
        )
        _log_similarity(state, cache_data, rank)

        # Stage 1: Exact lane — above threshold, return from cache directly
        if rank_threshold <= rank:
            cache_answers.append(
                CacheAnswerMatch(
                    rank=float(rank),
                    answer=cache_data.answers[0].answer,
                    search_data=search_data,
                    cache_data=cache_data,
                )
            )
            await state.chat_cache.data_manager.ahit_cache_callback(search_data)
            _vcache_update(state, search_data, rank / max(max_rank - min_rank, 1.0), was_correct=True)

        # Stage 2: Reference lane — in ambiguity band, store hint for fresh model call
        elif dual_mode and ref_low <= rank < ref_high and "_byte_reference_hint" not in state.context:
            answer_text = cache_data.answers[0].answer if cache_data.answers else ""
            state.context["_byte_reference_hint"] = str(answer_text)[:800]
            byte_log.debug(
                "dual_threshold: reference lane activated (rank=%.3f), storing hint", rank
            )
            try:
                from byte.telemetry import (
                    bump_research_counter as _bump,  # pylint: disable=import-outside-toplevel
                )
                _bump("dual_threshold_reference_hits")
            except Exception:  # pragma: no cover - defensive
                pass
    return sorted(cache_answers, key=lambda item: item.rank, reverse=True)


def _ttl_expired(cache_data: Any, ttl: float | None) -> bool:
    if ttl is None or not hasattr(cache_data, "create_on") or not cache_data.create_on:
        return False
    try:
        created = cache_data.create_on
        if not isinstance(created, datetime):
            return False
        age = (datetime.now() - created.replace(tzinfo=None)).total_seconds()
        if age > ttl:
            return True
    except (AttributeError, TypeError, ValueError):
        return False
    return False


def _is_cache_healthy_sync(state: PipelineRunState, cache_data: Any, search_data: Any) -> bool:
    return bool(
        cache_health_check(
        state.chat_cache.data_manager.v,
        {
            "embedding": cache_data.embedding_data,
            "search_result": search_data,
        },
        )
    )


async def _is_cache_healthy_async(state: PipelineRunState, cache_data: Any, search_data: Any) -> bool:
    return bool(
        await acache_health_check(
        state.chat_cache.data_manager.v,
        {
            "embedding": cache_data.embedding_data,
            "search_result": search_data,
        },
        )
    )


def _evaluation_payloads(state: PipelineRunState, cache_data: Any, search_data: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if "deps" in state.context and hasattr(cache_data.question, "deps"):
        return (
            {
                "question": state.context["deps"][0]["data"],
                "embedding": None,
            },
            {
                "question": cache_data.question.deps[0].data,
                "answer": cache_data.answers[0].answer,
                "search_result": search_data,
                "cache_data": cache_data,
                "embedding": None,
            },
        )
    return (
        {
            "question": state.pre_store_data,
            "embedding": state.embedding_data,
        },
        {
            "question": cache_data.question,
            "answer": cache_data.answers[0].answer,
            "search_result": search_data,
            "cache_data": cache_data,
            "embedding": cache_data.embedding_data,
        },
    )


def _rank_threshold(min_rank: float, max_rank: float, threshold: float, factor: float) -> float:
    rank_threshold = (max_rank - min_rank) * threshold * factor
    if rank_threshold > max_rank:
        return max_rank
    if rank_threshold < min_rank:
        return min_rank
    return rank_threshold


def _log_similarity(state: PipelineRunState, cache_data: Any, rank: float) -> None:
    byte_log.debug(
        "similarity: [user question] %s, [cache question] %s, [value] %f",
        _safe_log_value(state.chat_cache, state.pre_store_data),
        _safe_log_value(state.chat_cache, cache_data.question),
        rank,
    )


def _post_process_cache_answers(
    state: PipelineRunState, cache_answers: list[CacheAnswerMatch]
) -> Any:
    post_processor = state.chat_cache.post_process_messages_func
    if post_processor is temperature_softmax:
        return post_processor(
            messages=[item.answer for item in cache_answers],
            scores=[item.rank for item in cache_answers],
            temperature=state.temperature,
        )
    if isinstance(post_processor, LlmVerifier):
        return post_processor(
            messages=[item.answer for item in cache_answers],
            scores=[item.rank for item in cache_answers],
            original_question=state.pre_embedding_data,
        )
    return post_processor([item.answer for item in cache_answers])


def _materialize_cache_hit_sync(
    state: PipelineRunState,
    cache_answers: list[CacheAnswerMatch],
) -> Any:
    return_message = time_cal(
        lambda: _post_process_cache_answers(state, cache_answers),
        func_name="post_process",
        report_func=state.chat_cache.report.post,
    )()
    return _finalize_cache_hit_sync(state, cache_answers, return_message)


async def _materialize_cache_hit_async(
    state: PipelineRunState,
    cache_answers: list[CacheAnswerMatch],
) -> Any:
    return_message = await _time_cal_async(
        lambda: _post_process_cache_answers(state, cache_answers),
        func_name="post_process",
        report_func=state.chat_cache.report.post,
    )
    return await _finalize_cache_hit_async(state, cache_answers, return_message)


def _finalize_cache_hit_sync(
    state: PipelineRunState,
    cache_answers: list[CacheAnswerMatch],
    return_message: Any,
) -> Any:
    validated = _validated_cache_message(state, return_message)
    if validated is None:
        return NO_RESULT
    _record_cache_hit_metrics(state, cache_answers, validated)
    cache_match = next((item for item in cache_answers if item.answer == validated), None)
    if state.session is not None and cache_match is not None:
        state.chat_cache.data_manager.add_session(
            cache_match.search_data,
            state.session.name,
            state.pre_embedding_data,
        )
    _report_cache_hit_sync(state, cache_match)
    _record_ai_memory(
        state.chat_cache,
        state.kwargs,
        context=state.context,
        answer=validated,
        embedding_data=cache_match.cache_data.embedding_data if cache_match is not None else state.embedding_data,
        source="cache",
    )
    converted_response = state.cache_data_convert(validated)
    complete_coalesced_request(state, converted_response)
    _record_cache_stage_latency(
        state.chat_cache,
        state.kwargs,
        state.context,
        started_at=state.cache_stage_started_at,
        hit=True,
    )
    return converted_response


async def _finalize_cache_hit_async(
    state: PipelineRunState,
    cache_answers: list[CacheAnswerMatch],
    return_message: Any,
) -> Any:
    validated = _validated_cache_message(state, return_message)
    if validated is None:
        return NO_RESULT
    _record_cache_hit_metrics(state, cache_answers, validated)
    cache_match = next((item for item in cache_answers if item.answer == validated), None)
    if state.session is not None and cache_match is not None:
        await state.chat_cache.data_manager.aadd_session(
            cache_match.search_data,
            state.session.name,
            state.pre_embedding_data,
        )
    await _report_cache_hit_async(state, cache_match)
    _record_ai_memory(
        state.chat_cache,
        state.kwargs,
        context=state.context,
        answer=validated,
        embedding_data=cache_match.cache_data.embedding_data if cache_match is not None else state.embedding_data,
        source="cache",
    )
    converted_response = state.cache_data_convert(validated)
    complete_coalesced_request(state, converted_response)
    _record_cache_stage_latency(
        state.chat_cache,
        state.kwargs,
        state.context,
        started_at=state.cache_stage_started_at,
        hit=True,
    )
    return converted_response


def _validated_cache_message(state: PipelineRunState, return_message: Any) -> Any:
    if return_message is None:
        return None
    return_message, cache_assessment = _repair_cached_answer(
        state.chat_cache,
        state.kwargs,
        return_message,
        context=state.context,
        task_policy=state.context.get("_byte_task_policy") or {},
    )
    if (
        cache_assessment is not None
        and cache_assessment.constraint != "freeform"
        and not cache_assessment.accepted
    ):
        _record_failure_memory(
            state.chat_cache,
            state.kwargs,
            state.context,
            reason="cache_revalidation_failed",
        )
        return None
    if not _cache_reuse_allowed(state.chat_cache, state.kwargs, state.context, return_message):
        _record_failure_memory(
            state.chat_cache,
            state.kwargs,
            state.context,
            reason="unverified_code_answer",
        )
        return None
    return return_message


def _record_cache_hit_metrics(
    state: PipelineRunState,
    cache_answers: list[CacheAnswerMatch],
    return_message: Any,
) -> None:
    state.chat_cache.report.hint_cache()
    if state.chat_cache.config.adaptive_threshold:
        get_adaptive_threshold(state.chat_cache).record(hit=True)
    model_name = state.kwargs.get("model", "unknown")
    get_budget_tracker(state.chat_cache).record_cache_hit(
        model_name,
        prompt_tokens=len(str(state.pre_embedding_data).split()) * 2,
        completion_tokens=len(str(return_message).split()) * 2,
    )
    best_rank = cache_answers[0].rank if cache_answers else 0.0
    min_rank, max_rank = state.chat_cache.similarity_evaluation.range()
    factor = max_rank - min_rank
    sim_score = best_rank / factor if factor else best_rank
    try:
        get_quality_scorer(state.chat_cache).score(
            query=str(state.pre_store_data),
            cached_answer=str(return_message),
            similarity_score=sim_score,
        )
    except (AttributeError, CacheError, KeyError, TypeError, ValueError) as exc:
        best_effort_log(
            "failed to score cache hit quality",
            error=exc,
            code=ByteErrorCode.PIPELINE_CACHE_LOOKUP,
            boundary="pipeline.cache_lookup",
            stage="quality_score",
        )


def _report_cache_hit_sync(state: PipelineRunState, cache_match: CacheAnswerMatch | None) -> None:
    if cache_match is None or state.chat_cache.config.disable_report:
        return
    report_cache_data = cache_match.cache_data
    report_search_data = cache_match.search_data
    state.chat_cache.data_manager.report_cache(
        state.pre_store_data if isinstance(state.pre_store_data, str) else "",
        report_cache_data.question if isinstance(report_cache_data.question, str) else "",
        report_search_data[1],
        report_cache_data.answers[0].answer
        if isinstance(report_cache_data.answers[0].answer, str)
        else "",
        cache_match.rank,
        round(time.time() - state.start_time, 6),
    )


async def _report_cache_hit_async(state: PipelineRunState, cache_match: CacheAnswerMatch | None) -> None:
    if cache_match is None or state.chat_cache.config.disable_report:
        return
    report_cache_data = cache_match.cache_data
    report_search_data = cache_match.search_data
    await state.chat_cache.data_manager.areport_cache(
        state.pre_store_data if isinstance(state.pre_store_data, str) else "",
        report_cache_data.question if isinstance(report_cache_data.question, str) else "",
        report_search_data[1],
        report_cache_data.answers[0].answer
        if isinstance(report_cache_data.answers[0].answer, str)
        else "",
        cache_match.rank,
        round(time.time() - state.start_time, 6),
    )


__all__ = ["lookup_cache_async", "lookup_cache_sync", "record_cache_miss"]
