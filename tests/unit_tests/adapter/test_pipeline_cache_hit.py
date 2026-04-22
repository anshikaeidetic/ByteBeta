from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from byte.adapter.pipeline import _cache_hit


def _chat_cache() -> object:
    return SimpleNamespace(
        config=SimpleNamespace(adaptive_threshold=False, disable_report=True),
        similarity_evaluation=SimpleNamespace(range=lambda: (0.0, 1.0)),
        report=SimpleNamespace(post=MagicMock(), hint_cache=MagicMock()),
        post_process_messages_func=lambda messages: messages[0],
        data_manager=SimpleNamespace(add_session=MagicMock(), report_cache=MagicMock()),
    )


def _cache_answers(answer="cached answer") -> object:
    cache_data = SimpleNamespace(
        question="cached question",
        answers=[SimpleNamespace(answer=answer)],
        embedding_data="cached-embedding",
    )
    return [(0.92, answer, ("row", 1), cache_data)]


def test_resolve_sync_cache_hit_returns_converted_response_for_verified_hit() -> None:
    chat_cache = _chat_cache()
    budget_tracker = MagicMock()
    quality_scorer = MagicMock()

    with patch.object(
        _cache_hit,
        "_repair_cached_answer",
        return_value=("patched answer", SimpleNamespace(constraint="freeform", accepted=True)),
    ), patch.object(_cache_hit, "_cache_reuse_allowed", return_value=True), patch.object(
        _cache_hit, "get_budget_tracker", return_value=budget_tracker
    ), patch.object(
        _cache_hit, "get_quality_scorer", return_value=quality_scorer
    ), patch.object(
        _cache_hit, "_record_ai_memory"
    ) as record_ai_memory:
        response = _cache_hit.resolve_sync_cache_hit(
            chat_cache=chat_cache,
            request_kwargs={"model": "gpt-4o-mini"},
            context={},
            cache_answers=_cache_answers("raw answer"),
            temperature=0.0,
            pre_store_data="prompt",
            pre_embedding_data="prompt",
            embedding_data="live-embedding",
            session=None,
            start_time=0.0,
            cache_stage_started_at=0.0,
            coalesce_key=None,
            cache_data_convert=lambda answer: {"answer": answer},
        )

    assert response == {"answer": "patched answer"}
    chat_cache.report.hint_cache.assert_called_once()
    budget_tracker.record_cache_hit.assert_called_once()
    quality_scorer.score.assert_called_once()
    record_ai_memory.assert_called_once()


def test_resolve_sync_cache_hit_rejects_unverified_reuse_and_records_failure() -> None:
    chat_cache = _chat_cache()

    with patch.object(
        _cache_hit,
        "_repair_cached_answer",
        return_value=("cached answer", SimpleNamespace(constraint="freeform", accepted=True)),
    ), patch.object(_cache_hit, "_cache_reuse_allowed", return_value=False), patch.object(
        _cache_hit, "_record_failure_memory"
    ) as record_failure_memory:
        response = _cache_hit.resolve_sync_cache_hit(
            chat_cache=chat_cache,
            request_kwargs={"model": "gpt-4o-mini"},
            context={},
            cache_answers=_cache_answers(),
            temperature=0.0,
            pre_store_data="prompt",
            pre_embedding_data="prompt",
            embedding_data="live-embedding",
            session=None,
            start_time=0.0,
            cache_stage_started_at=0.0,
            coalesce_key=None,
            cache_data_convert=lambda answer: {"answer": answer},
        )

    assert response is None
    record_failure_memory.assert_called_once()
    assert record_failure_memory.call_args.kwargs["reason"] == "unverified_code_answer"
