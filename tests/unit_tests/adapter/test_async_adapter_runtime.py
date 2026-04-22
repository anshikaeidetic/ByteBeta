import pytest

from byte import Cache, Config
from byte.adapter.adapter import aadapt
from byte.manager.data_manager import DataManager
from byte.manager.scalar_data.base import CacheData
from byte.similarity_evaluation.similarity_evaluation import SimilarityEvaluation


class AsyncOnlySimilarity(SimilarityEvaluation):
    def __init__(self) -> None:
        self.calls = []

    def evaluation(self, src_dict, cache_dict, **kwargs) -> object:
        self.calls.append("sync_evaluation")
        return 1.0

    async def aevaluation(self, src_dict, cache_dict, **kwargs) -> object:
        self.calls.append("aevaluation")
        return 1.0

    def range(self) -> object:
        return (0.0, 1.0)


class AsyncTrackingDataManager(DataManager):
    def __init__(self, *, search_results=None, cache_data=None) -> None:
        self.calls = []
        self.search_results = list(search_results or [])
        self.cache_data = cache_data

    def save(self, question, answer, embedding_data, **kwargs) -> None:
        self.calls.append(("sync_save", question, answer, embedding_data, kwargs))

    def import_data(self, questions, answers, embedding_datas, session_ids) -> None:
        self.calls.append(("sync_import_data", questions, answers, embedding_datas, session_ids))

    def get_scalar_data(self, res_data, **kwargs) -> object:
        self.calls.append(("sync_get_scalar_data", res_data, kwargs))
        return self.cache_data

    def search(self, embedding_data, **kwargs) -> object:
        self.calls.append(("sync_search", embedding_data, kwargs))
        return list(self.search_results)

    def invalidate_by_query(self, query, *, embedding_func=None) -> object:
        self.calls.append(("sync_invalidate_by_query", query))
        return False

    def flush(self) -> None:
        self.calls.append(("sync_flush",))

    def add_session(self, res_data, session_id, pre_embedding_data) -> None:
        self.calls.append(("sync_add_session", res_data, session_id, pre_embedding_data))

    def list_sessions(self, session_id, key) -> object:
        self.calls.append(("sync_list_sessions", session_id, key))
        return []

    def delete_session(self, session_id) -> None:
        self.calls.append(("sync_delete_session", session_id))

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ) -> None:
        self.calls.append(
            (
                "sync_report_cache",
                user_question,
                cache_question,
                cache_question_id,
                cache_answer,
                similarity_value,
                cache_delta_time,
            )
        )

    def close(self) -> None:
        self.calls.append(("sync_close",))

    async def asearch(self, embedding_data, **kwargs) -> object:
        self.calls.append(("asearch", embedding_data, kwargs))
        return list(self.search_results)

    async def aget_scalar_data(self, res_data, **kwargs) -> object:
        self.calls.append(("aget_scalar_data", res_data, kwargs))
        return self.cache_data

    async def ahit_cache_callback(self, res_data, **kwargs) -> None:
        self.calls.append(("ahit_cache_callback", res_data, kwargs))

    async def aadd_session(self, res_data, session_id, pre_embedding_data) -> None:
        self.calls.append(("aadd_session", res_data, session_id, pre_embedding_data))

    async def areport_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ) -> None:
        self.calls.append(
            (
                "areport_cache",
                user_question,
                cache_question,
                cache_question_id,
                cache_answer,
                similarity_value,
                cache_delta_time,
            )
        )

    async def asave(self, question, answer, embedding_data, **kwargs) -> None:
        self.calls.append(("asave", question, answer, embedding_data, kwargs))


def _build_cache(data_manager, similarity) -> object:
    cache_obj = Cache()
    cache_obj.init(
        data_manager=data_manager,
        similarity_evaluation=similarity,
        pre_embedding_func=lambda request_kwargs, **_: request_kwargs["messages"][-1]["content"],
        embedding_func=lambda content, **_: f"emb::{content}",
        config=Config(
            context_compiler=False,
            response_repair=False,
            reasoning_reuse=False,
        ),
    )
    return cache_obj


def _cache_data_convert(cache_data) -> object:
    return {"choices": [{"message": {"content": cache_data}}], "byte": True}


def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> object:
    update_cache_func(llm_data["choices"][0]["message"]["content"])
    return llm_data


@pytest.mark.asyncio
async def test_aadapt_uses_async_cache_search_path() -> None:
    cache_data = CacheData(
        question="cached question", answers=["cached answer"], embedding_data="emb::cached"
    )
    manager = AsyncTrackingDataManager(search_results=[(1.0, "cache-id")], cache_data=cache_data)
    similarity = AsyncOnlySimilarity()
    cache_obj = _build_cache(manager, similarity)

    async def llm_handler(*args, **kwargs) -> None:
        raise AssertionError("LLM should not run on cache hit")

    result = await aadapt(
        llm_handler,
        _cache_data_convert,
        _update_cache_callback,
        cache_obj=cache_obj,
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "Describe cache reuse."}],
    )

    assert result["choices"][0]["message"]["content"] == "cached answer"
    assert any(call[0] == "asearch" for call in manager.calls)
    assert any(call[0] == "aget_scalar_data" for call in manager.calls)
    assert any(call[0] == "ahit_cache_callback" for call in manager.calls)
    assert "aevaluation" in similarity.calls
    assert not any(call[0].startswith("sync_") for call in manager.calls)
    assert "sync_evaluation" not in similarity.calls


@pytest.mark.asyncio
async def test_aadapt_uses_async_cache_save_on_miss() -> object:
    manager = AsyncTrackingDataManager()
    similarity = AsyncOnlySimilarity()
    cache_obj = _build_cache(manager, similarity)

    async def llm_handler(*args, **kwargs) -> object:
        return {
            "choices": [{"message": {"content": "live answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

    result = await aadapt(
        llm_handler,
        _cache_data_convert,
        _update_cache_callback,
        cache_obj=cache_obj,
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "Tell me something new."}],
    )

    assert result["choices"][0]["message"]["content"] == "live answer"
    assert any(call[0] == "asearch" for call in manager.calls)
    assert any(call[0] == "asave" for call in manager.calls)
    assert not any(call[0] in {"sync_search", "sync_save"} for call in manager.calls)
