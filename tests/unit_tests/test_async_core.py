import pytest

from byte import Cache
from byte.manager.data_manager import DataManager
from byte.manager.scalar_data.base import CacheData
from byte.similarity_evaluation.similarity_evaluation import SimilarityEvaluation


class RecordingDataManager(DataManager):
    def __init__(self) -> None:
        self.calls = []
        self.data = {"entry": ("question", "answer", "embedding", set(), None)}

    def save(self, question, answer, embedding_data, **kwargs) -> None:
        self.calls.append(("save", question, answer, embedding_data, kwargs))

    def import_data(self, questions, answers, embedding_datas, session_ids) -> None:
        self.calls.append(("import_data", questions, answers, embedding_datas, session_ids))

    def get_scalar_data(self, res_data, **kwargs) -> object:
        self.calls.append(("get_scalar_data", res_data, kwargs))
        return CacheData(question="q", answers=["a"])

    def search(self, embedding_data, **kwargs) -> object:
        self.calls.append(("search", embedding_data, kwargs))
        return []

    def invalidate_by_query(self, query, *, embedding_func=None) -> object:
        self.calls.append(("invalidate_by_query", query))
        return False

    def flush(self) -> None:
        self.calls.append(("flush",))

    def add_session(self, res_data, session_id, pre_embedding_data) -> None:
        self.calls.append(("add_session", res_data, session_id, pre_embedding_data))

    def list_sessions(self, session_id, key) -> object:
        self.calls.append(("list_sessions", session_id, key))
        return []

    def delete_session(self, session_id) -> None:
        self.calls.append(("delete_session", session_id))

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
                "report_cache",
                user_question,
                cache_question,
                cache_question_id,
                cache_answer,
                similarity_value,
                cache_delta_time,
            )
        )

    def close(self) -> None:
        self.calls.append(("close",))


class RecordingSimilarity(SimilarityEvaluation):
    def __init__(self) -> None:
        self.calls = []

    def evaluation(self, src_dict, cache_dict, **kwargs) -> object:
        self.calls.append((src_dict, cache_dict, kwargs))
        return 0.5

    def range(self) -> object:
        return (0.0, 1.0)


@pytest.mark.asyncio
async def test_data_manager_async_wrappers_delegate_to_sync_methods() -> None:
    manager = RecordingDataManager()

    await manager.asave("q", "a", "emb", source="unit")
    await manager.aimport_data(["q"], ["a"], ["emb"], [None])
    await manager.aget_scalar_data(("q", "a"))
    await manager.asearch("emb", top_k=1)
    await manager.aflush()
    await manager.aadd_session(("q", "a"), "session-1", "emb")
    await manager.alist_sessions("session-1", "key")
    await manager.adelete_session("session-1")
    await manager.areport_cache("u", "c", "id", "a", 1.0, 0.01)
    await manager.aclose()

    call_names = [item[0] for item in manager.calls]
    assert call_names == [
        "save",
        "import_data",
        "get_scalar_data",
        "search",
        "flush",
        "add_session",
        "list_sessions",
        "delete_session",
        "report_cache",
        "close",
    ]


@pytest.mark.asyncio
async def test_cache_async_methods_use_core_wrappers() -> None:
    manager = RecordingDataManager()
    cache_obj = Cache()
    cache_obj.init(
        data_manager=manager,
        embedding_func=lambda question, **_: f"emb::{question}",
    )

    await cache_obj.aimport_data(["hello"], ["world"])
    await cache_obj.aflush()
    await cache_obj.aclear()
    await cache_obj.aclose()

    import_call = next(item for item in manager.calls if item[0] == "import_data")
    assert import_call[1] == ["hello"]
    assert import_call[2] == ["world"]
    assert import_call[3] == ["emb::hello"]
    assert manager.data == {}
    assert ("flush",) in manager.calls
    assert ("close",) in manager.calls


@pytest.mark.asyncio
async def test_similarity_evaluation_async_wrapper_delegates() -> None:
    evaluation = RecordingSimilarity()

    score = await evaluation.aevaluation({"question": "q"}, {"question": "q"})

    assert score == 0.5
    assert len(evaluation.calls) == 1
