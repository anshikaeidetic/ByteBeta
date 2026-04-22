import hashlib
from dataclasses import dataclass

import numpy as np
import pytest

from byte import Cache, Config
from byte._backends import openai
from byte.manager.data_manager import SSDataManager
from byte.manager.scalar_data.base import CacheStorage
from byte.manager.vector_data.base import VectorBase
from byte.processor.pre import last_content
from byte.similarity_evaluation.distance import SearchDistanceEvaluation
from byte.utils.response import (
    get_message_from_openai_answer,
    get_stream_message_from_openai_answer,
)

pytestmark = pytest.mark.integration_mocked


@dataclass
class _SessionRecord:
    id: int
    session_id: str
    session_question: str


class InMemoryScalarStorage(CacheStorage):
    def __init__(self) -> None:
        self._data = {}
        self._deleted = set()
        self._sessions = []
        self._next_id = 1

    def create(self) -> None:
        return None

    def batch_insert(self, all_data) -> object:
        ids = []
        for cache_data in all_data:
            data_id = self._next_id
            self._next_id += 1
            self._data[data_id] = cache_data
            ids.append(data_id)
        return ids

    def get_data_by_id(self, key) -> object:
        if key in self._deleted:
            return None
        return self._data.get(key)

    def mark_deleted(self, keys) -> None:
        self._deleted.update(keys)

    def clear_deleted_data(self) -> None:
        for key in list(self._deleted):
            self._data.pop(key, None)
        self._deleted.clear()

    def get_ids(self, deleted=True) -> object:
        if deleted:
            return list(self._data.keys())
        return [key for key in self._data if key not in self._deleted]

    def count(self, state=0, is_all=False) -> object:
        return len(self.get_ids(deleted=is_all))

    def flush(self) -> None:
        return None

    def add_session(self, question_id, session_id, session_question) -> None:
        self._sessions.append(_SessionRecord(question_id, session_id, session_question))

    def list_sessions(self, session_id, key) -> object:
        if key is not None:
            return [record for record in self._sessions if record.id == key]
        if session_id is not None:
            return [record for record in self._sessions if record.session_id == session_id]
        return list(self._sessions)

    def delete_session(self, keys) -> None:
        self._sessions = [record for record in self._sessions if record.id not in keys]

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ) -> None:
        return None

    def close(self) -> None:
        return None


class InMemoryVectorStore(VectorBase):
    def __init__(self) -> None:
        self._vectors = {}

    def mul_add(self, datas) -> None:
        for item in datas:
            self._vectors[item.id] = np.asarray(item.data, dtype=np.float32)

    def search(self, data, top_k) -> object:
        query = np.asarray(data, dtype=np.float32)
        results = []
        for data_id, vector in self._vectors.items():
            distance = float(np.linalg.norm(query - vector))
            results.append((distance, data_id))
        results.sort(key=lambda item: item[0])
        if top_k is None or top_k < 0:
            return results
        return results[:top_k]

    def rebuild(self, ids=None) -> object:
        return True

    def delete(self, ids) -> object:
        for data_id in ids:
            self._vectors.pop(data_id, None)
        return True

    def get_embeddings(self, data_id) -> object:
        return self._vectors.get(data_id)

    def update_embeddings(self, data_id, emb) -> None:
        self._vectors[data_id] = np.asarray(emb, dtype=np.float32)


def _embedding(text, **kwargs) -> object:
    del kwargs
    digest = hashlib.sha256(str(text or "").encode("utf-8")).digest()
    vector = np.asarray(list(digest), dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def _build_cache() -> object:
    cache_obj = Cache()
    cache_obj.init(
        pre_embedding_func=last_content,
        embedding_func=_embedding,
        data_manager=SSDataManager(
            InMemoryScalarStorage(),
            InMemoryVectorStore(),
            None,
            None,
            max_size=128,
            clean_size=32,
        ),
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(enable_token_counter=False, similarity_threshold=0.95),
    )
    return cache_obj


def _get_stream_text(response) -> object:
    return "".join(get_stream_message_from_openai_answer(chunk) for chunk in response)


def test_mocked_service_tier_returns_cached_answer_without_upstream_llm(monkeypatch) -> None:
    cache_obj = _build_cache()
    question = "What is Byte?"
    answer = "Byte is an LLM runtime optimization layer."
    cache_obj.data_manager.save(question, answer, cache_obj.embedding_func(question))

    def fail_llm(*args, **kwargs) -> None:
        raise AssertionError("mocked service tier unexpectedly called the upstream llm")

    monkeypatch.setattr(openai.ChatCompletion, "llm", fail_llm, raising=False)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        cache_obj=cache_obj,
    )

    assert response.get("byte") is True
    assert get_message_from_openai_answer(response) == answer


def test_mocked_service_tier_self_heals_out_of_sync_vector_entry() -> None:
    cache_obj = _build_cache()
    stored_question = "what is apple?"
    stored_answer = "apple"
    cache_obj.data_manager.save(
        stored_question,
        stored_answer,
        cache_obj.embedding_func(stored_question),
    )

    trouble_query = "what is google?"
    cache_obj.data_manager.v.update_embeddings(1, cache_obj.embedding_func(trouble_query))

    incorrect = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": trouble_query}],
        search_only=True,
        stream=True,
        cache_obj=cache_obj,
    )
    assert _get_stream_text(incorrect) == stored_answer

    cache_obj.config.data_check = True
    assert (
        openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": trouble_query}],
            search_only=True,
            stream=True,
            cache_obj=cache_obj,
        )
        is None
    )

    cache_obj.config.data_check = False
    assert (
        openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": trouble_query}],
            search_only=True,
            stream=True,
            cache_obj=cache_obj,
        )
        is None
    )

    healed = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": stored_question}],
        search_only=True,
        stream=True,
        cache_obj=cache_obj,
    )
    assert _get_stream_text(healed) == stored_answer
