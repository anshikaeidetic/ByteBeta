from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from datetime import datetime
from threading import RLock
from typing import Any

import cachetools
import numpy as np
import requests

from byte.manager.eviction import EvictionBase
from byte.manager.eviction.distributed_cache import NoOpEviction
from byte.manager.eviction_manager import EvictionManager
from byte.manager.object_data.base import ObjectBase
from byte.manager.scalar_data.base import (
    Answer,
    CacheData,
    CacheStorage,
    DataType,
    Question,
    QuestionDep,
)
from byte.manager.vector_data.base import VectorBase, VectorData
from byte.utils.async_ops import run_sync
from byte.utils.error import CacheError, ParamError
from byte.utils.log import byte_log
from byte.utils.safe_pickle import dump_map_cache, load_map_cache


class DataManager(metaclass=ABCMeta):
    """DataManager manage the cache data, including save and search."""

    @abstractmethod
    def save(self, question, answer, embedding_data, **kwargs) -> None:
        pass

    @abstractmethod
    def import_data(
        self,
        questions: list[Any],
        answers: list[Any],
        embedding_datas: list[Any],
        session_ids: list[str | None],
    ) -> None:
        pass

    @abstractmethod
    def get_scalar_data(self, res_data, **kwargs) -> CacheData:
        pass

    def hit_cache_callback(self, res_data, **kwargs) -> None:
        return None

    @abstractmethod
    def search(self, embedding_data, **kwargs) -> None:
        """Search cache data according to the embedding data."""

    @abstractmethod
    def invalidate_by_query(self, query: str, *, embedding_func=None) -> bool:
        """Delete the entry whose original query matches *query*.

        Concrete implementations must provide this method to avoid silent
        failures or duck-typing fallbacks when queries need invalidation.
        """

    def flush(self) -> None:
        return None

    async def asave(self, question, answer, embedding_data, **kwargs) -> Any:
        return await run_sync(self.save, question, answer, embedding_data, **kwargs)

    async def aimport_data(
        self,
        questions: list[Any],
        answers: list[Any],
        embedding_datas: list[Any],
        session_ids: list[str | None],
    ) -> Any:
        return await run_sync(
            self.import_data,
            questions=questions,
            answers=answers,
            embedding_datas=embedding_datas,
            session_ids=session_ids,
        )

    async def aget_scalar_data(self, res_data, **kwargs) -> CacheData:
        return await run_sync(self.get_scalar_data, res_data, **kwargs)

    async def ahit_cache_callback(self, res_data, **kwargs) -> Any:
        return await run_sync(self.hit_cache_callback, res_data, **kwargs)

    async def asearch(self, embedding_data, **kwargs) -> Any:
        return await run_sync(self.search, embedding_data, **kwargs)

    async def ainvalidate_by_query(self, query: str, *, embedding_func=None) -> bool:
        return await run_sync(self.invalidate_by_query, query, embedding_func=embedding_func)

    async def aflush(self) -> Any:
        return await run_sync(self.flush)

    @abstractmethod
    def add_session(self, res_data, session_id, pre_embedding_data) -> None:
        pass

    async def aadd_session(self, res_data, session_id, pre_embedding_data) -> Any:
        return await run_sync(self.add_session, res_data, session_id, pre_embedding_data)

    @abstractmethod
    def list_sessions(self, session_id, key) -> None:
        pass

    async def alist_sessions(self, session_id, key) -> Any:
        return await run_sync(self.list_sessions, session_id, key)

    @abstractmethod
    def delete_session(self, session_id) -> None:
        pass

    async def adelete_session(self, session_id) -> Any:
        return await run_sync(self.delete_session, session_id)

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

    async def areport_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ) -> Any:
        return await run_sync(
            self.report_cache,
            user_question,
            cache_question,
            cache_question_id,
            cache_answer,
            similarity_value,
            cache_delta_time,
        )

    def close(self) -> Any:
        return self.flush()

    async def aclose(self) -> Any:
        return await run_sync(self.close)


class MapDataManager(DataManager):
    """MapDataManager stores all data in a map data structure."""

    def __init__(self, data_path, max_size, get_data_container=None) -> None:
        if get_data_container is None:
            self.data = cachetools.LRUCache(max_size)
        else:
            self.data = get_data_container(max_size)
        self.data_path = data_path
        self._lock = RLock()
        self.init()

    def init(self) -> None:
        try:
            with open(self.data_path, "rb") as handle:
                loaded = load_map_cache(
                    handle,
                    safe_globals=_SAFE_PICKLE_GLOBALS,
                    max_size=getattr(self.data, "maxsize", None),
                )
            with self._lock:
                self.data = loaded
        except FileNotFoundError:
            return
        except PermissionError as exc:
            raise CacheError(
                f"You don't have permission to access this file <{self.data_path}>."
            ) from exc

    @staticmethod
    def _emb_key(embedding_data) -> Any:
        if hasattr(embedding_data, "tobytes"):
            return embedding_data.tobytes()
        return embedding_data

    def save(self, question, answer, embedding_data, **kwargs) -> None:
        if isinstance(question, Question):
            question = question.content
        session = kwargs.get("session")
        session_id = {session.name} if session else set()
        with self._lock:
            self.data[self._emb_key(embedding_data)] = (
                question,
                answer,
                embedding_data,
                session_id,
                datetime.now(),
            )
        # LSH prefilter : index the stored question for future near-dup probes.
        try:
            from byte.manager.lsh_prefilter import (
                get_lsh_prefilter,  # pylint: disable=import-outside-toplevel
            )
            lsh = get_lsh_prefilter()
            if lsh is not None and isinstance(question, str) and question:
                lsh.index(self._emb_key(embedding_data), question)
        except Exception:  # pragma: no cover - defensive
            pass

    def import_data(
        self,
        questions: list[Any],
        answers: list[Any],
        embedding_datas: list[Any],
        session_ids: list[str | None],
    ) -> None:
        if (
            len(questions) != len(answers)
            or len(questions) != len(embedding_datas)
            or len(questions) != len(session_ids)
        ):
            raise ParamError("Make sure that all parameters have the same length")
        with self._lock:
            for i, embedding_data in enumerate(embedding_datas):
                self.data[self._emb_key(embedding_data)] = (
                    questions[i],
                    answers[i],
                    embedding_datas[i],
                    {session_ids[i]} if session_ids[i] else set(),
                    datetime.now(),
                )

    def get_scalar_data(self, res_data, **kwargs) -> CacheData:
        session = kwargs.get("session")
        if session:
            answer = res_data[1].answer if isinstance(res_data[1], Answer) else res_data[1]
            if not session.check_hit_func(session.name, list(res_data[3]), [res_data[0]], answer):
                return None

        create_on = res_data[4] if len(res_data) > 4 else None
        return CacheData(question=res_data[0], answers=res_data[1], create_on=create_on)

    def hit_cache_callback(self, res_data, **kwargs) -> None:
        return None

    def search(self, embedding_data, **kwargs) -> list[Any]:
        # LSH prefilter : probe LSH when the caller provides text.
        text = kwargs.pop("question_text", "") or ""
        if text:
            try:
                from byte.manager.lsh_prefilter import (
                    get_lsh_prefilter,  # pylint: disable=import-outside-toplevel
                )
                lsh = get_lsh_prefilter()
                if lsh is not None:
                    lsh.query(text)
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            with self._lock:
                question, answer, emb, session_ids, created_at = self.data[
                    self._emb_key(embedding_data)
                ]
            return [(question, answer, emb, set(session_ids), created_at)]
        except (KeyError, TypeError):
            return []

    def invalidate_by_query(self, query: str, *, embedding_func=None) -> bool:
        with self._lock:
            for key, value in list(self.data.items()):
                if value[0] == query:
                    del self.data[key]
                    return True
        return False

    def flush(self) -> None:
        try:
            with self._lock:
                snapshot = cachetools.LRUCache(
                    getattr(self.data, "maxsize", max(len(self.data), 1))
                )
                snapshot.update(self.data)
            with open(self.data_path, "wb") as handle:
                dump_map_cache(handle, snapshot)
        except PermissionError:
            byte_log.error("You don't have permission to access this file %s.", self.data_path)

    def add_session(self, res_data, session_id, pre_embedding_data) -> None:
        with self._lock:
            key = self._emb_key(pre_embedding_data)
            if key in self.data:
                self.data[key][3].add(session_id)
                return
        res_data[3].add(session_id)

    def list_sessions(self, session_id=None, key=None) -> list[Any]:
        session_ids = set()
        with self._lock:
            items = list(self.data.items())
        for key_name, value in items:
            if session_id and session_id in value[3]:
                session_ids.add(key_name)
            elif len(value[3]) > 0:
                session_ids.update(value[3])
        return list(session_ids)

    def delete_session(self, session_id) -> None:
        keys = self.list_sessions(session_id=session_id)
        with self._lock:
            for key_name in keys:
                if key_name not in self.data:
                    continue
                self.data[key_name][3].discard(session_id)
                if len(self.data[key_name][3]) == 0:
                    del self.data[key_name]

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
        self.flush()


def normalize(vec) -> Any:
    normalized_v = np.asarray(vec, dtype="float32")
    magnitude = np.linalg.norm(normalized_v)
    if magnitude == 0:
        return normalized_v
    return normalized_v / magnitude


class SSDataManager(DataManager):
    """Generate SSDataManager to manage scalar and vector data."""

    def __init__(
        self,
        s: CacheStorage,
        v: VectorBase,
        o: ObjectBase | None,
        e: EvictionBase | None,
        max_size,
        clean_size,
        policy="LRU",
    ) -> None:
        self.s = s
        self.v = v
        self.o = o
        self.eviction_manager = EvictionManager(self.s, self.v)
        if e is None:
            e = EvictionBase(
                name="memory",
                maxsize=max_size,
                clean_size=clean_size,
                policy=policy,
                on_evict=self._clear,
            )
        self.eviction_base = e

        if not isinstance(self.eviction_base, NoOpEviction):
            self.eviction_base.put(self.s.get_ids(deleted=False))

    def _clear(self, marked_keys) -> None:
        self.eviction_manager.soft_evict(marked_keys)
        if self.eviction_manager.check_evict():
            self.eviction_manager.delete()

    def save(self, question, answer, embedding_data, **kwargs) -> None:
        session = kwargs.get("session")
        session_id = session.name if session else None
        self.import_data([question], [answer], [embedding_data], [session_id])

    def _process_answer_data(self, answers: Answer | list[Answer]) -> Any:
        if isinstance(answers, Answer):
            answers = [answers]
        new_ans = []
        for ans in answers:
            if ans.answer_type != DataType.STR:
                new_ans.append(Answer(self.o.put(ans.answer), ans.answer_type))
            else:
                new_ans.append(ans)
        return new_ans

    def _process_question_data(self, question: str | Question) -> Any:
        if isinstance(question, Question):
            if question.deps is None:
                return question

            for dep in question.deps:
                if dep.dep_type == DataType.IMAGE_URL:
                    response = requests.get(dep.data, timeout=10)
                    response.raise_for_status()
                    dep.data = self.o.put(response.content)
            return question

        return Question(question)

    def import_data(
        self,
        questions: list[Any],
        answers: list[Answer],
        embedding_datas: list[Any],
        session_ids: list[str | None],
    ) -> None:
        if (
            len(questions) != len(answers)
            or len(questions) != len(embedding_datas)
            or len(questions) != len(session_ids)
        ):
            raise ParamError("Make sure that all parameters have the same length")
        cache_datas = []
        normalized_embeddings = [normalize(embedding_data) for embedding_data in embedding_datas]
        for i, embedding_data in enumerate(normalized_embeddings):
            if self.o is not None and not isinstance(answers[i], str):
                ans = self._process_answer_data(answers[i])
            else:
                ans = answers[i]

            cache_datas.append(
                CacheData(
                    question=self._process_question_data(questions[i]),
                    answers=ans,
                    embedding_data=embedding_data.astype("float32"),
                    session_id=session_ids[i],
                )
            )
        ids = self.s.batch_insert(cache_datas)
        self.v.mul_add(
            [
                VectorData(id=ids[i], data=embedding_data)
                for i, embedding_data in enumerate(normalized_embeddings)
            ]
        )
        self.eviction_base.put(ids)
        # LSH prefilter : index each new entry so future
        # lookups can short-circuit near-duplicate queries in O(1).
        try:
            from byte.manager.lsh_prefilter import (
                get_lsh_prefilter,  # pylint: disable=import-outside-toplevel
            )
            lsh = get_lsh_prefilter()
            if lsh is not None:
                for i, question in enumerate(questions):
                    text = question.content if isinstance(question, Question) else str(question or "")
                    if text and i < len(ids):
                        lsh.index(ids[i], text)
        except Exception:  # pragma: no cover - defensive (LSH must never break saves)
            pass

    def get_scalar_data(self, res_data, **kwargs) -> CacheData | None:
        session = kwargs.get("session")
        if isinstance(res_data, CacheData):
            cache_data = res_data
        else:
            cache_data = self.s.get_data_by_id(res_data[1])
        if cache_data is None:
            return None

        if session:
            cache_answer = (
                cache_data.answers[0].answer
                if isinstance(cache_data.answers[0], Answer)
                else cache_data.answers[0]
            )
            res_list = self.list_sessions(key=res_data[1])
            cache_session_ids = [result.session_id for result in res_list]
            cache_questions = [result.session_question for result in res_list]
            if not session.check_hit_func(
                session.name,
                cache_session_ids,
                cache_questions,
                cache_answer,
            ):
                return None

        for ans in cache_data.answers:
            if ans.answer_type != DataType.STR:
                ans.answer = self.o.get(ans.answer)
        return cache_data

    def hit_cache_callback(self, res_data, **kwargs) -> None:
        self.eviction_base.get(res_data[1])

    def search(self, embedding_data, **kwargs) -> Any:
        # LSH prefilter : probe LSH for near-duplicates first.
        text = kwargs.pop("question_text", "") or ""
        lsh_candidates: list[Any] = []
        if text:
            try:
                from byte.manager.lsh_prefilter import (
                    get_lsh_prefilter,  # pylint: disable=import-outside-toplevel
                )
                lsh = get_lsh_prefilter()
                if lsh is not None:
                    lsh_candidates = list(lsh.query(text))
            except Exception:  # pragma: no cover - defensive
                lsh_candidates = []
        embedding_data = normalize(embedding_data)
        top_k = kwargs.get("top_k", -1)
        results = self.v.search(data=embedding_data, top_k=top_k)
        # If LSH flagged any IDs that appear in the vector result set, bubble them
        # to the front so the similarity evaluator inspects them first.
        if lsh_candidates and results:
            candidate_set = set(lsh_candidates)
            prioritised = [r for r in results if r[1] in candidate_set]
            rest = [r for r in results if r[1] not in candidate_set]
            if prioritised:
                return prioritised + rest
        return results

    def invalidate_by_query(self, query: str, *, embedding_func=None) -> bool:
        if embedding_func is None:
            return False
        try:
            results = self.search(embedding_func(query), top_k=16)
            for _, candidate_id in results:
                cache_data = self.s.get_data_by_id(candidate_id)
                if cache_data is None:
                    continue
                stored_question = (
                    cache_data.question.content
                    if isinstance(cache_data.question, Question)
                    else cache_data.question
                )
                if stored_question != query:
                    continue
                self.s.mark_deleted([candidate_id])
                self.v.delete([candidate_id])
                return True
        except Exception as exc:  # pylint: disable=W0703
            byte_log.error(f"invalidate_by_query failed: {exc}")
        return False

    def flush(self) -> None:
        self.s.flush()
        self.v.flush()

    def add_session(self, res_data, session_id, pre_embedding_data) -> None:
        self.s.add_session(res_data[1], session_id, pre_embedding_data)

    def list_sessions(self, session_id=None, key=None) -> Any:
        res = self.s.list_sessions(session_id, key)
        if key:
            return res
        if session_id:
            return [result.id for result in res]
        return list(set(result.session_id for result in res))

    def delete_session(self, session_id) -> None:
        keys = self.list_sessions(session_id=session_id)
        self.s.delete_session(keys)

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ) -> None:
        self.s.report_cache(
            user_question,
            cache_question,
            cache_question_id,
            cache_answer,
            similarity_value,
            cache_delta_time,
        )

    def close(self) -> None:
        self.s.close()
        self.v.close()


_SAFE_PICKLE_GLOBALS = {
    ("builtins", "dict"): dict,
    ("builtins", "list"): list,
    ("builtins", "tuple"): tuple,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "str"): str,
    ("builtins", "bytes"): bytes,
    ("builtins", "bytearray"): bytearray,
    ("builtins", "int"): int,
    ("builtins", "float"): float,
    ("builtins", "bool"): bool,
    ("collections", "OrderedDict"): OrderedDict,
    ("datetime", "datetime"): datetime,
    ("cachetools", "LRUCache"): cachetools.LRUCache,
    ("byte.manager.scalar_data.base", "Answer"): Answer,
    ("byte.manager.scalar_data.base", "Question"): Question,
    ("byte.manager.scalar_data.base", "QuestionDep"): QuestionDep,
    ("byte.manager.scalar_data.base", "DataType"): DataType,
}
