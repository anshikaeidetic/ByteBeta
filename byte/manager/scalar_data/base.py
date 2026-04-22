from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Any

import numpy as np

from byte.utils.async_ops import run_sync


class DataType(IntEnum):
    STR = 0
    IMAGE_BASE64 = 1
    IMAGE_URL = 2


@dataclass
class Answer:
    """
    data_type:
        0: str
        1: base64 image
    """

    answer: Any
    answer_type: int = DataType.STR


@dataclass
class QuestionDep:
    """
    QuestionDep
    """

    name: str
    data: str
    dep_type: int = DataType.STR

    @classmethod
    def from_dict(cls, d: dict) -> Any:
        return cls(name=d["name"], data=d["data"], dep_type=d["dep_type"])


@dataclass
class Question:
    """
    Question
    """

    content: str
    deps: list[QuestionDep] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Any:
        deps = []
        for dep in d["deps"]:
            deps.append(QuestionDep.from_dict(dep))
        return cls(d["content"], deps)


@dataclass
class CacheData:
    """
    CacheData
    """

    question: str | Question
    answers: list[Answer]
    embedding_data: np.ndarray | None = None
    session_id: str | None = None
    create_on: datetime | None = None
    last_access: datetime | None = None

    def __init__(
        self,
        question,
        answers,
        embedding_data=None,
        session_id=None,
        create_on=None,
        last_access=None,
    ) -> None:
        self.question = question
        self.answers = []
        if isinstance(answers, (str, Answer)):
            answers = [answers]
        for data in answers:
            if isinstance(data, (list, tuple)):
                self.answers.append(Answer(*data))
            elif isinstance(data, Answer):
                self.answers.append(data)
            else:
                self.answers.append(Answer(answer=data))
        self.embedding_data = embedding_data
        self.session_id = session_id
        self.create_on = create_on
        self.last_access = last_access


class CacheStorage(metaclass=ABCMeta):
    """
    BaseStorage for scalar data.
    """

    @abstractmethod
    def create(self) -> None:
        pass

    async def acreate(self) -> Any:
        return await run_sync(self.create)

    @abstractmethod
    def batch_insert(self, all_data: list[CacheData]) -> None:
        pass

    async def abatch_insert(self, all_data: list[CacheData]) -> Any:
        return await run_sync(self.batch_insert, all_data)

    @abstractmethod
    def get_data_by_id(self, key) -> None:
        pass

    async def aget_data_by_id(self, key) -> Any:
        return await run_sync(self.get_data_by_id, key)

    @abstractmethod
    def mark_deleted(self, keys) -> None:
        pass

    async def amark_deleted(self, keys) -> Any:
        return await run_sync(self.mark_deleted, keys)

    @abstractmethod
    def clear_deleted_data(self) -> None:
        pass

    async def aclear_deleted_data(self) -> Any:
        return await run_sync(self.clear_deleted_data)

    @abstractmethod
    def get_ids(self, deleted=True) -> None:
        pass

    async def aget_ids(self, deleted=True) -> Any:
        return await run_sync(self.get_ids, deleted)

    @abstractmethod
    def count(self, state: int = 0, is_all: bool = False) -> None:
        pass

    async def acount(self, state: int = 0, is_all: bool = False) -> Any:
        return await run_sync(self.count, state, is_all)

    @abstractmethod
    def flush(self) -> None:
        pass

    async def aflush(self) -> Any:
        return await run_sync(self.flush)

    @abstractmethod
    def add_session(self, question_id, session_id, session_question) -> None:
        pass

    async def aadd_session(self, question_id, session_id, session_question) -> Any:
        return await run_sync(self.add_session, question_id, session_id, session_question)

    @abstractmethod
    def list_sessions(self, session_id, key) -> None:
        pass

    async def alist_sessions(self, session_id, key) -> Any:
        return await run_sync(self.list_sessions, session_id, key)

    @abstractmethod
    def delete_session(self, keys) -> None:
        pass

    async def adelete_session(self, keys) -> Any:
        return await run_sync(self.delete_session, keys)

    @abstractmethod
    @abstractmethod
    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ) -> None:
        pass

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

    @abstractmethod
    @abstractmethod
    def close(self) -> None:
        pass

    async def aclose(self) -> Any:
        return await run_sync(self.close)
