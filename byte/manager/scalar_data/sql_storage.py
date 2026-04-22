from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from byte.manager.scalar_data.base import CacheData, CacheStorage, Question, QuestionDep
from byte.utils import load_optional_module

DEFAULT_LEN_DOCT = {
    "question_question": 3000,
    "answer_answer": 3000,
    "session_id": 1000,
    "dep_name": 1000,
    "dep_data": 3000,
}


def _get_table_len(config: dict, column_alias: str) -> int:
    if config and column_alias in config and config[column_alias] > 0:
        return config[column_alias]
    return DEFAULT_LEN_DOCT.get(column_alias, 1000)


def get_models(table_prefix, db_type, table_len_config) -> tuple[Any, ...]:
    sqlalchemy = load_optional_module("sqlalchemy", package="sqlalchemy")
    sqlalchemy_ext = load_optional_module("sqlalchemy.ext.declarative", package="sqlalchemy")
    sqlalchemy_types = load_optional_module("sqlalchemy.types", package="sqlalchemy")

    Column = sqlalchemy.Column
    Sequence = sqlalchemy.Sequence
    DateTime = sqlalchemy_types.DateTime
    Float = sqlalchemy_types.Float
    Integer = sqlalchemy_types.Integer
    LargeBinary = sqlalchemy_types.LargeBinary
    String = sqlalchemy_types.String
    DynamicBase = sqlalchemy_ext.declarative_base(class_registry={})  # pylint: disable=C0103

    class QuestionTable(DynamicBase):
        __tablename__ = table_prefix + "_question"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            question_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(Integer, question_id_seq, primary_key=True, autoincrement=True)
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question = Column(String(_get_table_len(table_len_config, "question_question")), nullable=False)
        create_on = Column(DateTime, default=datetime.now)
        last_access = Column(DateTime, default=datetime.now)
        embedding_data = Column(LargeBinary, nullable=True)
        deleted = Column(Integer, default=0)

    class AnswerTable(DynamicBase):
        __tablename__ = table_prefix + "_answer"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            answer_id_seq = Sequence(f"{__tablename__}_id_seq")
            id = Column(Integer, answer_id_seq, primary_key=True, autoincrement=True)
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question_id = Column(Integer, nullable=False)
        answer = Column(String(_get_table_len(table_len_config, "answer_answer")), nullable=False)
        answer_type = Column(Integer, nullable=False)

    class SessionTable(DynamicBase):
        __tablename__ = table_prefix + "_session"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            session_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(Integer, session_id_seq, primary_key=True, autoincrement=True)
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question_id = Column(Integer, nullable=False)
        session_id = Column(String(_get_table_len(table_len_config, "session_id")), nullable=False)
        session_question = Column(
            String(_get_table_len(table_len_config, "question_question")),
            nullable=False,
        )

    class QuestionDepTable(DynamicBase):
        __tablename__ = table_prefix + "_question_dep"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            question_dep_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(Integer, question_dep_id_seq, primary_key=True, autoincrement=True)
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question_id = Column(Integer, nullable=False)
        dep_name = Column(String(_get_table_len(table_len_config, "dep_name")), nullable=False)
        dep_data = Column(String(_get_table_len(table_len_config, "dep_data")), nullable=False)
        dep_type = Column(Integer, nullable=False)

    class ReportTable(DynamicBase):
        __tablename__ = table_prefix + "_report"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            report_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(Integer, report_id_seq, primary_key=True, autoincrement=True)
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        user_question = Column(
            String(_get_table_len(table_len_config, "question_question")),
            nullable=False,
        )
        cache_question_id = Column(Integer, nullable=False)
        cache_question = Column(
            String(_get_table_len(table_len_config, "question_question")),
            nullable=False,
        )
        cache_answer = Column(String(_get_table_len(table_len_config, "answer_answer")), nullable=False)
        similarity = Column(Float, nullable=False)
        cache_delta_time = Column(Float, nullable=False)
        cache_time = Column(DateTime, default=datetime.now)
        extra = Column(String(_get_table_len(table_len_config, "question_question")), nullable=True)

    return QuestionTable, AnswerTable, QuestionDepTable, SessionTable, ReportTable


class SQLStorage(CacheStorage):
    """SQLAlchemy-backed scalar storage for Byte cache metadata."""

    def __init__(
        self,
        db_type: str = "sqlite",
        url: str = "sqlite:///./sqlite.db",
        table_name: str = "byte",
        table_len_config=None,
    ) -> None:
        if table_len_config is None:
            table_len_config = {}
        self._sqlalchemy = load_optional_module("sqlalchemy", package="sqlalchemy")
        self._sqlalchemy_orm = load_optional_module("sqlalchemy.orm", package="sqlalchemy")
        sqlalchemy_pool = load_optional_module("sqlalchemy.pool", package="sqlalchemy")
        self._url = url
        self._ques, self._answer, self._ques_dep, self._session, self._report = get_models(
            table_name,
            db_type,
            table_len_config,
        )
        engine_kwargs = {}
        if str(self._url).startswith("sqlite:///"):
            engine_kwargs["poolclass"] = sqlalchemy_pool.NullPool
        self._engine = self._sqlalchemy.create_engine(self._url, **engine_kwargs)
        self.Session = self._sqlalchemy_orm.sessionmaker(bind=self._engine)  # pylint: disable=invalid-name
        self.create()

    def create(self) -> None:
        self._ques.__table__.create(bind=self._engine, checkfirst=True)
        self._answer.__table__.create(bind=self._engine, checkfirst=True)
        self._ques_dep.__table__.create(bind=self._engine, checkfirst=True)
        self._session.__table__.create(bind=self._engine, checkfirst=True)
        self._report.__table__.create(bind=self._engine, checkfirst=True)

    def _insert(self, data: CacheData, session: Any) -> Any:
        ques_data = self._ques(
            question=data.question if isinstance(data.question, str) else data.question.content,
            embedding_data=data.embedding_data.tobytes() if data.embedding_data is not None else None,
        )
        session.add(ques_data)
        session.flush()
        if isinstance(data.question, Question) and data.question.deps is not None:
            all_deps = []
            for dep in data.question.deps:
                all_deps.append(
                    self._ques_dep(
                        question_id=ques_data.id,
                        dep_name=dep.name,
                        dep_data=dep.data,
                        dep_type=dep.dep_type,
                    )
                )
            session.add_all(all_deps)
        answers = data.answers if isinstance(data.answers, list) else [data.answers]
        all_data = []
        for answer in answers:
            all_data.append(
                self._answer(
                    question_id=ques_data.id,
                    answer=answer.answer,
                    answer_type=int(answer.answer_type),
                )
            )
        session.add_all(all_data)
        if data.session_id:
            session.add(
                self._session(
                    question_id=ques_data.id,
                    session_id=data.session_id,
                    session_question=data.question
                    if isinstance(data.question, str)
                    else data.question.content,
                )
            )
        return ques_data.id

    def batch_insert(self, all_data: list[CacheData]) -> Any:
        ids = []
        with self.Session() as session:
            for data in all_data:
                ids.append(self._insert(data, session))
            session.commit()
        return ids

    def get_data_by_id(self, key: int) -> CacheData | None:
        with self.Session() as session:
            qs = (
                session.query(self._ques)
                .filter(self._ques.id == key)
                .filter(self._ques.deleted == 0)
                .first()
            )
            if qs is None:
                return None
            last_access = qs.last_access
            qs.last_access = datetime.now()
            ans = (
                session.query(self._answer.answer, self._answer.answer_type)
                .filter(self._answer.question_id == qs.id)
                .all()
            )
            deps = (
                session.query(self._ques_dep.dep_name, self._ques_dep.dep_data, self._ques_dep.dep_type)
                .filter(self._ques_dep.question_id == qs.id)
                .all()
            )
            session_ids = (
                session.query(self._session.session_id)
                .filter(self._session.question_id == qs.id)
                .all()
            )
            res_ans = [(item.answer, item.answer_type) for item in ans]
            res_deps = [QuestionDep(item.dep_name, item.dep_data, item.dep_type) for item in deps]
            session.commit()
            return CacheData(
                question=qs.question if not deps else Question(qs.question, res_deps),
                answers=res_ans,
                embedding_data=np.frombuffer(qs.embedding_data, dtype=np.float32),
                session_id=session_ids,
                create_on=qs.create_on,
                last_access=last_access,
            )

    def get_ids(self, deleted=True) -> Any:
        state = -1 if deleted else 0
        with self.Session() as session:
            res = session.query(self._ques.id).filter(self._ques.deleted == state).all()
            return [item.id for item in res]

    def mark_deleted(self, keys) -> None:
        with self.Session() as session:
            session.query(self._ques).filter(self._ques.id.in_(keys)).update({"deleted": -1})
            session.commit()

    def clear_deleted_data(self) -> None:
        with self.Session() as session:
            objs = session.query(self._ques).filter(self._ques.deleted == -1)
            q_ids = [obj.id for obj in objs]
            session.query(self._answer).filter(self._answer.question_id.in_(q_ids)).delete()
            session.query(self._ques_dep).filter(self._ques_dep.question_id.in_(q_ids)).delete()
            session.query(self._session).filter(self._session.question_id.in_(q_ids)).delete()
            objs.delete()
            session.commit()

    def count(self, state: int = 0, is_all: bool = False) -> Any:
        with self.Session() as session:
            if is_all:
                return session.query(self._sqlalchemy.func.count(self._ques.id)).scalar()
            return (
                session.query(self._sqlalchemy.func.count(self._ques.id))
                .filter(self._ques.deleted == state)
                .scalar()
            )

    def flush(self) -> None:
        return None

    def add_session(self, question_id, session_id, session_question) -> None:
        with self.Session() as session:
            session.add(
                self._session(
                    question_id=question_id,
                    session_id=session_id,
                    session_question=session_question,
                )
            )
            session.commit()

    def delete_session(self, keys) -> None:
        with self.Session() as session:
            session.query(self._session).filter(self._session.id.in_(keys)).delete()
            session.commit()

    def list_sessions(self, session_id=None, key=None) -> Any:
        with self.Session() as session:
            query = session.query(self._session)
            if session_id:
                query = query.filter(self._session.session_id == session_id)
            elif key:
                query = query.filter(self._session.question_id == key)
            return query.all()

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ) -> None:
        with self.Session() as session:
            session.add(
                self._report(
                    user_question=user_question,
                    cache_question=cache_question,
                    cache_question_id=cache_question_id,
                    cache_answer=cache_answer,
                    similarity=similarity_value,
                    cache_delta_time=cache_delta_time,
                )
            )
            session.commit()

    def close(self) -> None:
        self._engine.dispose()

    def count_answers(self) -> Any:
        with self.Session() as session:
            return session.query(self._sqlalchemy.func.count(self._answer.id)).scalar()
