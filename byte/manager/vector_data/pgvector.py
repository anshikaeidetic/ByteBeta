from __future__ import annotations

from typing import Any

import numpy as np

from byte.manager.vector_data.base import VectorBase, VectorData
from byte.utils import load_optional_module


def _load_sqlalchemy_stack() -> tuple[Any, ...]:
    sqlalchemy = load_optional_module("sqlalchemy", package="sqlalchemy")
    sqlalchemy_ext = load_optional_module("sqlalchemy.ext.declarative", package="sqlalchemy")
    sqlalchemy_orm = load_optional_module("sqlalchemy.orm", package="sqlalchemy")
    sqlalchemy_types = load_optional_module("sqlalchemy.types", package="sqlalchemy")
    return sqlalchemy, sqlalchemy_ext, sqlalchemy_orm, sqlalchemy_types


def _get_model_and_index(table_prefix, vector_dimension, index_type, lists) -> tuple[Any, ...]:
    sqlalchemy, sqlalchemy_ext, _, sqlalchemy_types = _load_sqlalchemy_stack()
    base = sqlalchemy_ext.declarative_base()

    class _VectorType(sqlalchemy_types.UserDefinedType):
        cache_ok = True

        def __init__(self, precision=8) -> None:
            self.precision = precision

        def get_col_spec(self, **_) -> str:
            return f"vector({self.precision})"

        def bind_processor(self, dialect) -> Any:  # pylint: disable=unused-argument
            return lambda value: value

        def result_processor(self, dialect, coltype) -> Any:  # pylint: disable=unused-argument
            return lambda value: value

    class VectorStoreTable(base):
        __tablename__ = table_prefix + "_pg_vector_store"
        __table_args__ = {"extend_existing": True}
        id = sqlalchemy.Column(sqlalchemy_types.Integer, primary_key=True, autoincrement=False)
        embedding = sqlalchemy.Column(_VectorType(vector_dimension), nullable=False)

    vector_store_index = sqlalchemy.Index(
        f"idx_{table_prefix}_pg_vector_store_embedding",
        sqlalchemy.text(f"embedding {index_type}"),
        postgresql_using="ivfflat",
        postgresql_with={"lists": lists},
    )
    vector_store_index.table = VectorStoreTable.__table__
    return VectorStoreTable, vector_store_index


class PGVector(VectorBase):
    """pgvector-backed vector store."""

    INDEX_PARAM = {
        "L2": {"operator": "<->", "name": "vector_l2_ops"},
        "cosine": {"operator": "<=>", "name": "vector_cosine_ops"},
        "inner_product": {"operator": "<->", "name": "vector_ip_ops"},
    }

    def __init__(
        self,
        url: str,
        index_params: dict,
        collection_name: str = "byte",
        dimension: int = 0,
        top_k: int = 1,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"invalid `dim` param: {dimension} in the pgvector store.")
        self.dimension = dimension
        self.top_k = top_k
        self.index_params = index_params
        self._url = url
        self._sqlalchemy, _, sqlalchemy_orm, _ = _load_sqlalchemy_stack()
        self._sessionmaker = sqlalchemy_orm.sessionmaker
        self._store, self._index = _get_model_and_index(
            collection_name,
            dimension,
            index_type=self.INDEX_PARAM[index_params["index_type"]]["name"],
            lists=index_params["params"]["lists"],
        )
        self._connect(url)
        self._create_collection()

    def _connect(self, url) -> None:
        self._engine = self._sqlalchemy.create_engine(url, echo=False)
        self._session = self._sessionmaker(bind=self._engine)  # pylint: disable=invalid-name

    def _create_collection(self) -> None:
        with self._engine.connect() as con:
            con.execution_options(isolation_level="AUTOCOMMIT").execute(
                self._sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;")
            )

        self._store.__table__.create(bind=self._engine, checkfirst=True)
        self._index.create(bind=self._engine, checkfirst=True)

    def _query(self, session) -> Any:
        return session.query(self._store)

    def _format_data_for_search(self, data) -> str:
        return f"[{','.join(map(str, data))}]"

    def mul_add(self, datas: list[VectorData]) -> None:
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        entities = [
            {"id": data_id, "embedding": embedding.tolist()}
            for data_id, embedding in zip(id_array, np_data)
        ]
        with self._session() as session:
            session.bulk_insert_mappings(self._store, entities)
            session.commit()

    def search(self, data: np.ndarray, top_k: int = -1) -> Any:
        if top_k == -1:
            top_k = self.top_k

        formatted_data = self._format_data_for_search(data.reshape(1, -1)[0].tolist())
        index_config = self.INDEX_PARAM[self.index_params["index_type"]]
        similarity = self._store.embedding.op(index_config["operator"])(formatted_data)
        with self._session() as session:
            session.execute(
                self._sqlalchemy.text(
                    f"SET LOCAL ivfflat.probes = {self.index_params['params']['probes'] or 10};"
                )
            )
            search_result = (
                self._query(session)
                .add_columns(similarity.label("distances"))
                .order_by(similarity)
                .limit(top_k)
                .all()
            )
            return [(row[1], row[0].id) for row in search_result]

    def delete(self, ids) -> None:
        with self._session() as session:
            self._query(session).filter(self._store.id.in_(ids)).delete()
            session.commit()

    def rebuild(self, ids=None) -> None:  # pylint: disable=unused-argument
        with self._engine.connect() as con:
            con.execution_options(isolation_level="AUTOCOMMIT").execute(
                self._sqlalchemy.text(f"REINDEX INDEX CONCURRENTLY {self._index.name}")
            )

    def flush(self) -> None:
        with self._session() as session:
            session.flush()

    def close(self) -> None:
        self.flush()
        self._engine.dispose()

    def get_embeddings(self, data_id: int | str) -> Any | None:
        with self._session() as session:
            result = self._query(session).filter(self._store.id == int(data_id)).first()
            if result is None:
                return None
            return np.asarray(result.embedding, dtype="float32")

    def update_embeddings(self, data_id: int | str, emb: np.ndarray) -> None:
        with self._session() as session:
            (
                self._query(session)
                .filter(self._store.id == int(data_id))
                .update({"embedding": np.asarray(emb, dtype="float32").tolist()})
            )
            session.commit()
