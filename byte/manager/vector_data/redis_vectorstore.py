from __future__ import annotations

import importlib.util
import logging
from typing import Any

import numpy as np

from byte.manager.vector_data.base import VectorBase, VectorData
from byte.utils.error import ByteErrorCode
from byte.utils.log import byte_log, log_byte_error

if importlib.util.find_spec("redis") is None:
    raise ModuleNotFoundError(
        "Optional dependency 'redis' is not installed. Install it with `pip install redis`."
    )

# pylint: disable=C0413
try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ModuleNotFoundError:  # pragma: no cover - compatibility with older redis-py
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.client import Redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.query import Query


class RedisVectorStore(VectorBase):
    """vector store: Redis

    :param host: redis host, defaults to "localhost".
    :type host: str
    :param port: redis port, defaults to "6379".
    :type port: str
    :param username: redis username, defaults to None.
    :type username: str | None
    :param password: redis password, defaults to None.
    :type password: str | None
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param collection_name: the name of the index for Redis, defaults to "byte".
    :type collection_name: str
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int

    Example:
        .. code-block:: python

            from byte.manager import VectorBase

            vector_base = VectorBase("redis", dimension=10)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: str = "6379",
        username: str | None = None,
        password: str | None = None,
        dimension: int = 0,
        collection_name: str = "byte",
        top_k: int = 1,
        namespace: str = "",
    ) -> None:
        self._client = Redis(host=host, port=int(port), username=username, password=password)
        self.top_k = top_k
        self.dimension = dimension
        self.collection_name = collection_name
        self.namespace = namespace
        self.doc_prefix = f"{self.namespace}doc:"  # Prefix with the specified namespace
        self._create_collection(collection_name)

    def _check_index_exists(self, index_name: str) -> bool:
        """Check if Redis index exists."""
        try:
            self._client.ft(index_name).info()
        except Exception as exc:  # pragma: no cover - redis boundary
            log_byte_error(
                byte_log,
                logging.DEBUG,
                "Redis index probe failed.",
                error=exc,
                code=ByteErrorCode.STORAGE_SEARCH,
                boundary="storage.search",
                stage="redis_index_probe",
                exc_info=False,
            )
            byte_log.info("Index does not exist")
            return False
        byte_log.info("Index already exists")
        return True

    def _create_collection(self, collection_name) -> None:
        if self._check_index_exists(collection_name):
            byte_log.info("The %s already exists, and it will be used directly", collection_name)
        else:
            schema = (
                TagField("tag"),  # Tag Field Name
                VectorField(
                    "vector",  # Vector Field Name
                    "FLAT",
                    {  # Vector Index Type: FLAT or HNSW
                        "TYPE": "FLOAT32",  # FLOAT32 or FLOAT64
                        "DIM": self.dimension,  # Number of Vector Dimensions
                        "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                    },
                ),
            )
            definition = IndexDefinition(prefix=[self.doc_prefix], index_type=IndexType.HASH)

            # create Index
            self._client.ft(collection_name).create_index(fields=schema, definition=definition)

    def mul_add(self, datas: list[VectorData]) -> None:
        pipe = self._client.pipeline()

        for data in datas:
            key: int = data.id
            obj = {
                "vector": data.data.astype(np.float32).tobytes(),
            }
            pipe.hset(f"{self.doc_prefix}{key}", mapping=obj)

        pipe.execute()

    def search(self, data: np.ndarray, top_k: int = -1) -> Any:
        query = (
            Query(f"*=>[KNN {top_k if top_k > 0 else self.top_k} @vector $vec as score]")
            .sort_by("score")
            .return_fields("id", "score")
            .paging(0, top_k if top_k > 0 else self.top_k)
            .dialect(2)
        )
        query_params = {"vec": data.astype(np.float32).tobytes()}
        results = (
            self._client.ft(self.collection_name).search(query, query_params=query_params).docs
        )
        return [(float(result.score), int(result.id[len(self.doc_prefix) :])) for result in results]

    def rebuild(self, ids=None) -> bool:
        return True

    def delete(self, ids) -> None:
        pipe = self._client.pipeline()
        for data_id in ids:
            pipe.delete(f"{self.doc_prefix}{data_id}")
        pipe.execute()

    def get_embeddings(self, data_id) -> Any | None:
        payload = self._client.hget(f"{self.doc_prefix}{int(data_id)}", "vector")
        if payload is None:
            return None
        return np.frombuffer(payload, dtype=np.float32)

    def update_embeddings(self, data_id, emb: np.ndarray) -> bool:
        self._client.hset(
            f"{self.doc_prefix}{int(data_id)}",
            mapping={"vector": np.asarray(emb, dtype=np.float32).tobytes()},
        )
        return True
