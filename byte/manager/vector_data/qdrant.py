
from typing import Any

import numpy as np

from byte.manager.vector_data.base import VectorBase, VectorData
from byte.utils import lazy_optional_module
from byte.utils.log import byte_log

qdrant_client = lazy_optional_module("qdrant_client", package="qdrant-client")
qdrant_models = lazy_optional_module("qdrant_client.models", package="qdrant-client")


class QdrantVectorStore(VectorBase):
    """Qdrant Vector Store"""

    def __init__(
        self,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: bool | None = None,
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: float | None = None,
        host: str | None = None,
        collection_name: str | None = "byte",
        location: str | None = "./qdrant",
        dimension: int = 0,
        top_k: int = 1,
        flush_interval_sec: int = 5,
        index_params: dict | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"invalid `dim` param: {dimension} in the Qdrant vector store.")
        self._collection_name = collection_name
        self._in_memory = location == ":memory:"
        self.dimension = dimension
        self.top_k = top_k
        if self._in_memory or location is not None:
            self._create_local(location)
        else:
            self._create_remote(
                url, port, api_key, timeout, host, grpc_port, prefer_grpc, prefix, https
            )
        self._create_collection(collection_name, flush_interval_sec, index_params)

    def _create_local(self, location) -> None:
        self._client = qdrant_client.QdrantClient(location=location)

    def _create_remote(
        self, url, port, api_key, timeout, host, grpc_port, prefer_grpc, prefix, https
    ) -> None:
        self._client = qdrant_client.QdrantClient(
            url=url,
            port=port,
            api_key=api_key,
            timeout=timeout,
            host=host,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            prefix=prefix,
            https=https,
        )

    def _create_collection(
        self,
        collection_name: str,
        flush_interval_sec: int,
        index_params: dict | None = None,
    ) -> None:
        hnsw_config = qdrant_models.HnswConfigDiff(**(index_params or {}))
        vectors_config = qdrant_models.VectorParams(
            size=self.dimension,
            distance=qdrant_models.Distance.COSINE,
            hnsw_config=hnsw_config,
        )
        optimizers_config = qdrant_models.OptimizersConfigDiff(
            deleted_threshold=0.2,
            vacuum_min_vector_number=1000,
            flush_interval_sec=flush_interval_sec,
        )
        # check if the collection exists
        existing_collections = self._client.get_collections()
        for existing_collection in existing_collections.collections:
            if existing_collection.name == collection_name:
                byte_log.warning(
                    "The %s collection already exists, and it will be used directly.",
                    collection_name,
                )
                break
        else:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                optimizers_config=optimizers_config,
            )

    def mul_add(self, datas: list[VectorData]) -> None:
        points = [
            qdrant_models.PointStruct(id=d.id, vector=d.data.reshape(-1).tolist()) for d in datas
        ]
        self._client.upsert(collection_name=self._collection_name, points=points, wait=False)

    def search(self, data: np.ndarray, top_k: int = -1) -> list[Any]:
        if top_k == -1:
            top_k = self.top_k
        reshaped_data = data.reshape(-1).tolist()
        if hasattr(self._client, "search"):
            search_result = self._client.search(
                collection_name=self._collection_name,
                query_vector=reshaped_data,
                limit=top_k,
            )
        else:
            search_result = self._client.query_points(
                collection_name=self._collection_name,
                query=reshaped_data,
                limit=top_k,
            ).points
        return list(map(lambda x: (x.score, x.id), search_result))

    def delete(self, ids: list[str]) -> None:
        self._client.delete(collection_name=self._collection_name, points_selector=ids)

    def rebuild(self, ids=None) -> None:  # pylint: disable=unused-argument
        optimizers_config = qdrant_models.OptimizersConfigDiff(
            deleted_threshold=0.2, vacuum_min_vector_number=1000
        )
        try:
            self._client.update_collection(
                collection_name=self._collection_name,
                optimizers_config=optimizers_config,
            )
        except TypeError:
            self._client.update_collection(
                collection_name=self._collection_name,
                optimizer_config=optimizers_config,
            )

    def flush(self) -> None:
        # no need to flush manually as qdrant flushes automatically based on the optimizers_config for remote Qdrant
        pass

    def close(self) -> None:
        self.flush()
