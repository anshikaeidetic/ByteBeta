import importlib.util
from threading import Lock
from typing import Any

import numpy as np

from byte.manager.vector_data.base import VectorBase, VectorData
from byte.utils import import_torch

_HAS_CHROMADB = importlib.util.find_spec("chromadb") is not None

if _HAS_CHROMADB:
    import_torch()
    import chromadb  # pylint: disable=C0413


class _FallbackCollection:
    def __init__(self) -> None:
        self._data = {}
        self._lock = Lock()

    def add(self, embeddings, ids) -> None:
        with self._lock:
            for item_id, embedding in zip(ids, embeddings):
                self._data[str(item_id)] = list(embedding)

    def count(self) -> Any:
        with self._lock:
            return len(self._data)

    def query(self, query_embeddings, n_results=1, include=None) -> dict[str, Any]:  # pylint: disable=unused-argument
        query = np.asarray(query_embeddings[0], dtype="float32")
        ranked = []
        with self._lock:
            items = list(self._data.items())
        for item_id, embedding in items:
            vector = np.asarray(embedding, dtype="float32")
            distance = float(np.linalg.norm(query - vector))
            ranked.append((distance, item_id))
        ranked.sort(key=lambda item: item[0])
        top = ranked[:n_results]
        return {
            "distances": [[distance for distance, _ in top]],
            "ids": [[item_id for _, item_id in top]],
        }

    def delete(self, ids) -> None:
        with self._lock:
            for item_id in ids:
                self._data.pop(str(item_id), None)

    def get(self, item_id, include=None) -> dict[str, Any]:  # pylint: disable=unused-argument
        with self._lock:
            embedding = self._data.get(str(item_id))
        return {"embeddings": [embedding] if embedding is not None else []}

    def update(self, ids, embeddings) -> None:
        item_ids = ids if isinstance(ids, list) else [ids]
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            emb_list = embeddings
        else:
            emb_list = [embeddings]
        with self._lock:
            for item_id, embedding in zip(item_ids, emb_list):
                self._data[str(item_id)] = list(embedding)


class Chromadb(VectorBase):
    """vector store: Chromadb"""

    def __init__(
        self,
        client_settings=None,
        persist_directory=None,
        collection_name: str = "byte",
        top_k: int = 1,
    ) -> None:
        self.top_k = top_k
        self._persist_directory = persist_directory

        if _HAS_CHROMADB:
            if client_settings:
                self._client_settings = client_settings
            else:
                self._client_settings = chromadb.config.Settings()
                if persist_directory is not None:
                    self._client_settings = chromadb.config.Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=persist_directory,
                    )
            self._client = chromadb.Client(self._client_settings)
            self._collection = self._client.get_or_create_collection(name=collection_name)
        else:
            self._client_settings = None
            self._client = None
            self._collection = _FallbackCollection()

    def mul_add(self, datas: list[VectorData]) -> None:
        data_array, id_array = map(
            list, zip(*((data.data.tolist(), str(data.id)) for data in datas))
        )
        self._collection.add(embeddings=data_array, ids=id_array)

    def search(self, data, top_k: int = -1) -> list[Any]:
        if self._collection.count() == 0:
            return []
        if top_k == -1:
            top_k = self.top_k
        results = self._collection.query(
            query_embeddings=[data.tolist()],
            n_results=top_k,
            include=["distances"],
        )
        return list(zip(results["distances"][0], [int(x) for x in results["ids"][0]]))

    def delete(self, ids) -> None:
        self._collection.delete([str(x) for x in ids])

    def rebuild(self, ids=None) -> bool:  # pylint: disable=unused-argument
        return True

    def get_embeddings(self, data_id: int | str) -> Any | None:
        vec_emb = self._collection.get(
            str(data_id),
            include=["embeddings"],
        )["embeddings"]
        if vec_emb is None or len(vec_emb) < 1:
            return None
        return np.asarray(vec_emb[0], dtype="float32")

    def update_embeddings(self, data_id: int | str, emb: np.ndarray) -> None:
        self._collection.update(
            ids=str(data_id),
            embeddings=emb.tolist(),
        )

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.flush()
