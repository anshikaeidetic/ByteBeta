from __future__ import annotations

import importlib.util
import logging
import os
from threading import Lock
from typing import Any

import numpy as np

from byte.manager.vector_data.base import VectorBase, VectorData
from byte.utils.error import ByteErrorCode
from byte.utils.log import byte_log, log_byte_error

_HAS_HNSWLIB = importlib.util.find_spec("hnswlib") is not None

if _HAS_HNSWLIB:
    import hnswlib  # pylint: disable=C0413


class _FallbackIndex:
    def __init__(self) -> None:
        self._items = {}
        self._lock = Lock()

    def add_items(self, vectors, ids) -> None:
        with self._lock:
            for item_id, vector in zip(ids, vectors, strict=True):
                self._items[int(item_id)] = np.asarray(vector, dtype="float32")

    def knn_query(self, data, k=1) -> tuple[Any, ...]:
        query = np.asarray(data[0], dtype="float32")
        ranked = []
        with self._lock:
            items = list(self._items.items())
        for item_id, vector in items:
            distance = float(np.linalg.norm(query - vector))
            ranked.append((distance, item_id))
        ranked.sort(key=lambda item: item[0])
        top = ranked[:k]
        ids = np.asarray([[item_id for _, item_id in top]])
        distances = np.asarray([[distance for distance, _ in top]], dtype="float32")
        return ids, distances

    def get_items(self, ids) -> Any:
        with self._lock:
            items = [self._items[int(item_id)] for item_id in ids]
        return np.asarray(items, dtype="float32")

    def mark_deleted(self, item_id) -> None:
        with self._lock:
            self._items.pop(int(item_id), None)

    def save_index(self, index_path) -> None:
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        with self._lock:
            item_ids = np.asarray(list(self._items.keys()), dtype="int64")
            vectors = np.asarray(
                [self._items[int(item_id)] for item_id in item_ids],
                dtype="float32",
            )
        with open(index_path, "wb") as handle:
            np.savez(handle, ids=item_ids, vectors=vectors)

    def load_index(self, index_path, max_elements=None) -> None:  # pylint: disable=unused-argument
        with np.load(index_path, allow_pickle=False) as data:
            ids = data["ids"]
            vectors = data["vectors"]
        with self._lock:
            self._items = {
                int(item_id): np.asarray(vector, dtype="float32")
                for item_id, vector in zip(ids, vectors, strict=True)
            }

    def init_index(self, max_elements=0, ef_construction=0, M=0) -> None:  # pylint: disable=unused-argument
        return

    def set_ef(self, value) -> None:  # pylint: disable=unused-argument
        return


class Hnswlib(VectorBase):
    """vector store: hnswlib"""

    def __init__(self, index_file_path: str, dimension: int, top_k: int, max_elements: int) -> None:
        self._index_file_path = index_file_path
        self._dimension = dimension
        self._max_elements = max_elements
        self._top_k = top_k
        if _HAS_HNSWLIB:
            self._index = hnswlib.Index(space="l2", dim=self._dimension)
        else:
            self._index = _FallbackIndex()
        if os.path.isfile(self._index_file_path):
            self._index.load_index(self._index_file_path, max_elements=max_elements)
        else:
            self._index.init_index(max_elements=max_elements, ef_construction=100, M=16)
            self._index.set_ef(self._top_k * 2)

    def add(self, key: int, data: np.ndarray) -> None:
        np_data = np.array(data).astype("float32").reshape(1, -1)
        self._index.add_items(np_data, np.array([key]))

    def mul_add(self, datas: list[VectorData]) -> None:
        data_array = [data.data for data in datas]
        id_array = [data.id for data in datas]
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array)
        self._index.add_items(np_data, ids)

    def search(self, data: np.ndarray, top_k: int = -1) -> Any:
        np_data = np.array(data).astype("float32").reshape(1, -1)
        if top_k == -1:
            top_k = self._top_k
        ids, dist = self._index.knn_query(data=np_data, k=top_k)
        return [
            (float(distance), int(item_id))
            for distance, item_id in zip(dist[0].tolist(), ids[0].tolist(), strict=True)
        ]

    def rebuild(self, ids) -> None:
        all_data = self._index.get_items(ids)
        if _HAS_HNSWLIB:
            new_index = hnswlib.Index(space="l2", dim=self._dimension)
        else:
            new_index = _FallbackIndex()
        new_index.init_index(max_elements=self._max_elements, ef_construction=100, M=16)
        new_index.set_ef(self._top_k * 2)
        self._index = new_index
        datas = []
        for key, data in zip(ids, all_data, strict=True):
            datas.append(VectorData(id=key, data=data))
        self.mul_add(datas)

    def delete(self, ids) -> None:
        for item_id in ids:
            self._index.mark_deleted(item_id)

    def flush(self) -> None:
        self._index.save_index(self._index_file_path)

    def close(self) -> None:
        self.flush()

    def get_embeddings(self, data_id: int | str) -> Any | None:
        try:
            vectors = self._index.get_items([int(data_id)])
        except (AttributeError, TypeError, ValueError) as exc:
            log_byte_error(
                byte_log,
                logging.DEBUG,
                "HNSW embedding lookup failed.",
                error=exc,
                code=ByteErrorCode.STORAGE_READ,
                boundary="storage.read",
                stage="hnsw_get_embeddings",
                exc_info=False,
            )
            return None
        if vectors is None or len(vectors) < 1:
            return None
        return np.asarray(vectors[0], dtype="float32")

    def update_embeddings(self, data_id: int | str, emb: np.ndarray) -> None:
        vector_id = int(data_id)
        self.delete([vector_id])
        self.add(vector_id, np.asarray(emb, dtype="float32"))
