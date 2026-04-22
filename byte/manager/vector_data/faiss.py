import os
from typing import Any

import numpy as np

from byte.manager.vector_data.base import VectorBase, VectorData
from byte.utils import lazy_optional_module

faiss = lazy_optional_module("faiss", package="faiss-cpu")


class Faiss(VectorBase):
    """vector store: Faiss

    :param index_path: the path to Faiss index, defaults to 'faiss.index'.
    :type index_path: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    """

    def __init__(self, index_file_path, dimension, top_k) -> None:
        self._index_file_path = index_file_path
        self._dimension = dimension
        self._index = faiss.index_factory(self._dimension, "IDMap,Flat", faiss.METRIC_L2)
        self._top_k = top_k
        self._embedding_store = {}
        if os.path.isfile(index_file_path):
            loaded_index = faiss.read_index(index_file_path)
            if getattr(loaded_index, "d", None) != self._dimension:
                raise ValueError(
                    f"existing Faiss index dimension {loaded_index.d} does not match requested dimension {self._dimension}"
                )
            self._index = loaded_index

    def mul_add(self, datas: list[VectorData]) -> None:
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.asarray(id_array, dtype="int64")
        for vector_id, embedding in zip(ids.tolist(), np_data):
            self._embedding_store[int(vector_id)] = np.asarray(embedding, dtype="float32").copy()
        self._index.add_with_ids(np_data, ids)

    def search(self, data: np.ndarray, top_k: int = -1) -> list[Any] | None:
        if self._index.ntotal == 0:
            return None
        if top_k == -1:
            top_k = self._top_k
        np_data = np.array(data).astype("float32").reshape(1, -1)
        dist, ids = self._index.search(np_data, top_k)
        ids = [int(i) for i in ids[0]]
        return list(zip(dist[0], ids))

    def rebuild(self, ids=None) -> bool:
        return True

    def delete(self, ids) -> None:
        ids_to_remove = np.asarray(ids, dtype="int64")
        for vector_id in ids_to_remove.tolist():
            self._embedding_store.pop(int(vector_id), None)
        self._index.remove_ids(
            faiss.IDSelectorBatch(ids_to_remove.size, faiss.swig_ptr(ids_to_remove))
        )

    def get_embeddings(self, data_id) -> Any | None:
        embedding = self._embedding_store.get(int(data_id))
        if embedding is None:
            return None
        return np.asarray(embedding, dtype="float32").copy()

    def update_embeddings(self, data_id, emb: np.ndarray) -> None:
        vector_id = int(data_id)
        self.delete([vector_id])
        np_data = np.asarray(emb, dtype="float32").reshape(1, -1)
        ids = np.asarray([vector_id], dtype="int64")
        self._embedding_store[vector_id] = np_data[0].copy()
        self._index.add_with_ids(np_data, ids)

    def flush(self) -> None:
        os.makedirs(os.path.dirname(self._index_file_path) or ".", exist_ok=True)
        faiss.write_index(self._index, self._index_file_path)

    def close(self) -> None:
        self.flush()

    def count(self) -> Any:
        return self._index.ntotal
