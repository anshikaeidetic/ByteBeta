import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from byte.manager.vector_data.base import VectorBase, VectorData
from byte.utils import lazy_optional_module
from byte.utils.error import ByteErrorCode
from byte.utils.log import byte_log, log_byte_error

usearch_compiled = lazy_optional_module("usearch.compiled", package="usearch==0.22.3")
usearch_index = lazy_optional_module("usearch.index", package="usearch==0.22.3")


class USearch(VectorBase):
    """vector store: Usearch

    :param index_path: the path to Usearch index, defaults to 'index.usearch'.
    :type index_path: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    :param metric: the distance mrtric. 'l2', 'haversine' or other, default = 'ip'
    :type metric: str
    :param dtype: the quantization dtype, 'f16' or 'f8' if needed, default = 'f32'
    :type dtype: str
    :param connectivity: the frequency of the connections in the graph, optional
    :type connectivity: int
    :param expansion_add: the recall of indexing, optional
    :type expansion_add: int
    :param expansion_search: the quality of search, optional
    :type expansion_search: int
    """

    def __init__(
        self,
        index_file_path: str = "index.usearch",
        dimension: int = 64,
        top_k: int = 1,
        metric: str = "cos",
        dtype: str = "f32",
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ) -> None:
        self._index_file_path = index_file_path
        self._metadata_file_path = f"{index_file_path}.byte_vectors.npz"
        self._dimension = dimension
        self._top_k = top_k
        self._metric = metric
        self._dtype = dtype
        self._connectivity = connectivity
        self._expansion_add = expansion_add
        self._expansion_search = expansion_search
        self._index = self._build_index()
        self._vectors: dict[int, np.ndarray] = {}
        if os.path.isfile(self._index_file_path):
            self._index.load(self._index_file_path)
        self._load_vectors()

    def _build_index(self) -> Any:
        return usearch_index.Index(
            ndim=self._dimension,
            metric=getattr(usearch_compiled.MetricKind, self._metric.lower().capitalize()),
            dtype=self._dtype,
            connectivity=self._connectivity,
            expansion_add=self._expansion_add,
            expansion_search=self._expansion_search,
        )

    def _load_vectors(self) -> None:
        metadata_path = Path(self._metadata_file_path)
        if not metadata_path.is_file():
            return
        payload = np.load(metadata_path, allow_pickle=False)
        ids = payload.get("ids")
        vectors = payload.get("vectors")
        if ids is None or vectors is None:
            return
        for data_id, vector in zip(ids.tolist(), vectors):
            self._vectors[int(data_id)] = np.asarray(vector, dtype="float32")

    def _persist_vectors(self) -> None:
        metadata_path = Path(self._metadata_file_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._vectors:
            if metadata_path.exists():
                metadata_path.unlink()
            return
        ids = np.array(sorted(self._vectors), dtype=np.longlong)
        vectors = np.array([self._vectors[int(data_id)] for data_id in ids], dtype="float32")
        np.savez_compressed(metadata_path, ids=ids, vectors=vectors)

    def mul_add(self, datas: list[VectorData]) -> None:
        if not datas:
            return
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array, dtype=np.longlong)
        self._index.add(ids, np_data)
        for data_id, vector in zip(ids.tolist(), np_data):
            self._vectors[int(data_id)] = np.asarray(vector, dtype="float32")

    def search(self, data: np.ndarray, top_k: int = -1) -> list[Any]:
        if top_k == -1:
            top_k = self._top_k
        np_data = np.array(data).astype("float32").reshape(1, -1)
        ids, dist, _ = self._index.search(np_data, top_k)
        return list(zip(dist[0], ids[0]))

    def rebuild(self, ids=None) -> bool:
        items = sorted(self._vectors.items())
        self._index = self._build_index()
        if items:
            np_ids = np.array([int(data_id) for data_id, _ in items], dtype=np.longlong)
            np_vectors = np.array([vector for _, vector in items], dtype="float32")
            self._index.add(np_ids, np_vectors)
        return True

    def delete(self, ids) -> bool:
        if isinstance(ids, (list, tuple, set, np.ndarray)):
            ids_to_delete = [int(item) for item in ids]
        else:
            ids_to_delete = [int(ids)]
        removed = False
        for data_id in ids_to_delete:
            removed = self._vectors.pop(int(data_id), None) is not None or removed
        if not removed:
            return False
        remove = getattr(self._index, "remove", None)
        if callable(remove):
            try:
                remove(np.array(ids_to_delete, dtype=np.longlong))
                return True
            except (AttributeError, TypeError, ValueError) as exc:
                log_byte_error(
                    byte_log,
                    logging.DEBUG,
                    "USearch remove path failed; rebuilding index.",
                    error=exc,
                    code=ByteErrorCode.STORAGE_WRITE,
                    boundary="storage.write",
                    stage="usearch_remove",
                    exc_info=False,
                )
        self.rebuild()
        return True

    def flush(self) -> None:
        Path(self._index_file_path).parent.mkdir(parents=True, exist_ok=True)
        self._index.save(self._index_file_path)
        self._persist_vectors()

    def close(self) -> None:
        self.flush()

    def count(self) -> Any:
        return len(self._vectors)

    def get_embeddings(self, data_id) -> Any | None:
        vector = self._vectors.get(int(data_id))
        if vector is None:
            return None
        return np.asarray(vector, dtype="float32").copy()

    def update_embeddings(self, data_id, emb: np.ndarray) -> bool:
        vector = np.asarray(emb, dtype="float32")
        self._vectors[int(data_id)] = vector
        self.rebuild()
        return True
