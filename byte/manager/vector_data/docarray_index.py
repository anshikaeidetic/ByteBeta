from typing import Any

import numpy as np

from byte.manager.vector_data.base import VectorBase, VectorData
from byte.utils import load_optional_module


def _load_docarray_stack() -> tuple[Any, ...]:
    docarray = load_optional_module("docarray", package="docarray")
    docarray_index = load_optional_module("docarray.index", package="docarray")
    docarray_typing = load_optional_module("docarray.typing", package="docarray")

    class DocarrayVectorData(docarray.BaseDoc):
        """Class representing a vector data element with an ID and associated data."""

        id: int
        data: docarray_typing.NdArray

    return docarray.DocList, docarray_index.InMemoryExactNNIndex, DocarrayVectorData


class DocArrayIndex(VectorBase):
    """
    Class representing in-memory exact nearest neighbor index for vector search.

    :param index_file_path: the path to docarray index, defaults to 'docarray_index.bin'.
    :type index_file_path: str
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    """

    def __init__(self, index_file_path: str, top_k: int) -> None:
        self._doc_list_cls, index_cls, self._doc_cls = _load_docarray_stack()
        self._index = index_cls[self._doc_cls](index_file_path=index_file_path)
        self._index_file_path = index_file_path
        self._top_k = top_k

    def mul_add(self, datas: list[VectorData]) -> None:
        """
        Add multiple vector data elements to the index.

        :param datas: A list of vector data elements to be added.
        """
        docs = self._doc_list_cls(
            self._doc_cls(id=data.id, data=data.data) for data in datas
        )
        self._index.index(docs)

    def search(self, data: np.ndarray, top_k: int = -1) -> list[tuple[float, int]] | None:
        """
        Search for the nearest vector data elements in the index.

        :param data: The query vector data.
        :param top_k: The number of top matches to return.
        :return: A list of tuples, each containing the match score and
            the ID of the matched vector data element.
        """
        if len(self._index) == 0:
            return None
        if top_k == -1:
            top_k = self._top_k
        docs, scores = self._index.find(data, search_field="data", limit=top_k)
        return list(zip(scores, docs.id))

    def rebuild(self, ids: list[int] | None = None) -> bool:
        """
        In the case of DocArrayIndex, the rebuild operation is not needed.
        """
        return True

    def delete(self, ids: list[str] | None) -> None:
        """
        Delete the specified vector data elements from the index.

        :param ids: A list of IDs of the vector data elements to be deleted.
        """
        if ids is not None:
            del self._index[ids]

    def flush(self) -> None:
        self._index.persist(self._index_file_path)

    def close(self) -> None:
        self.flush()
