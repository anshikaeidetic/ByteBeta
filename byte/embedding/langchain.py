from typing import Any

import numpy as np

from byte.embedding.base import BaseEmbedding
from byte.utils import load_optional_attr


def _load_embeddings_type() -> Any:
    try:
        return load_optional_attr(
            "langchain_core.embeddings",
            "Embeddings",
            package="langchain-core",
        )
    except ModuleNotFoundError:
        return load_optional_attr(
            "langchain.embeddings.base",
            "Embeddings",
            package="langchain-core",
        )


class LangChain(BaseEmbedding):
    """Generate text embedding for given text using LangChain.

    :param embeddings: the LangChain Embeddings object.
    :type embeddings: Embeddings
    :param dimension: The vector dimension after embedding is calculated by
        calling embed once by default.  If you know the dimension, pass it here
        to skip that extra call.
    :type dimension: int

    Example:
        .. code-block:: python

            from byte.embedding import LangChain
            from langchain_openai import OpenAIEmbeddings

            test_sentence = 'Hello, world.'
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            encoder = LangChain(embeddings=embeddings)
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, embeddings: Any, dimension: int = 0) -> None:
        _load_embeddings_type()
        self._embeddings: Any = embeddings
        self._dimension: int = (
            dimension if dimension != 0 else len(self._embeddings.embed_query("foo"))
        )

    def to_embeddings(self, data, **kwargs) -> Any:
        vector_data = self._embeddings.embed_query(data)
        return np.array(vector_data).astype("float32")

    @property
    def dimension(self) -> int:
        return self._dimension
