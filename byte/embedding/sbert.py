from typing import Any

import numpy as np

from byte.embedding.base import BaseEmbedding
from byte.utils import lazy_optional_module

sentence_transformers = lazy_optional_module(
    "sentence_transformers",
    package="sentence-transformers",
)


class SBERT(BaseEmbedding):
    """Generate sentence embedding for given text using pretrained models of Sentence Transformers.

    :param model: model name, defaults to 'all-MiniLM-L6-v2'.
    :type model: str

    Example:
        .. code-block:: python

            from byte.embedding import SBERT

            test_sentence = 'Hello, world.'
            encoder = SBERT('all-MiniLM-L6-v2')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.model = sentence_transformers.SentenceTransformer(model)
        self.model.eval()
        self.__dimension = None

    def to_embeddings(self, data, **_) -> Any:
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        if not isinstance(data, list):
            data = [data]
        emb = self.model.encode(data).squeeze(0)

        if not self.__dimension:
            self.__dimension = len(emb)
        return np.array(emb).astype("float32")

    @property
    def dimension(self) -> Any:
        """Embedding dimension.

        :return: embedding dimension
        """
        if not self.__dimension:
            self.__dimension = len(self.to_embeddings("foo"))
        return self.__dimension
