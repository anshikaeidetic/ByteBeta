import os
from typing import Any

import numpy as np

from byte.embedding.base import BaseEmbedding
from byte.utils import lazy_optional_module

openai = lazy_optional_module("openai", package="openai")


def _load_openai_sdk() -> Any:
    return openai


def _build_client(api_key=None, api_base=None) -> Any | None:
    openai = _load_openai_sdk()
    if not hasattr(openai, "OpenAI"):
        if api_key:
            openai.api_key = api_key
        if api_base:
            openai.api_base = api_base
        return None

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if api_base:
        client_kwargs["base_url"] = api_base
    return openai.OpenAI(**client_kwargs)


def _extract_embedding(response) -> Any:
    if isinstance(response, dict):
        data = response["data"][0]
        return data["embedding"] if isinstance(data, dict) else data.embedding
    data = response.data[0]
    return data["embedding"] if isinstance(data, dict) else data.embedding


class OpenAI(BaseEmbedding):
    """Generate text embedding for given text using OpenAI.

    Supports the latest ``text-embedding-3-small``, ``text-embedding-3-large``,
    and the legacy ``text-embedding-ada-002`` models.

    :param model: model name, defaults to ``text-embedding-3-small``.
    :type model: str
    :param api_key: OpenAI API Key.  When ``None`` the key is read from the
        ``OPENAI_API_KEY`` environment variable.
    :type api_key: str
    :param api_base: Optional custom API base URL (for Azure, proxies, etc.).
    :type api_base: str
    :param dimensions: Optional explicit output dimensionality (only
        ``text-embedding-3-*`` models support this).  Leave ``None`` to use the
        model's native dimension.
    :type dimensions: int

    Example:
        .. code-block:: python

            from byte.embedding import OpenAI

            test_sentence = 'Hello, world.'
            encoder = OpenAI(api_key='your_openai_key')
            embed = encoder.to_embeddings(test_sentence)
    """

    _DIM_DICT = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = _build_client(api_key=api_key, api_base=api_base)
        self.model = model
        self._dimensions = dimensions  # user-requested dim override

        if dimensions:
            self.__dimension = dimensions
        elif model in self._DIM_DICT:
            self.__dimension = self._DIM_DICT[model]
        else:
            self.__dimension = None

    def to_embeddings(self, data, **_) -> Any:
        """Generate embedding given text input.

        :param data: text in string.
        :type data: str
        :return: a text embedding in shape of (dim,).
        """
        kwargs = {"model": self.model, "input": data}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        openai = _load_openai_sdk()
        if self._client is not None:
            response = self._client.embeddings.create(**kwargs)
        else:
            response = openai.Embedding.create(**kwargs)
        return np.array(_extract_embedding(response)).astype("float32")

    @property
    def dimension(self) -> Any:
        """Embedding dimension.

        :return: embedding dimension
        """
        if not self.__dimension:
            foo_emb = self.to_embeddings("foo")
            self.__dimension = len(foo_emb)
        return self.__dimension

    @staticmethod
    def dim_dict() -> Any:
        return OpenAI._DIM_DICT
