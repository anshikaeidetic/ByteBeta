import pytest

from byte.embedding import LangChain
from byte.utils import import_langchain

import_langchain()
pytest.importorskip("langchain_core")
from langchain_core.embeddings import FakeEmbeddings


def test_langchain_embedding() -> None:
    size = 10
    lc = LangChain(embeddings=FakeEmbeddings(size=size))
    data = lc.to_embeddings("foo")
    assert len(data) == size

    lc = LangChain(embeddings=FakeEmbeddings(size=size), dimension=size)
    data = lc.to_embeddings("foo")
    assert len(data) == size
