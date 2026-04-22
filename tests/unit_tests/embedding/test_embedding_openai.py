import os
from unittest.mock import MagicMock, patch

from byte.adapter.api import _get_model
from byte.embedding import OpenAI


def test_embedding() -> object:
    os.environ["OPENAI_API_KEY"] = "API"

    def get_return_value(dimension) -> object:
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0] * dimension,
                    "index": 0,
                }
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8,
            },
        }

    def build_mock_client(dimension) -> object:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = get_return_value(dimension)
        return mock_client

    with patch("byte.embedding.openai.openai.OpenAI") as mock_openai_client:
        dimension = 1536
        mock_openai_client.return_value = build_mock_client(dimension)
        oa = OpenAI()
        assert oa.dimension == dimension
        assert len(oa.to_embeddings("foo")) == dimension

    with patch("byte.embedding.openai.openai.OpenAI") as mock_openai_client:
        dimension = 1536
        mock_openai_client.return_value = build_mock_client(dimension)
        oa = OpenAI(api_key="openai")
        assert oa.dimension == dimension
        assert len(oa.to_embeddings("foo")) == dimension

    with patch("byte.embedding.openai.openai.OpenAI") as mock_openai_client:
        dimension = 512
        mock_openai_client.return_value = build_mock_client(dimension)
        oa = OpenAI(model="test_embedding")
        assert oa.dimension == dimension
        assert len(oa.to_embeddings("foo")) == dimension

    with patch("byte.embedding.openai.openai.OpenAI") as mock_openai_client:
        dimension = 1536
        mock_openai_client.return_value = build_mock_client(dimension)
        oa = _get_model("openai")
        assert oa.dimension == dimension
        assert len(oa.to_embeddings("foo")) == dimension
