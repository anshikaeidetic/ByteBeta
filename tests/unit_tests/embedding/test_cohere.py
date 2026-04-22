import os
import types
from unittest.mock import patch

from byte.adapter.api import _get_model
from byte.embedding import Cohere
from byte.utils import import_cohere

import_cohere()


def test_embedding() -> None:
    os.environ["CO_API_KEY"] = "API"

    with patch("cohere.Client.embed") as mock_create:
        dimension = 4096
        mock_create.return_value = types.SimpleNamespace(embeddings=[[0] * dimension])
        c1 = Cohere()
        assert c1.dimension == dimension
        assert len(c1.to_embeddings("foo")) == dimension

    with patch("cohere.Client.embed") as mock_create:
        dimension = 512
        mock_create.return_value = types.SimpleNamespace(embeddings=[[0] * dimension])
        c1 = Cohere("foo")
        assert c1.dimension == dimension
        assert len(c1.to_embeddings("foo")) == dimension

    with patch("cohere.Client.embed") as mock_create:
        dimension = 4096
        mock_create.return_value = types.SimpleNamespace(embeddings=[[0] * dimension])
        c1 = _get_model("cohere")
        assert c1.dimension == dimension
        assert len(c1.to_embeddings("foo")) == dimension
