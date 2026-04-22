import os
from types import SimpleNamespace
from unittest.mock import patch

from byte.adapter.api import _get_eval
from byte.utils import import_cohere

import_cohere()


def test_cohere_rerank() -> None:
    os.environ["CO_API_KEY"] = "API"

    evaluation = _get_eval("cohere")

    min_value, max_value = evaluation.range()
    assert min_value < 0.001
    assert max_value > 0.999

    with patch("cohere.Client.rerank") as mock_create:
        mock_create.return_value = SimpleNamespace(results=[])
        evaluation = _get_eval("cohere")
        score = evaluation.evaluation(
            {"question": "What is the color of sky?"},
            {"answer": "the color of sky is blue"},
        )
        assert score < 0.01

    with patch("cohere.Client.rerank") as mock_create:
        mock_create.return_value = SimpleNamespace(
            results=[SimpleNamespace(relevance_score=0.9871293, index=0)]
        )
        evaluation = _get_eval("cohere")
        score = evaluation.evaluation(
            {"question": "What is the color of sky?"},
            {"answer": "the color of sky is blue"},
        )
        assert score > 0.9
