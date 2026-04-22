import numpy as np
import pytest

from byte.adapter.api import _get_eval
from byte.similarity_evaluation import SequenceMatchEvaluation
from byte.similarity_evaluation.sequence_match import reweight


def normalize(vec) -> object:
    norm = np.linalg.norm(vec)
    return vec / norm


def _test_evaluation(evaluation) -> None:
    evaluation = SequenceMatchEvaluation([0.1, 0.2, 0.7], "onnx")
    evaluation.evaluation(
        {"question": "USER:foo1\nUSER:foo2\nUSER:foo3\n"},
        {"question": "USER:foo1\nUSER:foo2\nUSER:foo3\n"},
    )
    evaluation.evaluation(
        {"question": "USER:foo1\nUSER:foo2\nUSER:foo3\n"},
        {"question": "USER:foo1\nUSER:foo2\n"},
    )
    evaluation = SequenceMatchEvaluation([0.2, 0.8], "onnx")
    evaluation.evaluation(
        {"question": "USER:foo1\nUser:foo2\nUser:foo3\n"},
        {"question": "USER:foo1\nUser:foo2\n"},
    )
    assert True


@pytest.mark.requires_feature("onnx")
def test_sequence_match() -> None:
    evaluation = SequenceMatchEvaluation([0.1, 0.2, 0.7], "onnx")
    evaluation.range()
    _test_evaluation(evaluation)


@pytest.mark.requires_feature("onnx")
def test_get_eval() -> None:
    evaluation = _get_eval(
        strategy="sequence_match",
        kws={
            "embedding_extractor": "onnx",
            "weights": [0.1, 0.2, 0.7],
            "embedding_config": {"model": "byte/paraphrase-albert-onnx"},
        },
    )
    _test_evaluation(evaluation)


def test_reweigth() -> None:
    ws = reweight([0.7, 0.2, 0.1], 4)
    assert len(ws) == 3
    ws = reweight([0.7, 0.2, 0.1], 3)
    assert len(ws) == 3
    ws = reweight([0.7, 0.2, 0.1], 2)
    assert len(ws) == 2
    ws = reweight([0.7, 0.2, 0.1], 1)
    assert len(ws) == 1


if __name__ == "__main__":
    test_sequence_match()
