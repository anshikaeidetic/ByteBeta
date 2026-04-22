from byte.adapter.api import _get_model
from byte.embedding import Onnx


def test_onnx() -> None:
    t = Onnx()
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"

    t = _get_model("onnx")
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"
