from byte.adapter.api import _get_model
from byte.embedding import PaddleNLP


def test_paddlenlp() -> None:
    t = PaddleNLP("ernie-3.0-nano-zh")
    dimension = t.dimension
    data = t.to_embeddings("中国")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"

    t = _get_model(model_src="paddlenlp", model_config={"model": "ernie-3.0-nano-zh"})
    dimension = t.dimension
    data = t.to_embeddings("中国")
    assert len(data) == dimension, f"{len(data)}, {t.dimension}"
