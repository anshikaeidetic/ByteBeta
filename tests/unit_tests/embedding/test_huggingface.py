from types import SimpleNamespace

import torch

from byte.adapter.api import _get_model
from byte.embedding import Huggingface


class _FakeTokenizer:
    pad_token = None

    def __call__(self, data, **_) -> object:
        batch = len(data) if isinstance(data, list) else 1
        return {
            "input_ids": torch.ones((batch, 3), dtype=torch.long),
            "attention_mask": torch.ones((batch, 3), dtype=torch.long),
        }


class _FakeModel:
    def __init__(self, hidden_size: int = 4) -> None:
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def eval(self) -> None:
        return None

    def __call__(self, **_) -> object:
        return SimpleNamespace(
            last_hidden_state=torch.tensor(
                [[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]]
            )
        )


def test_huggingface(monkeypatch) -> None:
    monkeypatch.setattr(
        "byte.embedding.huggingface.AutoModel.from_pretrained",
        lambda model: _FakeModel(),
    )
    monkeypatch.setattr(
        "byte.embedding.huggingface.AutoTokenizer.from_pretrained",
        lambda model: _FakeTokenizer(),
    )

    t = Huggingface("distilbert-base-uncased")
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"

    t = _get_model(model_src="huggingface", model_config={"model": "distilbert-base-uncased"})
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"
