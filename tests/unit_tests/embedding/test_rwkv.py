import importlib
import importlib.machinery
import sys
from types import ModuleType, SimpleNamespace

import torch

import byte.adapter._api_init as api_init


class _FakeTokenizer:
    def __call__(self, data, **_) -> object:
        return {"input_ids": torch.ones((1, 3), dtype=torch.long)}


class _FakeModel:
    def __init__(self, hidden_size: int = 5) -> None:
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def eval(self) -> None:
        return None

    def __call__(self, input_ids) -> object:
        return SimpleNamespace(last_hidden_state=torch.ones((1, 3, self.config.hidden_size)))


class _FakeEncoder:
    def __init__(self, dimension: int = 5) -> None:
        self.dimension = dimension

    def to_embeddings(self, _data) -> object:
        return [0.0] * self.dimension


def _module_stub(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for attr_name, value in attributes.items():
        setattr(module, attr_name, value)
    return module


def _install_rwkv_stubs(monkeypatch) -> None:
    transformers_stub = _module_stub(
        "transformers",
        AutoTokenizer=type(
            "AutoTokenizer",
            (),
            {"from_pretrained": staticmethod(lambda _model_name: _FakeTokenizer())},
        ),
        RwkvModel=type(
            "RwkvModel",
            (),
            {"from_pretrained": staticmethod(lambda _model_name: _FakeModel())},
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    sys.modules.pop("byte.embedding.rwkv", None)


def test_rwkv(monkeypatch) -> None:
    _install_rwkv_stubs(monkeypatch)
    module = importlib.import_module("byte.embedding.rwkv")

    t = module.Rwkv("sgugger/rwkv-430M-pile")
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"

    monkeypatch.setattr(api_init, "Rwkv", lambda **_: _FakeEncoder())
    t = api_init._get_model(model_src="rwkv", model_config={"model": "sgugger/rwkv-430M-pile"})
    data = t.to_embeddings("foo")
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"
