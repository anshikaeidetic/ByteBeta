import importlib
import importlib.machinery
import io
import math
import sys
import wave
from types import ModuleType, SimpleNamespace

import torch

import byte.adapter._api_init as api_init


class _FakeProcessor:
    def __init__(self, sampling_rate: int = 16_000) -> None:
        self.feature_extractor = SimpleNamespace(sampling_rate=sampling_rate)

    def __call__(self, audio, sampling_rate, return_tensors) -> object:
        return {"input_values": torch.as_tensor(audio).unsqueeze(0)}


class _FakeModel:
    def __init__(self, hidden_size: int = 6) -> None:
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def __call__(self, **_) -> object:
        return SimpleNamespace(last_hidden_state=torch.ones((1, 2, self.config.hidden_size)))


class _FakeEncoder:
    def __init__(self, dimension: int = 6) -> None:
        self.dimension = dimension

    def to_embeddings(self, _data) -> object:
        return [0.0] * self.dimension


def _sine_wave_bytes(sample_rate: int = 16_000, duration_s: float = 0.1) -> io.BytesIO:
    frame_count = int(sample_rate * duration_s)
    amplitude = 16_000
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            sample = int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        handle.writeframes(frames)
    buffer.seek(0)
    return buffer


def _module_stub(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for attr_name, value in attributes.items():
        setattr(module, attr_name, value)
    return module


class _Resample:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __call__(self, waveform) -> object:
        return waveform


def _install_data2vec_stubs(monkeypatch) -> None:
    transformers_stub = _module_stub(
        "transformers",
        Data2VecAudioModel=type(
            "Data2VecAudioModel",
            (),
            {"from_pretrained": staticmethod(lambda _model_name: _FakeModel())},
        ),
        Wav2Vec2Processor=type(
            "Wav2Vec2Processor",
            (),
            {"from_pretrained": staticmethod(lambda _model_name: _FakeProcessor())},
        ),
    )
    torchaudio_stub = _module_stub(
        "torchaudio",
        load=lambda _audio_path: (_ for _ in ()).throw(RuntimeError("use wave fallback")),
        transforms=SimpleNamespace(Resample=_Resample),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_stub)
    sys.modules.pop("byte.embedding.data2vec", None)


def test_data2vec_audio(monkeypatch) -> None:
    _install_data2vec_stubs(monkeypatch)
    module = importlib.import_module("byte.embedding.data2vec")

    audio = _sine_wave_bytes()
    t = module.Data2VecAudio("facebook/data2vec-audio-base-960h")
    data = t.to_embeddings(audio)
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"

    monkeypatch.setattr(api_init, "Data2VecAudio", lambda **_: _FakeEncoder())
    audio = _sine_wave_bytes()
    t = api_init._get_model("data2vecaudio")
    data = t.to_embeddings(audio)
    assert len(data) == t.dimension, f"{len(data)}, {t.dimension}"
