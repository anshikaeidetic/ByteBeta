import wave
from typing import Any

import numpy as np

from byte.embedding.base import BaseEmbedding
from byte.utils import lazy_optional_module

torch = lazy_optional_module("torch", package="torch")
torchaudio = lazy_optional_module("torchaudio", package="torchaudio")
transformers = lazy_optional_module("transformers", package="transformers")


class Data2VecAudio(BaseEmbedding):
    """Generate audio embedding for given audio using pretrained models from Data2Vec.

    :param model: model name, defaults to 'facebook/data2vec-audio-base-960h'.
    :type model: str

    Example:
        .. code-block:: python

            from byte.embedding import Data2VecAudio

            audio_file = 'test.wav'
            encoder = Data2VecAudio(model='facebook/data2vec-audio-base-960h')
            embed = encoder.to_embeddings(audio_file)
    """

    def __init__(self, model_name="facebook/data2vec-audio-base-960h") -> None:
        self.model = transformers.Data2VecAudioModel.from_pretrained(model_name)
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(model_name)
        self.__dimension = self.model.config.hidden_size
        self.sr = self.processor.feature_extractor.sampling_rate

    def to_embeddings(self, data, **_) -> Any:
        """Generate embedding given text input

        :param data: path to audio file.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        audio = self.load_audio(data, self.sr)
        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        feat = last_hidden_states[:, -1, :].flatten().detach().cpu().numpy()
        return np.array(feat).astype("float32")

    def load_audio(self, audio_path, target_sr) -> Any:
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except (ImportError, OSError, RuntimeError):
            waveform, sample_rate = self._load_audio_with_wave(audio_path)
        waveform = torch.mean(waveform, axis=0)
        transform = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = transform(waveform)
        return waveform

    @staticmethod
    def _load_audio_with_wave(audio_path) -> tuple[Any, ...]:
        if hasattr(audio_path, "seek"):
            audio_path.seek(0)
        with wave.open(audio_path, "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())

        dtype = {
            1: np.uint8,
            2: np.int16,
            4: np.int32,
        }.get(sample_width)
        if dtype is None:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        waveform = np.frombuffer(frames, dtype=dtype)
        if channels > 1:
            waveform = waveform.reshape(-1, channels).T
        else:
            waveform = waveform.reshape(1, -1)

        if dtype == np.uint8:
            waveform = (waveform.astype("float32") - 128.0) / 128.0
        else:
            max_value = float(np.iinfo(dtype).max) or 1.0
            waveform = waveform.astype("float32") / max_value

        return torch.from_numpy(waveform), sample_rate

    @property
    def dimension(self) -> Any:
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
