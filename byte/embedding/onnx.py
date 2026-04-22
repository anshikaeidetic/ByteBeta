from typing import Any

import numpy as np

from byte.embedding.base import BaseEmbedding
from byte.utils import lazy_optional_module

onnxruntime = lazy_optional_module("onnxruntime", package="onnxruntime")
huggingface_hub = lazy_optional_module("huggingface_hub", package="huggingface-hub")
transformers = lazy_optional_module("transformers", package="transformers")

_MODEL_ALIASES = {
    "byte/paraphrase-albert-onnx": "GPTCache/paraphrase-albert-onnx",
}


class Onnx(BaseEmbedding):
    """Generate text embedding for given text using ONNX Model.

    Example:
        .. code-block:: python

            from byte.embedding import Onnx

            test_sentence = 'Hello, world.'
            encoder = Onnx(model="sentence-transformers/paraphrase-albert-small-v2")
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model="sentence-transformers/paraphrase-albert-small-v2") -> None:
        self.model = _MODEL_ALIASES.get(model, model)
        self._tokenizer_name = "sentence-transformers/paraphrase-albert-small-v2"
        self.tokenizer = None
        self.ort_session = None
        self._input_names: set[str] = set()
        self.__dimension = None

    def _ensure_loaded(self) -> None:
        if self.ort_session is not None and self.tokenizer is not None and self.__dimension is not None:
            return
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self._tokenizer_name)
        try:
            onnx_model_path = huggingface_hub.hf_hub_download(
                repo_id=self.model,
                filename="onnx/model.onnx",
            )
        except OSError:
            onnx_model_path = huggingface_hub.hf_hub_download(
                repo_id=self.model,
                filename="model.onnx",
            )
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        self._input_names = {item.name for item in self.ort_session.get_inputs()}
        config = transformers.AutoConfig.from_pretrained(self._tokenizer_name)
        self.__dimension = config.hidden_size

    def to_embeddings(self, data, **_) -> Any:
        """Generate embedding given text input.

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        self._ensure_loaded()
        encoded_text = self.tokenizer(
            data,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        ort_inputs = {}
        for name in self._input_names:
            if name in encoded_text:
                ort_inputs[name] = encoded_text[name].astype("int64")
            elif name == "token_type_ids":
                ort_inputs[name] = np.zeros_like(encoded_text["input_ids"], dtype="int64")

        ort_outputs = self.ort_session.run(None, ort_inputs)
        ort_feat = ort_outputs[0]
        emb = self.post_proc(ort_feat, ort_inputs["attention_mask"])
        return emb.flatten()

    def post_proc(self, token_embeddings, attention_mask) -> Any:
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1).repeat(token_embeddings.shape[-1], -1).astype(float)
        )
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

    @property
    def dimension(self) -> Any:
        """Embedding dimension.

        :return: embedding dimension
        """
        self._ensure_loaded()
        return self.__dimension
