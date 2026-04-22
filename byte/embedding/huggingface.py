"""Hugging Face embedding adapter with lazy optional dependency loading."""

from typing import Any

import numpy as np

from byte.embedding.base import BaseEmbedding
from byte.utils import lazy_optional_attr, lazy_optional_module

torch = lazy_optional_module("torch", package="torch")
AutoConfig = lazy_optional_attr("transformers", "AutoConfig", package="transformers")
AutoModel = lazy_optional_attr("transformers", "AutoModel", package="transformers")
AutoTokenizer = lazy_optional_attr("transformers", "AutoTokenizer", package="transformers")


class Huggingface(BaseEmbedding):
    """Generate sentence embedding for given text using pretrained models from Huggingface transformers.

    :param model: model name, defaults to 'distilbert-base-uncased'.
    :type model: str

    Example:
        .. code-block:: python

            from byte.embedding import Huggingface

            test_sentence = 'Hello, world.'
            encoder = Huggingface(model='distilbert-base-uncased')
            embed = encoder.to_embeddings(test_sentence)

            test_sentence = '什么是Github'
            huggingface = Huggingface(model='uer/albert-base-chinese-cluecorpussmall')
            embed = huggingface.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "distilbert-base-uncased") -> None:
        self.model = AutoModel.from_pretrained(model)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = "[PAD]"
        try:
            self.__dimension = self.model.config.hidden_size
        except AttributeError:
            config = AutoConfig.from_pretrained(model)
            self.__dimension = config.hidden_size

    def to_embeddings(self, data, **_) -> Any:
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        if not isinstance(data, list):
            data = [data]
        inputs = self.tokenizer(data, padding=True, truncation=True, return_tensors="pt")
        outs = self.model(**inputs).last_hidden_state
        emb = self.post_proc(outs, inputs).squeeze(0).detach().numpy()
        return np.array(emb).astype("float32")

    def post_proc(self, token_embeddings, inputs) -> Any:
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embs = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        return sentence_embs

    @property
    def dimension(self) -> Any:
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
