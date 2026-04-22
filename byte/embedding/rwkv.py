from typing import Any

import numpy as np

from byte.embedding.base import BaseEmbedding
from byte.utils import lazy_optional_module

torch = lazy_optional_module("torch", package="torch")
transformers = lazy_optional_module("transformers", package="transformers")


class Rwkv(BaseEmbedding):
    """Generate sentence embedding for given text using RWKV models.

    :param model: model name, defaults to 'sgugger/rwkv-430M-pile'. Check
      https://huggingface.co/docs/transformers/model_doc/rwkv for more avaliable models.
    :type model: str

    Example:
        .. code-block:: python

            from byte.embedding import Rwkv

            test_sentence = 'Hello, world.'
            encoder = Rwkv(model='sgugger/rwkv-430M-pile')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "sgugger/rwkv-430M-pile") -> None:
        self.model = transformers.RwkvModel.from_pretrained(model)
        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        try:
            self.__dimension = self.model.config.hidden_size
        except Exception:  # pylint: disable=W0703
            config = transformers.AutoConfig.from_pretrained(model)
            self.__dimension = config.hidden_size

    def to_embeddings(self, data, **_) -> Any:
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        inputs = self.tokenizer(data, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"])
        emb = outputs.last_hidden_state[0, 0, :].detach().float().cpu().numpy()
        return np.array(emb).astype("float32")

    @property
    def dimension(self) -> Any:
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
