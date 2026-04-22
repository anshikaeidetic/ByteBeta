from typing import Any

from byte.embedding.base import BaseEmbedding
from byte.utils import lazy_optional_module
from byte.utils.error import ParamError

pil_image = lazy_optional_module("PIL.Image", package="pillow")
uform_module = lazy_optional_module("uform", package="uform==0.2.1")


class UForm(BaseEmbedding):
    """Generate multi-modal embeddings using pretrained models from UForm.

    :param model: model name, defaults to 'unum-cloud/uform-vl-english'.
    :type model: str
    :param embedding_type: type of embedding, defaults to 'text'. options: text, image
    :type embedding_type: str

    Example:
        .. code-block:: python

            from byte.embedding import UForm

            test_sentence = 'Hello, world.'
            encoder = UForm(model='unum-cloud/uform-vl-english')
            embed = encoder.to_embeddings(test_sentence)

            test_sentence = '什么是Github'
            encoder = UForm(model='unum-cloud/uform-vl-multilingual')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(
        self,
        model: str | Any = "unum-cloud/uform-vl-english",
        embedding_type: str = "text",
    ) -> None:
        if isinstance(model, str):
            self.__model = uform_module.get_model(model)
        else:
            self.__model = model
        self.__embedding_type = embedding_type
        if embedding_type == "text":
            self.__dimension = self.__model.text_encoder.proj.out_features
        elif embedding_type == "image":
            self.__dimension = self.__model.img_encoder.proj.out_features
        else:
            raise ParamError(f"Unknown embedding type: {embedding_type}")

    def to_embeddings(self, data: Any, **_) -> Any:
        """Generate embedding given text input or a path to a file.

        :param data: text in string, or a path to an image file.
        :type data: str

        :return: an embedding in shape of (dim,).
        """
        if self.__embedding_type == "image":
            data = pil_image.open(data)
            data = self.__model.preprocess_image(data)
            emb = self.__model.encode_image(data)
        else:
            data = self.__model.preprocess_text(data)
            emb = self.__model.encode_text(data)
        return emb.detach().numpy().flatten()

    @property
    def dimension(self) -> Any:
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
