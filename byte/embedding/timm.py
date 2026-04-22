from typing import Any

import numpy as np

from byte.embedding.base import BaseEmbedding
from byte.utils import lazy_optional_module

torch = lazy_optional_module("torch", package="torch")
pil_image = lazy_optional_module("PIL.Image", package="pillow")
timm_data = lazy_optional_module("timm.data", package="timm")
timm_models = lazy_optional_module("timm.models", package="timm")


class Timm(BaseEmbedding):
    """Generate image embedding for given image using pretrained models from Timm.

    :param model: model name, defaults to 'resnet34'.
    :type model: str

    Example:
        .. code-block:: python

            import requests
            from io import BytesIO
            from byte.embedding import Timm


            encoder = Timm(model='resnet50')
            embed = encoder.to_embeddings('path/to/image')
    """

    def __init__(self, model: str = "resnet18", device: str = "default") -> None:
        if device == "default":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_name = model
        self.model = timm_models.create_model(model_name=model, pretrained=True)
        self.model.eval()

        try:
            self.__dimension = self.model.embed_dim
        except Exception:  # pylint: disable=W0703
            self.__dimension = None

    def to_embeddings(self, data, skip_preprocess: bool = False, **_) -> Any:
        """Generate embedding given image data

        :param data: image path.
        :type data: str
        :param skip_preprocess: flag to skip preprocess, defaults to False, enable this if the input data is torch.tensor.
        :type skip_preprocess: bool

        :return: an image embedding in shape of (dim,).
        """
        if not skip_preprocess:
            data = self.preprocess(data)
        if data.dim() == 3:
            data = data.unsqueeze(0)
        feats = self.model.forward_features(data)
        emb = self.post_proc(feats).squeeze(0).detach().numpy()

        return np.array(emb).astype("float32")

    def post_proc(self, features) -> Any:
        features = features.to("cpu")
        if features.dim() == 3:
            features = features[:, 0]
        if features.dim() == 4:
            global_pool = torch.nn.AdaptiveAvgPool2d(1)
            features = global_pool(features)
            features = features.flatten(1)
        assert features.dim() == 2, f"Invalid output dim {features.dim()}"
        return features

    def preprocess(self, image_path) -> Any:
        """Load image from path and then transform image to torch.tensor with model transformations.

        :param image_path: image path.
        :type image_path: str

        :return: an image tensor (without batch size).
        """
        data_cfg = timm_data.resolve_data_config(self.model.pretrained_cfg)
        transform = timm_data.create_transform(**data_cfg)

        image = pil_image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        return image_tensor

    @property
    def dimension(self) -> Any:
        """Embedding dimension.

        :return: embedding dimension
        """
        if not self.__dimension:
            input_size = self.model.pretrained_cfg["input_size"]
            dummy_input = torch.rand((1,) + input_size)
            feats = self.to_embeddings(dummy_input, skip_preprocess=True)
            self.__dimension = feats.shape[0]
        return self.__dimension
