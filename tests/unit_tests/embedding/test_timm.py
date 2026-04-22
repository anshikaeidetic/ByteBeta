from io import BytesIO

import pytest

pytest.importorskip("timm")
pytest.importorskip("PIL")

from PIL import Image

from byte.adapter.api import _get_model
from byte.embedding import Timm


def _make_image_file() -> object:
    image = Image.new("RGB", (32, 32), color=(12, 34, 56))
    image_file = BytesIO()
    image.save(image_file, format="PNG")
    image_file.seek(0)
    return image_file


def test_timm() -> None:
    image_file = _make_image_file()

    encoder = Timm(model="resnet50")
    embed = encoder.to_embeddings(image_file)
    assert len(embed) == encoder.dimension

    encoder = _get_model(model_src="timm", model_config={"model": "resnet50"})
    embed = encoder.to_embeddings(_make_image_file())
    assert len(embed) == encoder.dimension


if __name__ == "__main__":
    test_timm()
