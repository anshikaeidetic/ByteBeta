from io import BytesIO

import pytest

pytest.importorskip("vit")
pytest.importorskip("PIL")

from PIL import Image

from byte.adapter.api import _get_model
from byte.utils import import_pillow, import_vit


def _make_image() -> object:
    image = Image.new("RGB", (32, 32), color=(120, 80, 40))
    image_file = BytesIO()
    image.save(image_file, format="PNG")
    image_file.seek(0)
    return Image.open(image_file)


def test_timm() -> None:
    import_vit()
    import_pillow()

    from byte.embedding import ViT

    image = _make_image()
    encoder = ViT(model="google/vit-base-patch16-384")
    embed = encoder.to_embeddings(image)
    assert len(embed) == encoder.dimension

    encoder = _get_model(model_src="vit")
    embed = encoder.to_embeddings(_make_image())
    assert len(embed) == encoder.dimension


if __name__ == "__main__":
    test_timm()
