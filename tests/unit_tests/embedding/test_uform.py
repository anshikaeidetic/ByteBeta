from io import BytesIO

import pytest

pytest.importorskip("uform")
pytest.importorskip("PIL")

from PIL import Image

from byte.adapter.api import _get_model
from byte.utils import import_pillow, import_uform
from byte.utils.error import ParamError

import_uform()
import_pillow()


def _make_image_file() -> object:
    image = Image.new("RGB", (32, 32), color=(90, 45, 12))
    image_file = BytesIO()
    image.save(image_file, format="PNG")
    image_file.seek(0)
    return image_file


def _load_uform_encoder(**model_config) -> object:
    try:
        return _get_model("uform", model_config=model_config)
    except RuntimeError as exc:
        pytest.skip(f"uform model is unavailable in this environment: {exc}")


def test_uform() -> None:
    encoder = _load_uform_encoder()
    embed = encoder.to_embeddings("Hello, world.")
    assert len(embed) == encoder.dimension

    encoder = _load_uform_encoder(embedding_type="image")
    embed = encoder.to_embeddings(_make_image_file())
    assert len(embed) == encoder.dimension

    is_exception = False
    try:
        _get_model("uform", model_config={"embedding_type": "foo"})
    except ParamError:
        is_exception = True
    assert is_exception
