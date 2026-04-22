"""Response conversion and cached-response helpers for the Byte OpenAI backend."""

import base64
import os
import time
from io import BytesIO
from typing import Any


def _openai_response_to_dict(response) -> Any:
    """Convert an openai v2.x response object to a dict for backward compat."""
    if isinstance(response, dict):
        return response
    try:
        return response.model_dump()
    except AttributeError:
        return response

def _construct_resp_from_cache(return_message, saved_token) -> dict[str, Any]:
    return {
        "byte": True,
        "saved_token": saved_token,
        "choices": [
            {
                "message": {"role": "assistant", "content": return_message},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "chat.completion",
    }


def _construct_stream_resp_from_cache(return_message, saved_token) -> list[Any]:
    created = int(time.time())
    return [
        {
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}],
            "created": created,
            "object": "chat.completion.chunk",
        },
        {
            "choices": [
                {
                    "delta": {"content": return_message},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": created,
            "object": "chat.completion.chunk",
        },
        {
            "byte": True,
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": created,
            "object": "chat.completion.chunk",
            "saved_token": saved_token,
        },
    ]


def _construct_text_from_cache(return_text) -> dict[str, Any]:
    return {
        "byte": True,
        "choices": [
            {
                "text": return_text,
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "text_completion",
    }


def _construct_image_create_resp_from_cache(image_data, response_format, size) -> dict[str, Any]:
    from byte.utils import import_pillow  # pylint: disable=C0415

    import_pillow()
    from PIL import Image as PILImage  # pylint: disable=C0415

    img_bytes = base64.b64decode(image_data)
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    img = PILImage.open(img_file)
    new_size = tuple(int(a) for a in size.split("x"))
    output_bytes = img_bytes
    output_ext = ".png"
    if new_size != img.size:
        img = img.resize(new_size)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        buffered.seek(0)
        output_bytes = buffered.getvalue()
        output_ext = ".jpeg"
    else:
        image_format = (img.format or "").lower()
        if image_format in {"jpg", "jpeg"}:
            output_ext = ".jpeg"
        elif image_format:
            output_ext = f".{image_format}"

    if response_format == "url":
        target_url = os.path.abspath(str(int(time.time())) + output_ext)
        with open(target_url, "wb") as f:
            f.write(output_bytes)
        image_data = target_url
    elif response_format == "b64_json":
        image_data = base64.b64encode(output_bytes).decode("ascii")
    else:
        raise AttributeError(
            f"Invalid response_format: {response_format} is not one of ['url', 'b64_json']"
        )

    return {
        "byte": True,
        "created": int(time.time()),
        "data": [{response_format: image_data}],
    }


def _construct_audio_text_from_cache(return_text) -> dict[str, Any]:
    return {
        "byte": True,
        "text": return_text,
    }
