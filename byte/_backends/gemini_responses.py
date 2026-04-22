
"""Response, cache, and stream conversion helpers for Gemini."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any


def _response_to_openai_format(response, model="") -> Any:
    """Convert Gemini response to OpenAI-compatible dict format."""
    if isinstance(response, dict):
        return response

    text_content = ""
    if hasattr(response, "text"):
        text_content = response.text or ""
    elif hasattr(response, "candidates") and response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text"):
                text_content += part.text

    usage = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(um, "total_token_count", 0) or 0,
        }
    else:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return {
        "byte_provider": "gemini",
        "choices": [
            {
                "message": {"role": "assistant", "content": text_content},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "model": model,
        "object": "chat.completion",
        "usage": usage,
    }

def _extract_gemini_audio_bytes(response) -> tuple[Any, ...]:
    if isinstance(response, dict):
        candidates = response.get("candidates", []) or []
    else:
        candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if isinstance(candidate, dict):
            parts = candidate.get("content", {}).get("parts", []) or []
        else:
            parts = getattr(content, "parts", None) or []
        for part in parts:
            if isinstance(part, dict):
                inline_data = part.get("inline_data")
            else:
                inline_data = getattr(part, "inline_data", None)
            if not inline_data:
                continue
            if isinstance(inline_data, dict):
                payload = inline_data.get("data")
                mime_type = inline_data.get("mime_type")
            else:
                payload = getattr(inline_data, "data", None)
                mime_type = getattr(inline_data, "mime_type", None)
            if payload:
                return payload, mime_type
    raise ValueError("Gemini response did not contain audio bytes.")


def _image_bytes_to_response(images, response_format) -> dict[str, Any]:
    payload = []
    for index, item in enumerate(images):
        image_bytes = item.get("bytes") or b""
        mime_type = item.get("mime_type") or "image/png"
        if response_format == "b64_json":
            payload.append({"b64_json": base64.b64encode(image_bytes).decode("ascii")})
            continue
        if response_format != "url":
            raise AttributeError(
                f"Invalid response_format: {response_format} is not one of ['url', 'b64_json']"
            )

        extension = ".png"
        if "/" in mime_type:
            extension = f".{mime_type.split('/', 1)[1]}"
        target = os.path.abspath(f"{int(time.time())}_{index}{extension}")
        with open(target, "wb") as file_obj:
            file_obj.write(image_bytes)
        payload.append({"url": target})

    return {
        "created": int(time.time()),
        "byte_provider": "gemini",
        "data": payload,
    }


def _gemini_image_response_to_openai(response, response_format) -> Any:
    if isinstance(response, dict):
        generated = response.get("generated_images", []) or []
    else:
        generated = getattr(response, "generated_images", None) or []
    images = []
    for item in generated:
        if isinstance(item, dict):
            image = item.get("image", {}) or {}
            image_bytes = image.get("image_bytes")
            mime_type = image.get("mime_type") or "image/png"
        else:
            image = getattr(item, "image", None)
            image_bytes = getattr(image, "image_bytes", None)
            mime_type = getattr(image, "mime_type", None) or "image/png"
        if image_bytes:
            images.append({"bytes": image_bytes, "mime_type": mime_type})
    return _image_bytes_to_response(images, response_format)


def _construct_audio_text_from_cache(return_text) -> dict[str, Any]:
    return {
        "byte": True,
        "text": return_text,
    }


def _construct_speech_from_cache(cache_data, response_format) -> dict[str, Any]:
    payload = json.loads(cache_data)
    return {
        "byte": True,
        "audio": base64.b64decode(payload["audio"]),
        "format": payload.get("format", response_format),
        "mime_type": payload.get("mime_type"),
    }


def _construct_image_from_cache(cache_data, response_format) -> Any:
    payload = json.loads(cache_data)
    images = [
        {
            "bytes": base64.b64decode(item["b64"]),
            "mime_type": item.get("mime_type") or "image/png",
        }
        for item in payload.get("images", [])
    ]
    response = _image_bytes_to_response(images, response_format)
    response["byte"] = True
    return response

def _sync_stream_generator(response, model="") -> Any:
    """Convert Gemini stream chunks to OpenAI-compatible chunk dicts."""
    for chunk in response:
        text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
        yield {
            "choices": [
                {
                    "delta": {"content": text},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "model": model,
            "object": "chat.completion.chunk",
        }
    # Final stop chunk
    yield {
        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
        "created": int(time.time()),
        "model": model,
        "object": "chat.completion.chunk",
    }


async def _async_stream_generator(response, model="") -> Any:
    """Convert Gemini async stream chunks to OpenAI-compatible chunk dicts."""
    async for chunk in response:
        text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
        yield {
            "choices": [
                {
                    "delta": {"content": text},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "model": model,
            "object": "chat.completion.chunk",
        }
    yield {
        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
        "created": int(time.time()),
        "model": model,
        "object": "chat.completion.chunk",
    }


def _construct_resp_from_cache(return_message) -> dict[str, Any]:
    return {
        "byte": True,
        "byte_provider": "gemini",
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


def _construct_stream_resp_from_cache(return_message) -> list[Any]:
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
        },
    ]


async def _async_iter(items) -> Any:
    for item in items:
        yield item

__all__ = [
    "_async_iter",
    "_async_stream_generator",
    "_construct_audio_text_from_cache",
    "_construct_image_from_cache",
    "_construct_resp_from_cache",
    "_construct_speech_from_cache",
    "_construct_stream_resp_from_cache",
    "_extract_gemini_audio_bytes",
    "_gemini_image_response_to_openai",
    "_image_bytes_to_response",
    "_response_to_openai_format",
    "_sync_stream_generator",
]
