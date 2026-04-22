"""Streaming, upload, and token helpers for the Byte OpenAI backend."""

import base64
from io import BytesIO
from typing import Any

from byte.utils.error import wrap_error
from byte.utils.response import get_image_from_openai_b64, get_image_from_openai_url
from byte.utils.token import token_counter

from ._openai_response import _openai_response_to_dict


async def async_iter(input_list) -> Any:
    for item in input_list:
        yield item


def _stream_generator(response) -> Any:
    """Convert openai v2.x stream to dict-yielding generator."""
    for chunk in response:
        yield _openai_response_to_dict(chunk)


async def _async_stream_generator(response) -> Any:
    """Convert openai v2.x async stream to dict-yielding async generator."""
    async for chunk in response:
        yield _openai_response_to_dict(chunk)


def _maybe_wrap_openai_error(exc: Exception) -> Exception | None:
    """Wrap OpenAI SDK errors without forcing the SDK into mock-only paths."""

    try:
        import openai as _openai  # pylint: disable=C0415
    except ModuleNotFoundError:
        return None

    if isinstance(exc, _openai.OpenAIError):
        return wrap_error(exc)
    return None


class _NamedBytesIO(BytesIO):
    def __init__(self, payload: bytes, name: str) -> None:
        super().__init__(payload)
        self.name = name

    def peek(self, size: int = -1) -> bytes:
        position = self.tell()
        data = self.read() if size is None or size < 0 else self.read(size)
        self.seek(position)
        return data


def _materialize_upload(file_obj) -> dict[str, Any]:
    if isinstance(file_obj, dict):
        payload = file_obj.get("bytes", b"")
        return {
            "name": str(file_obj.get("name") or "upload.bin"),
            "bytes": payload if isinstance(payload, bytes) else bytes(payload),
        }

    if isinstance(file_obj, (bytes, bytearray)):
        return {"name": "upload.bin", "bytes": bytes(file_obj)}

    name = str(getattr(file_obj, "name", "upload.bin") or "upload.bin")
    if hasattr(file_obj, "read"):
        position = file_obj.tell() if hasattr(file_obj, "tell") else None
        payload = file_obj.read()
        if position is not None and hasattr(file_obj, "seek"):
            file_obj.seek(position)
    else:
        payload = file_obj.peek()
    return {"name": name, "bytes": payload if isinstance(payload, bytes) else bytes(payload)}


def _open_upload(file_payload) -> Any:
    if isinstance(file_payload, dict):
        return _NamedBytesIO(
            file_payload.get("bytes", b""),
            str(file_payload.get("name") or "upload.bin"),
        )
    return file_payload


def _extract_image_b64(llm_data) -> Any:
    try:
        img_b64 = get_image_from_openai_b64(llm_data)
        return img_b64.encode("ascii") if isinstance(img_b64, str) else img_b64
    except Exception:  # pylint: disable=broad-except
        return get_image_from_openai_url(llm_data)


def _extract_audio_bytes(response) -> Any:
    if isinstance(response, bytes):
        return response

    if hasattr(response, "read"):
        return response.read()

    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content

    if isinstance(response, dict):
        audio_data = response.get("audio")
        if isinstance(audio_data, bytes):
            return audio_data
        if isinstance(audio_data, str):
            try:
                return base64.b64decode(audio_data)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError("OpenAI speech audio field is not valid base64") from exc

        raw_data = response.get("data")
        if isinstance(raw_data, bytes):
            return raw_data
        if isinstance(raw_data, str):
            try:
                return base64.b64decode(raw_data)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError("OpenAI speech data field is not valid base64") from exc

    raise ValueError("Unsupported OpenAI speech response type")


def _num_tokens_from_messages(messages) -> Any:
    """Returns the number of tokens used by a list of messages."""
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += token_counter(value)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
