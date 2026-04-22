
"""Compatibility facade for Byte's Google Gemini backend."""

from __future__ import annotations

from byte.adapter.adapter import aadapt, adapt
from byte.adapter.base import BaseCacheLLM
from byte.adapter.prompt_cache_bridge import (
    apply_native_prompt_cache,
    strip_native_prompt_cache_hints,
)
from byte.manager.scalar_data.base import Answer, DataType
from byte.utils.multimodal import extract_content_parts, extract_text_content, materialize_upload

from .gemini_audio import Audio, Speech
from .gemini_chat import ChatCompletion
from .gemini_clients import (
    _build_namespace,
    _create_async_client,
    _create_client,
    _get_async_client,
    _get_client,
    _get_genai_types,
)
from .gemini_images import Image
from .gemini_messages import _build_content_config, _build_genai_part, _convert_messages
from .gemini_responses import (
    _async_iter,
    _async_stream_generator,
    _construct_audio_text_from_cache,
    _construct_image_from_cache,
    _construct_resp_from_cache,
    _construct_speech_from_cache,
    _construct_stream_resp_from_cache,
    _extract_gemini_audio_bytes,
    _gemini_image_response_to_openai,
    _image_bytes_to_response,
    _response_to_openai_format,
    _sync_stream_generator,
)

__all__ = [
    "Answer",
    "Audio",
    "BaseCacheLLM",
    "ChatCompletion",
    "DataType",
    "Image",
    "Speech",
    "_async_iter",
    "_async_stream_generator",
    "_build_content_config",
    "_build_genai_part",
    "_build_namespace",
    "_construct_audio_text_from_cache",
    "_construct_image_from_cache",
    "_construct_resp_from_cache",
    "_construct_speech_from_cache",
    "_construct_stream_resp_from_cache",
    "_convert_messages",
    "_create_async_client",
    "_create_client",
    "_extract_gemini_audio_bytes",
    "_gemini_image_response_to_openai",
    "_get_async_client",
    "_get_client",
    "_get_genai_types",
    "_image_bytes_to_response",
    "_response_to_openai_format",
    "_sync_stream_generator",
    "aadapt",
    "adapt",
    "apply_native_prompt_cache",
    "extract_content_parts",
    "extract_text_content",
    "materialize_upload",
    "strip_native_prompt_cache_hints",
]
