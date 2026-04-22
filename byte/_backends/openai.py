"""Compatibility facade for the Byte OpenAI backend."""

from byte._backends._openai_chat import ChatCompletion, Completion
from byte._backends._openai_media import Audio, Image, Moderation, Speech
from byte._backends._openai_support import (
    async_iter,
)
from byte._backends._openai_transport import _get_async_client, _get_client
from byte.adapter.adapter import aadapt, adapt
from byte.adapter.base import BaseCacheLLM
from byte.adapter.prompt_cache_bridge import (
    apply_native_prompt_cache,
    strip_native_prompt_cache_hints,
)
from byte.manager.scalar_data.base import Answer, DataType
from byte.utils.error import wrap_error
from byte.utils.response import (
    get_audio_text_from_openai_answer,
    get_image_from_openai_b64,
    get_image_from_openai_url,
    get_message_from_openai_answer,
    get_stream_message_from_openai_answer,
    get_text_from_openai_answer,
)

__all__ = [
    "Answer",
    "Audio",
    "BaseCacheLLM",
    "ChatCompletion",
    "Completion",
    "_get_async_client",
    "_get_client",
    "DataType",
    "Image",
    "Moderation",
    "Speech",
    "aadapt",
    "adapt",
    "apply_native_prompt_cache",
    "async_iter",
    "get_audio_text_from_openai_answer",
    "get_image_from_openai_b64",
    "get_image_from_openai_url",
    "get_message_from_openai_answer",
    "get_stream_message_from_openai_answer",
    "get_text_from_openai_answer",
    "strip_native_prompt_cache_hints",
    "wrap_error",
]
