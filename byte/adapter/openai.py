"""Compatibility wrapper for the OpenAI backend adapter."""

from byte._backends import openai as _backend
from byte.adapter._provider_wrapper import bind_backend_module

Answer = _backend.Answer
Audio = _backend.Audio
BaseCacheLLM = _backend.BaseCacheLLM
ChatCompletion = _backend.ChatCompletion
Completion = _backend.Completion
DataType = _backend.DataType
Image = _backend.Image
Moderation = _backend.Moderation
Speech = _backend.Speech
aadapt = _backend.aadapt
adapt = _backend.adapt
apply_native_prompt_cache = _backend.apply_native_prompt_cache
async_iter = _backend.async_iter
get_audio_text_from_openai_answer = _backend.get_audio_text_from_openai_answer
get_image_from_openai_b64 = _backend.get_image_from_openai_b64
get_image_from_openai_url = _backend.get_image_from_openai_url
get_message_from_openai_answer = _backend.get_message_from_openai_answer
get_stream_message_from_openai_answer = _backend.get_stream_message_from_openai_answer
get_text_from_openai_answer = _backend.get_text_from_openai_answer
strip_native_prompt_cache_hints = _backend.strip_native_prompt_cache_hints
wrap_error = _backend.wrap_error

__all__ = [
    "Answer",
    "Audio",
    "BaseCacheLLM",
    "ChatCompletion",
    "Completion",
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

bind_backend_module(__name__, _backend)
