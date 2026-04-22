"""Compatibility wrapper for the Gemini backend adapter."""

from byte._backends import gemini as _backend
from byte.adapter._provider_wrapper import bind_backend_module

Answer = _backend.Answer
Audio = _backend.Audio
BaseCacheLLM = _backend.BaseCacheLLM
ChatCompletion = _backend.ChatCompletion
DataType = _backend.DataType
Image = _backend.Image
Speech = _backend.Speech
aadapt = _backend.aadapt
adapt = _backend.adapt
apply_native_prompt_cache = _backend.apply_native_prompt_cache
extract_content_parts = _backend.extract_content_parts
extract_text_content = _backend.extract_text_content
materialize_upload = _backend.materialize_upload
strip_native_prompt_cache_hints = _backend.strip_native_prompt_cache_hints

__all__ = [
    "Answer",
    "Audio",
    "BaseCacheLLM",
    "ChatCompletion",
    "DataType",
    "Image",
    "Speech",
    "aadapt",
    "adapt",
    "apply_native_prompt_cache",
    "extract_content_parts",
    "extract_text_content",
    "materialize_upload",
    "strip_native_prompt_cache_hints",
]

bind_backend_module(__name__, _backend)
