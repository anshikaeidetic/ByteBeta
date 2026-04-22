"""Compatibility wrapper for the Anthropic backend adapter."""

from byte._backends import anthropic as _backend
from byte.adapter._provider_wrapper import bind_backend_module

Answer = _backend.Answer
BaseCacheLLM = _backend.BaseCacheLLM
ChatCompletion = _backend.ChatCompletion
DataType = _backend.DataType
aadapt = _backend.aadapt
adapt = _backend.adapt
apply_native_prompt_cache = _backend.apply_native_prompt_cache
extract_content_parts = _backend.extract_content_parts
extract_text_content = _backend.extract_text_content
strip_native_prompt_cache_hints = _backend.strip_native_prompt_cache_hints

__all__ = [
    "Answer",
    "BaseCacheLLM",
    "ChatCompletion",
    "DataType",
    "aadapt",
    "adapt",
    "apply_native_prompt_cache",
    "extract_content_parts",
    "extract_text_content",
    "strip_native_prompt_cache_hints",
]

bind_backend_module(__name__, _backend)
