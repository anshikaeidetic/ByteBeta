"""Compatibility wrapper for the Groq backend adapter."""

from byte._backends import groq as _backend
from byte.adapter._provider_wrapper import bind_backend_module

Answer = _backend.Answer
Audio = _backend.Audio
BaseCacheLLM = _backend.BaseCacheLLM
ChatCompletion = _backend.ChatCompletion
DataType = _backend.DataType
Speech = _backend.Speech
aadapt = _backend.aadapt
adapt = _backend.adapt
apply_native_prompt_cache = _backend.apply_native_prompt_cache
materialize_upload = _backend.materialize_upload
open_upload = _backend.open_upload
strip_native_prompt_cache_hints = _backend.strip_native_prompt_cache_hints

__all__ = [
    "Answer",
    "Audio",
    "BaseCacheLLM",
    "ChatCompletion",
    "DataType",
    "Speech",
    "aadapt",
    "adapt",
    "apply_native_prompt_cache",
    "materialize_upload",
    "open_upload",
    "strip_native_prompt_cache_hints",
]

bind_backend_module(__name__, _backend)
