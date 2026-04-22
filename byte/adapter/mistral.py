"""Compatibility wrapper for the Mistral backend adapter."""

from byte._backends import mistral as _backend
from byte.adapter._provider_wrapper import bind_backend_module

Answer = _backend.Answer
BaseCacheLLM = _backend.BaseCacheLLM
ChatCompletion = _backend.ChatCompletion
DataType = _backend.DataType
aadapt = _backend.aadapt
adapt = _backend.adapt
apply_native_prompt_cache = _backend.apply_native_prompt_cache
async_wrap_sync_iterator = _backend.async_wrap_sync_iterator
iter_openai_sse_chunks = _backend.iter_openai_sse_chunks
replay_as_stream = _backend.replay_as_stream
request_json = _backend.request_json
strip_native_prompt_cache_hints = _backend.strip_native_prompt_cache_hints

__all__ = [
    "Answer",
    "BaseCacheLLM",
    "ChatCompletion",
    "DataType",
    "aadapt",
    "adapt",
    "apply_native_prompt_cache",
    "async_wrap_sync_iterator",
    "iter_openai_sse_chunks",
    "replay_as_stream",
    "request_json",
    "strip_native_prompt_cache_hints",
]

bind_backend_module(__name__, _backend)
