"""Compatibility wrapper for the llama.cpp backend adapter."""

from byte._backends import llama_cpp as _backend
from byte.adapter._provider_wrapper import bind_backend_module

ChatCompletion = _backend.ChatCompletion
Completion = _backend.Completion
DataType = _backend.DataType

__all__ = [
    "ChatCompletion",
    "Completion",
    "DataType",
]

bind_backend_module(__name__, _backend)
