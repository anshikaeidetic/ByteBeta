"""Compatibility wrapper for the Hugging Face backend adapter."""

from byte._backends import huggingface as _backend
from byte.adapter._provider_wrapper import bind_backend_module

Answer = _backend.Answer
BaseCacheLLM = _backend.BaseCacheLLM
ChatCompletion = _backend.ChatCompletion
Completion = _backend.Completion
DataType = _backend.DataType
aadapt = _backend.aadapt
adapt = _backend.adapt
describe_huggingface_runtime = _backend.describe_huggingface_runtime
get_huggingface_runtime = _backend.get_huggingface_runtime

__all__ = [
    "Answer",
    "BaseCacheLLM",
    "ChatCompletion",
    "Completion",
    "DataType",
    "aadapt",
    "adapt",
    "describe_huggingface_runtime",
    "get_huggingface_runtime",
]

bind_backend_module(__name__, _backend)
