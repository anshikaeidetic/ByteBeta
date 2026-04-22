"""Thin response-layer facade for the split adapter runtime."""

from byte.adapter.pipeline.stream import (
    _finalize_stream_memory,
    _stream_chunk_text,
    _wrap_async_stream_with_memory,
    _wrap_sync_stream_with_memory,
)
from byte.adapter.pipeline.utils import (
    _build_synthetic_response,
    _extract_llm_answer,
    _set_llm_answer,
    _try_record_budget,
)

__all__ = [
    "_build_synthetic_response",
    "_extract_llm_answer",
    "_finalize_stream_memory",
    "_set_llm_answer",
    "_stream_chunk_text",
    "_try_record_budget",
    "_wrap_async_stream_with_memory",
    "_wrap_sync_stream_with_memory",
]
