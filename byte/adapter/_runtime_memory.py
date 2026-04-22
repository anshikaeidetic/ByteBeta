"""Thin memory-layer facade for the split adapter runtime."""

from byte.adapter.pipeline.memory import (
    _aembed_request,
    _embed_request,
    _make_coalesce_key,
    _record_ai_memory,
    _record_execution_memory,
    _record_failure_memory,
    _record_reasoning_memory,
    _record_workflow_outcome,
    _resolve_similarity_threshold,
)

__all__ = [
    "_aembed_request",
    "_embed_request",
    "_make_coalesce_key",
    "_record_ai_memory",
    "_record_execution_memory",
    "_record_failure_memory",
    "_record_reasoning_memory",
    "_record_workflow_outcome",
    "_resolve_similarity_threshold",
]
