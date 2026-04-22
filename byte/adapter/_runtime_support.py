"""Thin support-layer facade for the split adapter runtime."""

from byte.adapter.pipeline.context import (
    _apply_request_namespaces,
    _compile_context_if_needed,
)
from byte.adapter.pipeline.utils import (
    acache_health_check,
    cache_health_check,
    get_budget_tracker,
    get_quality_scorer,
)

__all__ = [
    "_apply_request_namespaces",
    "_compile_context_if_needed",
    "acache_health_check",
    "cache_health_check",
    "get_budget_tracker",
    "get_quality_scorer",
]
