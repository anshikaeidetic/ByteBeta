
"""Composed memory feature mixin used by ``byte.core.Cache``."""

from __future__ import annotations

from byte._core_memory_execution import _CacheExecutionMemoryMixin
from byte._core_memory_interactions import _CacheInteractionMemoryMixin
from byte._core_memory_optimization import _CacheOptimizationMemoryMixin
from byte._core_memory_patterns import _CachePatternMemoryMixin
from byte._core_memory_reasoning import _CacheReasoningMemoryMixin
from byte._core_memory_snapshot import _CacheMemorySnapshotMixin
from byte._core_memory_tools import _CacheToolMemoryMixin


class _CacheMemoryMixin(
    _CacheToolMemoryMixin,
    _CacheInteractionMemoryMixin,
    _CacheExecutionMemoryMixin,
    _CachePatternMemoryMixin,
    _CacheOptimizationMemoryMixin,
    _CacheReasoningMemoryMixin,
    _CacheMemorySnapshotMixin,
):
    """Aggregate memory feature groups without changing ``Cache`` inheritance."""


__all__ = ["_CacheMemoryMixin"]
