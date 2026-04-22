"""Canonical adapter-runtime entrypoints."""

from byte.adapter._runtime_async import aadapt
from byte.adapter._runtime_support import (
    acache_health_check,
    cache_health_check,
    get_budget_tracker,
    get_quality_scorer,
)
from byte.adapter._runtime_sync import adapt

__all__ = ["adapt", "aadapt", "get_budget_tracker", "get_quality_scorer", "cache_health_check", "acache_health_check"]
