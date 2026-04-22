
"""Compatibility facade for H2O runtime implementation internals."""

from __future__ import annotations

from byte.h2o._runtime_factory import get_huggingface_runtime
from byte.h2o._runtime_runtime import H2ORuntime

__all__ = ["H2ORuntime", "get_huggingface_runtime"]
