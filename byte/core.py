"""Public cache singleton and lazy proxy surface."""

from __future__ import annotations

import atexit
from typing import Any, cast

import byte._core_lifecycle as _core_lifecycle
from byte._core_lifecycle import _CacheLifecycleMixin, _close_live_caches
from byte._core_memory import _CacheMemoryMixin


class Cache(_CacheLifecycleMixin, _CacheMemoryMixin):
    """ByteAI Cache core object."""


def get_default_cache() -> Cache:
    """Return the process-wide default cache, creating it lazily when needed."""

    with _core_lifecycle._DEFAULT_CACHE_LOCK:
        default_cache = cast(Cache | None, _core_lifecycle._DEFAULT_CACHE)
        if default_cache is None or getattr(default_cache, "_closed", False):
            default_cache = Cache()
            _core_lifecycle._DEFAULT_CACHE = default_cache
        return default_cache


class LazyCacheProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_default_cache(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(get_default_cache(), name, value)

    def __repr__(self) -> str:
        return repr(get_default_cache())

    def __byte_cache_owner__(self) -> Cache:
        return get_default_cache()


cache = LazyCacheProxy()

atexit.register(_close_live_caches)

__all__ = ["Cache", "cache", "get_default_cache"]
