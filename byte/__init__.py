"""Byte package exports."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

__version__ = "1.0.6"
PRODUCT_NAME = "Byte"
PRODUCT_SHORT_NAME = "Byte"
PACKAGE_IMPORT_NAME = "byte"
DISTRIBUTION_NAME = "byteai-cache"
COMPATIBILITY_REPO_NAME = "ByteNew"
COMPATIBILITY_CRD_KIND = "ByteCache"
LEGACY_GATEWAY_CHAT_PATH = "/byte/gateway/chat"
METRIC_NAMESPACE = "byteai"
INTERNAL_AUTH_HEADER = "X-Byte-Internal-Token"


def _load_config() -> Any:
    from byte.config import Config

    return Config


def _load_cache() -> Any:
    from byte.core import Cache

    return Cache


def _load_cache_instance() -> Any:
    from byte.core import cache

    return cache


def _load_default_cache() -> Any:
    from byte.core import get_default_cache

    return get_default_cache


def _load_client() -> Any:
    from byte.client import ByteClient

    return ByteClient


_LAZY_EXPORTS: dict[str, Callable[[], Any]] = {
    "ByteClient": _load_client,
    "Cache": _load_cache,
    "Config": _load_config,
    "cache": _load_cache_instance,
    "get_default_cache": _load_default_cache,
}
_LAZY_SUBMODULES = frozenset(
    {
        "client",
        "config",
        "core",
        "mcp_gateway",
        "telemetry",
    }
)


def __getattr__(name: str) -> Any:
    loader = _LAZY_EXPORTS.get(name)
    if loader is not None:
        value = loader()
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "COMPATIBILITY_CRD_KIND",
    "COMPATIBILITY_REPO_NAME",
    "DISTRIBUTION_NAME",
    "INTERNAL_AUTH_HEADER",
    "LEGACY_GATEWAY_CHAT_PATH",
    "METRIC_NAMESPACE",
    "PACKAGE_IMPORT_NAME",
    "PRODUCT_NAME",
    "PRODUCT_SHORT_NAME",
    "ByteClient",
    "Cache",
    "Config",
    "__version__",
    "cache",
    "get_default_cache",
]
