"""Lazy optional-runtime helpers for benchmark entrypoints and runners."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def load_optional_module(module_name: str, *, package: str | None = None) -> Any:
    """Import an optional dependency only when a live benchmark path needs it."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        install_target = package or module_name
        raise RuntimeError(
            f"Optional dependency '{install_target}' is required for this live benchmark path. "
            f"Install it with `pip install {install_target}` or the matching Byte extra."
        ) from exc


def create_openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """Create an OpenAI-compatible client without importing the SDK at module import time."""
    openai_sdk = load_optional_module("openai", package="openai")
    client_kwargs = dict(kwargs)
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    return openai_sdk.OpenAI(**client_kwargs)


def load_chat_backend(provider: str) -> Any:
    """Resolve a benchmark chat backend lazily from the provider name."""
    module_map = {
        "openai": "byte._backends.openai",
        "anthropic": "byte._backends.anthropic",
        "deepseek": "byte._backends.deepseek",
    }
    normalized = str(provider or "").strip().lower()
    try:
        module_name = module_map[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider backend: {provider}") from exc
    return import_module(module_name).ChatCompletion


__all__ = ["create_openai_client", "load_chat_backend", "load_optional_module"]
