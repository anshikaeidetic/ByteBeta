
"""Lazy Google GenAI client resolution for the Gemini backend."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from byte.adapter.client_pool import get_async_client as get_pooled_async_client
from byte.adapter.client_pool import get_sync_client as get_pooled_sync_client


def _get_genai_types() -> Any | None:
    try:
        from google.genai import types  # pylint: disable=C0415
    except ImportError:
        return None
    return types

def _build_namespace(**kwargs) -> Any:
    return SimpleNamespace(**kwargs)


def _resolve_backend_callable(name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    backend = sys.modules.get("byte._backends.gemini")
    if backend is None:
        return fallback
    candidate = getattr(backend, name, fallback)
    return candidate if callable(candidate) else fallback


def _create_client(**client_kwargs) -> Any:
    try:
        from google import genai  # pylint: disable=C0415
    except ImportError:
        from byte.utils import import_google_genai  # pylint: disable=C0415

        import_google_genai()
        from google import genai  # pylint: disable=C0415

    return genai.Client(**client_kwargs)


def _create_async_client(**client_kwargs) -> Any:
    try:
        from google import genai  # pylint: disable=C0415
    except ImportError:
        from byte.utils import import_google_genai  # pylint: disable=C0415

        import_google_genai()
        from google import genai  # pylint: disable=C0415

    return genai.Client(**client_kwargs)


def _get_client(api_key=None, **kwargs) -> Any:
    """Create or reuse a Google GenAI client instance."""
    client_kwargs = {}
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        client_kwargs["api_key"] = key

    return get_pooled_sync_client("gemini", _create_client, **client_kwargs)


def _get_async_client(api_key=None, **kwargs) -> Any:
    """Create or reuse an async Google GenAI client instance."""
    client_kwargs = {}
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        client_kwargs["api_key"] = key

    return get_pooled_async_client("gemini", _create_async_client, **client_kwargs)

__all__ = [
    "_build_namespace",
    "_get_async_client",
    "_get_client",
    "_get_genai_types",
    "_resolve_backend_callable",
]
