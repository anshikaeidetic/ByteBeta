"""Client creation and pooling helpers for the Byte OpenAI backend."""

import os
from typing import Any

from byte.adapter.client_pool import get_async_client as get_pooled_async_client
from byte.adapter.client_pool import get_sync_client as get_pooled_sync_client


def _create_client(**client_kwargs) -> Any:
    import openai as _openai  # pylint: disable=C0415

    return _openai.OpenAI(**client_kwargs)


def _create_async_client(**client_kwargs) -> Any:
    import openai as _openai  # pylint: disable=C0415

    return _openai.AsyncOpenAI(**client_kwargs)


def _get_client(api_key=None, **kwargs) -> Any:
    """Create or reuse an OpenAI client instance."""
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    elif os.getenv("OPENAI_API_KEY"):
        client_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    base_url = kwargs.pop("api_base", None) or os.getenv("OPENAI_API_BASE")
    if base_url:
        client_kwargs["base_url"] = base_url

    return get_pooled_sync_client("openai", _create_client, **client_kwargs)


def _get_async_client(api_key=None, **kwargs) -> Any:
    """Create or reuse an async OpenAI client instance."""
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    elif os.getenv("OPENAI_API_KEY"):
        client_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    base_url = kwargs.pop("api_base", None) or os.getenv("OPENAI_API_BASE")
    if base_url:
        client_kwargs["base_url"] = base_url

    return get_pooled_async_client("openai", _create_async_client, **client_kwargs)
