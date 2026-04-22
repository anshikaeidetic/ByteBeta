from __future__ import annotations

import asyncio
from collections.abc import Callable
from threading import Lock
from typing import Any

_SYNC_CLIENTS: dict[tuple[str, Any], Any] = {}
_ASYNC_CLIENTS: dict[tuple[str, int, Any], Any] = {}
_LOCK = Lock()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze(val))
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def get_sync_client(provider: str, factory: Callable[..., Any], **kwargs) -> Any:
    key = (provider, _freeze(kwargs))
    with _LOCK:
        client = _SYNC_CLIENTS.get(key)
        if client is None:
            client = factory(**kwargs)
            _SYNC_CLIENTS[key] = client
        return client


def get_async_client(provider: str, factory: Callable[..., Any], **kwargs) -> Any:
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = 0
    key = (provider, loop_id, _freeze(kwargs))
    with _LOCK:
        client = _ASYNC_CLIENTS.get(key)
        if client is None:
            client = factory(**kwargs)
            _ASYNC_CLIENTS[key] = client
        return client


def clear_client_pools() -> None:
    with _LOCK:
        _SYNC_CLIENTS.clear()
        _ASYNC_CLIENTS.clear()
