"""Async helpers for sync-first internals."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from typing import Any


async def run_sync(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Execute a synchronous callable in a worker thread.

    Note: This is a bridging step using `asyncio.to_thread` which avoids blocking
    the event loop but still consumes thread-pool slots. For high-concurrency
    deployments, the real win would be true native async implementations in
    vector/scalar backends (aioredis, asyncpg, motor).
    """
    return await asyncio.to_thread(functools.partial(func, *args, **kwargs))
