"""Ordered and speculative dispatch helpers for routed adapter execution."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from byte.utils.error import ByteErrorCode, CacheError, ProviderRequestError

from ._router_attempts import _attempt_target_async, _attempt_target_sync
from ._router_registry import RouteTarget
from ._router_selection import _annotate_response


def _run_speculative_sync(
    targets: Sequence[RouteTarget],
    *,
    surface: str,
    source_model: str,
    strategy: str,
    retries: int,
    backoff_ms: float,
    cooldown_seconds: float,
    config: Any,
    kwargs: dict[str, Any],
) -> Any:
    attempted = [target.qualified_model for target in targets]
    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        future_map = {
            executor.submit(
                _attempt_target_sync,
                target,
                surface=surface,
                retries=retries,
                backoff_ms=backoff_ms,
                cooldown_seconds=cooldown_seconds,
                config=config,
                kwargs=kwargs,
            ): target
            for target in targets
        }
        speculative_error: CacheError | ProviderRequestError | None = None
        for future in as_completed(future_map):
            target = future_map[future]
            try:
                response = future.result()
                return _annotate_response(
                    response,
                    target=target,
                    source_model=source_model,
                    attempted=attempted,
                    fallback_used=len(attempted) > 1,
                    strategy=f"{strategy}:speculative",
                )
            except (CacheError, ProviderRequestError) as exc:
                speculative_error = exc
        if speculative_error is not None and len(targets) == len(attempted):
            raise speculative_error
    return None


async def _run_speculative_async(
    targets: Sequence[RouteTarget],
    *,
    surface: str,
    source_model: str,
    strategy: str,
    retries: int,
    backoff_ms: float,
    cooldown_seconds: float,
    config: Any,
    kwargs: dict[str, Any],
) -> Any:
    attempted = [target.qualified_model for target in targets]
    task_map: dict[asyncio.Future[Any], RouteTarget] = {
        asyncio.create_task(
            _attempt_target_async(
                target,
                surface=surface,
                retries=retries,
                backoff_ms=backoff_ms,
                cooldown_seconds=cooldown_seconds,
                config=config,
                kwargs=kwargs,
            )
        ): target
        for target in targets
    }
    speculative_error: CacheError | ProviderRequestError | None = None
    for task in asyncio.as_completed(list(task_map.keys())):
        target = task_map[task]
        try:
            response = await task
            for pending in task_map:
                if pending is not task and not pending.done():
                    pending.cancel()
            return _annotate_response(
                response,
                target=target,
                source_model=source_model,
                attempted=attempted,
                fallback_used=len(attempted) > 1,
                strategy=f"{strategy}:speculative",
            )
        except (CacheError, ProviderRequestError) as exc:
            speculative_error = exc
    if speculative_error is not None and len(targets) == len(attempted):
        raise speculative_error
    return None


def _run_ordered_sync(
    targets: Sequence[RouteTarget],
    *,
    surface: str,
    source_model: str,
    strategy: str,
    retries: int,
    backoff_ms: float,
    cooldown_seconds: float,
    config: Any,
    kwargs: dict[str, Any],
) -> Any:
    last_error: CacheError | ProviderRequestError | None = None
    attempted: list[str] = []
    for target in targets:
        attempted.append(target.qualified_model)
        try:
            response = _attempt_target_sync(
                target,
                surface=surface,
                retries=retries,
                backoff_ms=backoff_ms,
                cooldown_seconds=cooldown_seconds,
                config=config,
                kwargs=kwargs,
            )
            return _annotate_response(
                response,
                target=target,
                source_model=source_model,
                attempted=attempted,
                fallback_used=len(attempted) > 1,
                strategy=strategy,
            )
        except (CacheError, ProviderRequestError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise CacheError(
        "Byte router could not resolve a valid provider target.",
        code=ByteErrorCode.ROUTER_NO_TARGET,
    )


async def _run_ordered_async(
    targets: Sequence[RouteTarget],
    *,
    surface: str,
    source_model: str,
    strategy: str,
    retries: int,
    backoff_ms: float,
    cooldown_seconds: float,
    config: Any,
    kwargs: dict[str, Any],
) -> Any:
    last_error: CacheError | ProviderRequestError | None = None
    attempted: list[str] = []
    for target in targets:
        attempted.append(target.qualified_model)
        try:
            response = await _attempt_target_async(
                target,
                surface=surface,
                retries=retries,
                backoff_ms=backoff_ms,
                cooldown_seconds=cooldown_seconds,
                config=config,
                kwargs=kwargs,
            )
            return _annotate_response(
                response,
                target=target,
                source_model=source_model,
                attempted=attempted,
                fallback_used=len(attempted) > 1,
                strategy=strategy,
            )
        except (CacheError, ProviderRequestError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise CacheError(
        "Byte router could not resolve a valid provider target.",
        code=ByteErrorCode.ROUTER_NO_TARGET,
    )


__all__ = [
    "_run_ordered_async",
    "_run_ordered_sync",
    "_run_speculative_async",
    "_run_speculative_sync",
]
