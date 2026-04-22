"""Provider-dispatch attempt helpers for routed adapter execution."""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from typing import Any

from byte.utils.error import (
    ByteErrorCode,
    CacheError,
    ProviderRequestError,
    classify_error,
    set_error_metadata,
)
from byte.utils.log import byte_log, log_byte_error

from ._router_registry import _REGISTRY, RouteTarget
from ._router_resolution import _PROVIDER_MODULES, _provider_request_kwargs
from ._router_selection import _is_retryable_error


def _attempt_target_sync(
    target: RouteTarget,
    *,
    surface: str,
    retries: int,
    backoff_ms: float,
    cooldown_seconds: float,
    config: Any,
    kwargs: dict[str, Any],
) -> Any:
    last_error: CacheError | ProviderRequestError | None = None
    target_callable = _surface_callable(target.provider, surface, async_mode=False)
    target_kwargs = _provider_request_kwargs(target.provider, kwargs, config)
    for attempt in range(max(0, retries) + 1):
        started = time.time()
        try:
            response = target_callable(model=target.model, **target_kwargs)
            _REGISTRY.record_result(
                target,
                success=True,
                latency_ms=(time.time() - started) * 1000,
            )
            return response
        except Exception as exc:  # noqa: BLE001, RUF100 - provider boundary
            normalized = _normalize_router_error(exc, target=target)
            last_error = normalized
            _record_router_failure(
                target,
                error=normalized,
                started=started,
                cooldown_seconds=cooldown_seconds,
            )
            if attempt >= retries or not _is_retryable_error(normalized):
                break
            if backoff_ms > 0:
                time.sleep(backoff_ms / 1000.0)
    if last_error is not None:
        raise set_error_metadata(last_error, code=ByteErrorCode.ROUTER_RETRY_EXHAUSTED)
    raise CacheError(
        "Byte router could not resolve a valid provider target.",
        code=ByteErrorCode.ROUTER_NO_TARGET,
    )


async def _attempt_target_async(
    target: RouteTarget,
    *,
    surface: str,
    retries: int,
    backoff_ms: float,
    cooldown_seconds: float,
    config: Any,
    kwargs: dict[str, Any],
) -> Any:
    last_error: CacheError | ProviderRequestError | None = None
    target_callable = _surface_callable(target.provider, surface, async_mode=True)
    target_kwargs = _provider_request_kwargs(target.provider, kwargs, config)
    for attempt in range(max(0, retries) + 1):
        started = time.time()
        try:
            response = await target_callable(model=target.model, **target_kwargs)
            _REGISTRY.record_result(
                target,
                success=True,
                latency_ms=(time.time() - started) * 1000,
            )
            return response
        except Exception as exc:  # noqa: BLE001, RUF100 - provider boundary
            normalized = _normalize_router_error(exc, target=target)
            last_error = normalized
            _record_router_failure(
                target,
                error=normalized,
                started=started,
                cooldown_seconds=cooldown_seconds,
            )
            if attempt >= retries or not _is_retryable_error(normalized):
                break
            if backoff_ms > 0:
                await asyncio.sleep(backoff_ms / 1000.0)
    if last_error is not None:
        raise set_error_metadata(last_error, code=ByteErrorCode.ROUTER_RETRY_EXHAUSTED)
    raise CacheError(
        "Byte router could not resolve a valid provider target.",
        code=ByteErrorCode.ROUTER_NO_TARGET,
    )


def _normalize_router_error(exc: Exception, *, target: RouteTarget) -> CacheError | ProviderRequestError:
    retryable = _is_retryable_error(exc)
    if isinstance(exc, ProviderRequestError):
        return set_error_metadata(
            exc,
            retryable=retryable if exc.retryable is None else exc.retryable,
            provider=target.provider,
        )
    if isinstance(exc, CacheError):
        return set_error_metadata(
            exc,
            code=getattr(exc, "code", None) or ByteErrorCode.ROUTER_DISPATCH,
            retryable=retryable,
            provider=target.provider,
        )
    wrapped = CacheError(
        str(exc),
        code=ByteErrorCode.ROUTER_DISPATCH,
        retryable=retryable,
        provider=target.provider,
    )
    wrapped.__cause__ = exc
    return wrapped


def _record_router_failure(
    target: RouteTarget,
    *,
    error: CacheError | ProviderRequestError,
    started: float,
    cooldown_seconds: float,
) -> None:
    retryable = _is_retryable_error(error)
    _REGISTRY.record_result(
        target,
        success=False,
        latency_ms=(time.time() - started) * 1000,
        error=str(error),
        cooldown_seconds=cooldown_seconds if retryable else 0.0,
    )
    info = classify_error(error, boundary="router.dispatch", provider=target.provider)
    log_byte_error(
        byte_log,
        logging.WARNING,
        "router target attempt failed",
        error=error,
        code=info.code,
        boundary="router.dispatch",
        provider=target.provider,
        stage=target.qualified_model,
    )


def _surface_callable(provider: str, surface: str, *, async_mode: bool) -> Any:
    module_name = _PROVIDER_MODULES.get(provider)
    if not module_name:
        raise CacheError(
            f"Unsupported provider target: {provider}",
            code=ByteErrorCode.ROUTER_NO_TARGET,
        )
    module = importlib.import_module(module_name)
    if surface == "chat_completion":
        return getattr(module.ChatCompletion, "acreate" if async_mode else "create")
    if surface == "text_completion":
        return getattr(module.Completion, "acreate" if async_mode else "create")
    if surface == "image":
        return module.Image.create
    if surface == "audio_transcribe":
        return module.Audio.transcribe
    if surface == "audio_translate":
        return module.Audio.translate
    if surface == "speech":
        return module.Speech.create
    if surface == "moderation":
        return module.Moderation.create
    raise CacheError(
        f"Unsupported router surface: {surface}",
        code=ByteErrorCode.ROUTER_NO_TARGET,
    )


__all__ = [
    "_attempt_target_async",
    "_attempt_target_sync",
    "_surface_callable",
]
