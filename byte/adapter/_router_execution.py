"""Sync and async routed execution facade for provider backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from byte import cache

from ._router_attempts import _attempt_target_async, _attempt_target_sync, _surface_callable
from ._router_dispatch import (
    _run_ordered_async,
    _run_ordered_sync,
    _run_speculative_async,
    _run_speculative_sync,
)
from ._router_registry import RouteTarget
from ._router_resolution import resolve_provider_model
from ._router_selection import _flatten_targets, _order_targets


def route_completion(
    *,
    surface: str,
    model: str,
    async_mode: bool = False,
    **kwargs: Any,
) -> Any:
    """Route a completion request to the configured provider execution strategy."""

    return _route_surface(surface=surface, model=model, async_mode=async_mode, **kwargs)


def _route_surface(
    *,
    surface: str,
    model: str,
    async_mode: bool,
    **kwargs: Any,
) -> Any:
    cache_obj = kwargs.get("cache_obj", cache)
    config = getattr(cache_obj, "config", None)
    provider_hint = str(kwargs.pop("byte_provider_hint", "") or "")
    aliases = dict(getattr(config, "routing_model_aliases", {}) or {})
    aliases.update(kwargs.pop("byte_model_aliases", {}) or {})
    strategy = str(
        kwargs.pop("byte_routing_strategy", "")
        or getattr(config, "routing_strategy", "priority")
        or "priority"
    ).lower()
    retries = int(
        kwargs.pop("byte_retry_attempts", getattr(config, "routing_retry_attempts", 0) or 0)
    )
    backoff_ms = float(
        kwargs.pop(
            "byte_retry_backoff_ms",
            getattr(config, "routing_retry_backoff_ms", 0.0) or 0.0,
        )
    )
    cooldown_seconds = float(
        kwargs.pop(
            "byte_cooldown_seconds",
            getattr(config, "routing_cooldown_seconds", 15.0) or 15.0,
        )
    )
    fallback_models = list(kwargs.pop("byte_fallback_models", []) or [])
    model_fallbacks = dict(getattr(config, "routing_fallbacks", {}) or {})

    targets = resolve_provider_model(
        model,
        provider_hint=provider_hint,
        aliases=aliases,
        allow_backend_target=False,
    )
    if fallback_models:
        for fallback_model in fallback_models:
            targets.extend(
                resolve_provider_model(
                fallback_model,
                provider_hint=provider_hint,
                aliases=aliases,
                allow_backend_target=True,
            )
            )
    elif model_fallbacks:
        fallback_list = list(model_fallbacks.get(str(model or ""), []) or [])
        if not fallback_list and targets:
            fallback_list = list(model_fallbacks.get(targets[0].qualified_model, []) or [])
        for fallback_model in fallback_list:
            targets.extend(
                resolve_provider_model(
                    fallback_model,
                    provider_hint=provider_hint,
                    aliases=aliases,
                    allow_backend_target=True,
                )
            )

    flattened_targets = _flatten_targets(targets)
    ordered_targets = _order_targets(
        flattened_targets,
        strategy=strategy,
        surface=surface,
        alias_key=str(model or ""),
        request_kwargs=kwargs,
    )

    if async_mode:
        return _aroute_targets(
            ordered_targets,
            surface=surface,
            source_model=str(model or ""),
            strategy=strategy,
            retries=retries,
            backoff_ms=backoff_ms,
            cooldown_seconds=cooldown_seconds,
            config=config,
            **kwargs,
        )
    return _route_targets(
        ordered_targets,
        surface=surface,
        source_model=str(model or ""),
        strategy=strategy,
        retries=retries,
        backoff_ms=backoff_ms,
        cooldown_seconds=cooldown_seconds,
        config=config,
        **kwargs,
    )


def _route_targets(
    targets: Sequence[RouteTarget],
    *,
    surface: str,
    source_model: str,
    strategy: str,
    retries: int,
    backoff_ms: float,
    cooldown_seconds: float,
    config: Any,
    **kwargs: Any,
) -> Any:
    speculative_enabled = bool(getattr(config, "speculative_routing", False))
    max_parallel = max(1, int(getattr(config, "speculative_max_parallel", 2) or 2))
    if speculative_enabled and len(targets) > 1:
        speculative_result = _run_speculative_sync(
            list(targets[: min(len(targets), max_parallel)]),
            surface=surface,
            source_model=source_model,
            strategy=strategy,
            retries=retries,
            backoff_ms=backoff_ms,
            cooldown_seconds=cooldown_seconds,
            config=config,
            kwargs=kwargs,
        )
        if speculative_result is not None:
            return speculative_result
        targets = list(targets[min(len(targets), max_parallel) :])

    return _run_ordered_sync(
        targets,
        surface=surface,
        source_model=source_model,
        strategy=strategy,
        retries=retries,
        backoff_ms=backoff_ms,
        cooldown_seconds=cooldown_seconds,
        config=config,
        kwargs=kwargs,
    )


async def _aroute_targets(
    targets: Sequence[RouteTarget],
    *,
    surface: str,
    source_model: str,
    strategy: str,
    retries: int,
    backoff_ms: float,
    cooldown_seconds: float,
    config: Any,
    **kwargs: Any,
) -> Any:
    speculative_enabled = bool(getattr(config, "speculative_routing", False))
    max_parallel = max(1, int(getattr(config, "speculative_max_parallel", 2) or 2))
    if speculative_enabled and len(targets) > 1:
        speculative_result = await _run_speculative_async(
            list(targets[: min(len(targets), max_parallel)]),
            surface=surface,
            source_model=source_model,
            strategy=strategy,
            retries=retries,
            backoff_ms=backoff_ms,
            cooldown_seconds=cooldown_seconds,
            config=config,
            kwargs=kwargs,
        )
        if speculative_result is not None:
            return speculative_result
        targets = list(targets[min(len(targets), max_parallel) :])

    return await _run_ordered_async(
        targets,
        surface=surface,
        source_model=source_model,
        strategy=strategy,
        retries=retries,
        backoff_ms=backoff_ms,
        cooldown_seconds=cooldown_seconds,
        config=config,
        kwargs=kwargs,
    )


__all__ = [
    "_aroute_targets",
    "_attempt_target_async",
    "_attempt_target_sync",
    "_route_surface",
    "_route_targets",
    "_surface_callable",
    "route_completion",
]
