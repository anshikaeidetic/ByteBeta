"""Target ordering and retry policy helpers for routed execution."""

import random
import time
from collections.abc import Sequence
from typing import Any

from byte.utils.error import ByteErrorCode, CacheError

from ._router_registry import _REGISTRY, RouteTarget
from ._router_resolution import _enrich_target, _supports_surface


def _flatten_targets(targets: Sequence[Any]) -> list[RouteTarget]:
    flattened = []
    for item in targets:
        if isinstance(item, list):
            flattened.extend(_flatten_targets(item))
        elif isinstance(item, RouteTarget):
            flattened.append(item)
    deduped = []
    seen = set()
    for target in flattened:
        key = (target.provider, target.model)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target)
    return deduped


def _order_targets(
    targets: Sequence[RouteTarget],
    *,
    strategy: str,
    surface: str,
    alias_key: str,
    request_kwargs: dict[str, Any] | None,
) -> list[RouteTarget]:
    available = [target for target in targets if _supports_surface(target.provider, surface)]
    if not available:
        raise CacheError(
            f"No provider target supports the `{surface}` surface.",
            code=ByteErrorCode.ROUTER_NO_TARGET,
        )

    now = time.time()
    cooled = [target for target in available if _REGISTRY.current_cooldown(target) <= now]
    candidates = cooled or list(available)

    if strategy in {"simple_shuffle", "shuffle"}:
        shuffled = list(candidates)
        random.shuffle(shuffled)
        return [_enrich_target(target, request_kwargs=request_kwargs) for target in shuffled]

    if strategy in {"round_robin", "least_busy"}:
        ordered = _REGISTRY.choose_round_robin(f"{surface}:{alias_key}", list(candidates))
        return [_enrich_target(target, request_kwargs=request_kwargs) for target in ordered]

    enriched = [_enrich_target(target, request_kwargs=request_kwargs) for target in candidates]

    if strategy in {"latency", "latency_based", "lowest_latency"}:
        return sorted(
            enriched,
            key=lambda item: (
                item.metadata.get("avg_latency_ms", 0.0),
                -item.metadata.get("health_score", 1.0),
            ),
        )

    if strategy in {"cost", "cost_based", "lowest_cost"}:
        has_known_cost = any(
            item.metadata.get("estimated_cost_usd") is not None for item in enriched
        )
        return sorted(
            enriched,
            key=lambda item: (
                item.metadata.get("estimated_cost_usd")
                if item.metadata.get("estimated_cost_usd") is not None
                else float("inf")
                if has_known_cost
                else 0.0,
                -item.metadata.get("health_score", 1.0),
                item.metadata.get("avg_latency_ms", 0.0),
            ),
        )

    if strategy == "health_weighted":
        return sorted(
            enriched,
            key=lambda item: (
                -item.metadata.get("health_score", 1.0),
                item.metadata.get("estimated_cost_usd")
                if item.metadata.get("estimated_cost_usd") is not None
                else float("inf"),
                item.metadata.get("avg_latency_ms", 0.0),
            ),
        )

    return enriched

def _annotate_response(
    response: Any,
    *,
    target: RouteTarget,
    source_model: str,
    attempted: Sequence[str],
    fallback_used: bool,
    strategy: str,
) -> Any:
    if not isinstance(response, dict):
        return response
    payload = dict(response)
    payload.setdefault("byte_provider", target.provider)
    payload["byte_router"] = {
        "source_model": source_model,
        "selected_model": target.model,
        "selected_provider": target.provider,
        "selected_target": target.qualified_model,
        "attempted_targets": list(attempted),
        "fallback_used": bool(fallback_used),
        "alias": target.alias,
        "strategy_source": target.source,
        "strategy": strategy,
        "estimated_cost_usd": target.metadata.get("estimated_cost_usd"),
        "health_score": target.metadata.get("health_score"),
        "avg_latency_ms": target.metadata.get("avg_latency_ms"),
    }
    return payload

def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return any(
        token in text
        for token in (
            "timeout",
            "timed out",
            "temporarily",
            "rate limit",
            "429",
            "resource_exhausted",
            "overloaded",
            "service unavailable",
            "connection",
        )
    )
