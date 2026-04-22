
"""In-process model-route performance tracking."""

from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from typing import Any

from byte.processor._model_router_types import ModelRouteDecision


class RoutePerformanceTracker:
    """Small in-process learner for route quality and latency."""

    def __init__(self, window_size: int = 64) -> None:
        self._window_size = max(4, int(window_size or 64))
        self._lock = Lock()
        self._buckets: dict[str, dict[str, deque[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self._window_size))
        )

    def record(self, route_key: str, tier: str, *, accepted: bool, latency_ms: float) -> None:
        if not route_key or not tier:
            return
        with self._lock:
            bucket = self._buckets[route_key][tier]
            bucket.append(
                {
                    "accepted": bool(accepted),
                    "latency_ms": round(float(latency_ms or 0.0), 4),
                }
            )

    def summary(self, route_key: str, tier: str) -> dict[str, Any]:
        with self._lock:
            bucket = list(self._buckets.get(route_key, {}).get(tier, ()))
        samples = len(bucket)
        if samples == 0:
            return {
                "samples": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
            }
        success_rate = sum(1 for item in bucket if item["accepted"]) / samples
        avg_latency = sum(item["latency_ms"] for item in bucket) / samples
        return {
            "samples": samples,
            "success_rate": round(success_rate, 4),
            "avg_latency_ms": round(avg_latency, 2),
        }

    def prefer_expensive(self, route_key: str, config: Any) -> bool:
        if not getattr(config, "routing_adaptive", False):
            return False
        stats = self.summary(route_key, "cheap")
        min_samples = max(1, int(getattr(config, "routing_adaptive_min_samples", 6) or 6))
        quality_floor = float(getattr(config, "routing_adaptive_quality_floor", 0.75) or 0.75)
        return stats["samples"] >= min_samples and stats["success_rate"] < quality_floor

    def stats(self) -> dict[str, Any]:
        with self._lock:
            payload = {}
            for route_key, route_buckets in self._buckets.items():
                route_payload = {}
                for tier, bucket in route_buckets.items():
                    samples = len(bucket)
                    if samples == 0:
                        route_payload[tier] = {
                            "samples": 0,
                            "success_rate": 0.0,
                            "avg_latency_ms": 0.0,
                        }
                        continue
                    success_rate = sum(1 for item in bucket if item["accepted"]) / samples
                    avg_latency = sum(item["latency_ms"] for item in bucket) / samples
                    route_payload[tier] = {
                        "samples": samples,
                        "success_rate": round(success_rate, 4),
                        "avg_latency_ms": round(avg_latency, 2),
                    }
                payload[route_key] = route_payload
        return payload

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


_route_tracker = RoutePerformanceTracker()


def record_route_outcome(
    decision: ModelRouteDecision | None,
    *,
    accepted: bool,
    latency_ms: float,
) -> None:
    if decision is None:
        return
    _route_tracker.record(
        decision.route_key,
        decision.tier,
        accepted=accepted,
        latency_ms=latency_ms,
    )


def route_performance_stats() -> dict[str, Any]:
    return _route_tracker.stats()


def clear_route_performance() -> None:
    _route_tracker.clear()

def _meets_budget_quality_floor(route_key: str, config: Any) -> bool:
    quality_floor = float(getattr(config, "budget_quality_floor", 0.75) or 0.75)
    stats = _route_tracker.summary(route_key, "cheap")
    samples = int(stats.get("samples", 0) or 0)
    if samples == 0:
        return True
    return float(stats.get("success_rate", 0.0) or 0.0) >= quality_floor


def _cheap_latency_beats_expensive(route_key: str) -> bool:
    cheap = _route_tracker.summary(route_key, "cheap")
    expensive = _route_tracker.summary(route_key, "expensive")
    cheap_latency = float(cheap.get("avg_latency_ms", 0.0) or 0.0)
    expensive_latency = float(expensive.get("avg_latency_ms", 0.0) or 0.0)
    if cheap_latency <= 0:
        return True
    if expensive_latency <= 0:
        return True
    return cheap_latency <= expensive_latency

__all__ = [
    "RoutePerformanceTracker",
    "_cheap_latency_beats_expensive",
    "_meets_budget_quality_floor",
    "_route_tracker",
    "clear_route_performance",
    "record_route_outcome",
    "route_performance_stats",
]
