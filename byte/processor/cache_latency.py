from collections import defaultdict, deque
from threading import Lock
from typing import Any


class CacheLatencyTracker:
    def __init__(self, max_samples: int = 64) -> None:
        self._max_samples = max(8, int(max_samples or 64))
        self._lock = Lock()
        self._buckets: dict[str, dict[str, deque[dict[str, float]]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self._max_samples))
        )

    def record(self, route_key: str, stage: str, *, latency_ms: float, hit: bool) -> None:
        if not route_key or not stage:
            return
        with self._lock:
            self._buckets[route_key][stage].append(
                {
                    "latency_ms": round(float(latency_ms or 0.0), 4),
                    "hit": 1.0 if hit else 0.0,
                }
            )

    def summary(self, route_key: str, stage: str) -> dict[str, float]:
        with self._lock:
            bucket = list(self._buckets.get(route_key, {}).get(stage, ()))
        samples = len(bucket)
        if samples <= 0:
            return {
                "samples": 0,
                "hit_count": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "hit_rate": 0.0,
            }
        latencies: list[float] = sorted(
            float(item.get("latency_ms", 0.0) or 0.0) for item in bucket
        )
        avg_latency = sum(latencies) / samples
        hit_count = sum(1 for item in bucket if float(item.get("hit", 0.0) or 0.0) > 0.0)
        hit_rate = hit_count / samples
        p95_index = min(samples - 1, max(0, int(round((samples - 1) * 0.95))))
        return {
            "samples": samples,
            "hit_count": hit_count,
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(latencies[p95_index], 2),
            "hit_rate": round(hit_rate, 4),
        }

    def should_bypass(self, route_key: str, stage: str, config: Any) -> bool:
        if not bool(getattr(config, "cache_latency_guard", True)):
            return False
        stats = self.summary(route_key, stage)
        min_samples = max(1, int(getattr(config, "cache_latency_min_samples", 8) or 8))
        probe_samples = max(
            min_samples, int(getattr(config, "cache_latency_probe_samples", 32) or 32)
        )
        force_miss_samples = max(
            probe_samples,
            int(getattr(config, "cache_latency_force_miss_samples", 64) or 64),
        )
        min_hits = max(0, int(getattr(config, "cache_latency_min_hits", 4) or 4))
        samples = int(stats.get("samples", 0) or 0)
        hit_count = int(stats.get("hit_count", 0) or 0)
        if samples < probe_samples:
            return False
        if hit_count == 0 and samples < force_miss_samples:
            return False
        if 0 < hit_count < min_hits:
            return False
        target_ms = float(getattr(config, "budget_latency_target_ms", 1200.0) or 1200.0)
        p95_multiplier = float(getattr(config, "cache_latency_p95_multiplier", 1.15) or 1.15)
        min_hit_rate = float(getattr(config, "cache_latency_min_hit_rate", 0.1) or 0.1)
        return (
            float(stats.get("p95_latency_ms", 0.0) or 0.0) > (target_ms * p95_multiplier)
            and float(stats.get("hit_rate", 0.0) or 0.0) <= min_hit_rate
        )

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


_tracker = CacheLatencyTracker()


def record_cache_stage_outcome(route_key: str, stage: str, *, latency_ms: float, hit: bool) -> None:
    _tracker.record(route_key, stage, latency_ms=latency_ms, hit=hit)


def cache_stage_latency_stats(route_key: str, stage: str) -> dict[str, float]:
    return _tracker.summary(route_key, stage)


def should_bypass_cache_stage(route_key: str, stage: str, config: Any) -> bool:
    return _tracker.should_bypass(route_key, stage, config)


def clear_cache_stage_latency() -> None:
    _tracker.clear()
