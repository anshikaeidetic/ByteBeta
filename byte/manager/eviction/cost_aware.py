"""Cost-aware cache eviction (arXiv 2508.07675).

Based on "Semantic Caching for Low-Cost LLM Serving: From Offline Learning to
Online Adaptation" (Zhao et al., August 2025). The paper argues that semantic
caches have a fundamentally different eviction problem than traditional
caches: the cost of serving a cached-but-mismatched answer (user sees wrong /
low-quality content) differs from the cost of a simple miss (one more LLM
call). Their algorithm weights each cached entry by an estimated mismatch
cost and evicts the entries with the LOWEST keep-value when room is needed.

This module provides `CostAwareCacheEviction`, a drop-in replacement for
`MemoryCacheEviction` that:

1. Tracks a per-entry cost score in [0, 1] supplied at `put()` time.
2. Evicts the entry with the lowest (cost_score * recency_decay) first.
3. Emits `byteai_eviction_cost_aware_*` counters on every eviction.

Score interpretation: HIGHER score = more valuable to keep. If no score is
supplied, the entry defaults to 0.5 (neutral). The recency decay multiplier
gives very recent inserts a small bonus so the policy isn't purely cost-greedy.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from byte.manager.eviction.base import EvictionBase
from byte.utils.log import byte_log


class CostAwareCacheEviction(EvictionBase):
    """Evicts the lowest-value cached entry first (arXiv 2508.07675)."""

    def __init__(
        self,
        maxsize: int = 1000,
        clean_size: int = 0,
        on_evict: Callable[[list[Any]], None] | None = None,
        default_score: float = 0.5,
        recency_half_life_s: float = 3600.0,
        **_kwargs: Any,
    ) -> None:
        self._maxsize = max(1, int(maxsize))
        self._clean_size = max(1, int(clean_size or max(1, self._maxsize // 10)))
        self._on_evict = on_evict
        self._default_score = max(0.0, min(1.0, float(default_score)))
        # λ for exponential decay: score_t = score_0 * exp(-t / half_life)
        self._half_life = max(1.0, float(recency_half_life_s))
        self._items: dict[Any, dict[str, float]] = {}  # key -> {"score": ..., "ts": ...}

    # ── public API (matches EvictionBase + MemoryCacheEviction semantics) ──

    def put(self, objs: list[Any]) -> None:
        """Insert one or more keys. Each entry stores the default_score until overridden.

        Callers that know the per-entry quality should use `put_with_score()`.
        """
        for key in objs or []:
            self._insert(key, self._default_score)

    def put_with_score(self, key: Any, score: float) -> None:
        """Insert a key with an explicit cost score in [0, 1]."""
        self._insert(key, score)

    def record_score(self, key: Any, score: float) -> None:
        """Update the cost score of an existing entry (e.g., after a quality check)."""
        if key in self._items:
            s = max(0.0, min(1.0, float(score)))
            self._items[key]["score"] = s

    def get(self, obj: Any) -> Any:
        """Probe the eviction store; updates the recency timestamp as a side effect."""
        entry = self._items.get(obj)
        if entry is None:
            return None
        entry["ts"] = time.time()  # refresh recency on access
        return True

    @property
    def policy(self) -> str:
        return "COST_AWARE"

    # ── internal ──

    def _insert(self, key: Any, score: float) -> None:
        now = time.time()
        s = max(0.0, min(1.0, float(score)))
        self._items[key] = {"score": s, "ts": now}
        if len(self._items) > self._maxsize:
            self._evict_lowest_value()

    def _value(self, entry: dict[str, float], now: float) -> float:
        """Composite keep-value: raw score with exponential recency decay."""
        age = max(0.0, now - entry.get("ts", now))
        # Decay factor in (0, 1]; older entries retain less of their base score.
        decay = 2.0 ** (-age / self._half_life)
        return entry["score"] * decay

    def _evict_lowest_value(self) -> None:
        """Remove the `clean_size` entries with the smallest keep-value."""
        now = time.time()
        ranked = sorted(
            self._items.items(),
            key=lambda kv: self._value(kv[1], now),
        )
        victims: list[Any] = []
        total_saved = 0.0
        to_remove = min(self._clean_size, max(1, len(self._items) - self._maxsize + self._clean_size - 1))
        for key, entry in ranked[:to_remove]:
            victims.append(key)
            total_saved += entry.get("score", 0.0)
            self._items.pop(key, None)

        if victims:
            try:
                from byte.telemetry import (
                    bump_research_counter as _bump,  # pylint: disable=import-outside-toplevel
                )
                _bump("eviction_cost_aware_evictions", len(victims))
                _bump("eviction_cost_aware_savings", int(total_saved * 1000))
            except Exception:  # pragma: no cover - defensive
                pass
            if self._on_evict is not None:
                try:
                    self._on_evict(victims)
                except Exception as exc:  # pylint: disable=W0703
                    byte_log.warning("cost-aware eviction on_evict hook failed: %s", exc)

    # ── introspection ──

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of the eviction store for observability."""
        now = time.time()
        if not self._items:
            return {"size": 0, "avg_score": 0.0, "avg_value": 0.0}
        scores = [e["score"] for e in self._items.values()]
        values = [self._value(e, now) for e in self._items.values()]
        return {
            "size": len(self._items),
            "avg_score": sum(scores) / len(scores),
            "avg_value": sum(values) / len(values),
            "min_value": min(values),
            "max_value": max(values),
        }


__all__ = ["CostAwareCacheEviction"]
