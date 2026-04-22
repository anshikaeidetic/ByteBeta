"""Adaptive similarity threshold — auto-tunes the cache hit boundary.

Tracks a rolling window of similarity scores from recent cache lookups
and adjusts the threshold to approach a target hit rate, staying within
safe bounds so output quality is never degraded.

This feature is **output-safe**: it only changes the cache-hit/miss
boundary — never the content of cached answers.
"""

from collections import deque


class AdaptiveThreshold:
    """Self-tuning similarity threshold.

    :param base_threshold: the user-configured starting threshold (0–1)
    :param target_hit_rate: desired fraction of requests served from cache
    :param floor: minimum allowed threshold (safety bound)
    :param ceiling: maximum allowed threshold
    :param window_size: number of recent lookups to consider
    :param step: how much to adjust per recalculation

    Example::

        at = AdaptiveThreshold(base_threshold=0.8, target_hit_rate=0.5)

        for request in requests:
            threshold = at.current_threshold
            hit = check_cache(request, threshold)
            at.record(hit=hit, score=similarity_score)
    """

    def __init__(
        self,
        base_threshold: float = 0.8,
        target_hit_rate: float = 0.5,
        floor: float = 0.6,
        ceiling: float = 0.95,
        window_size: int = 200,
        step: float = 0.01,
    ) -> None:
        self._base = base_threshold
        self._threshold = base_threshold
        self._target = target_hit_rate
        self._floor = max(floor, 0.0)
        self._ceiling = min(ceiling, 1.0)
        self._step = step
        self._window: deque = deque(maxlen=window_size)
        self._hit_count = 0
        self._total_count = 0

    @property
    def current_threshold(self) -> float:
        """The current (possibly adjusted) similarity threshold."""
        return self._threshold

    def record(self, hit: bool, score: float | None = None) -> None:
        """Record the outcome of a cache lookup.

        :param hit: whether the lookup was a cache hit
        :param score: the similarity score (if available)
        """
        self._window.append(hit)
        self._total_count += 1
        if hit:
            self._hit_count += 1

        # Re-evaluate every time the window is full
        if len(self._window) == self._window.maxlen:
            self._adjust()

    def _adjust(self) -> None:
        """Nudge the threshold toward the target hit rate."""
        current_rate = sum(self._window) / len(self._window)

        if current_rate < self._target:
            # Too few hits → lower threshold to be more permissive
            self._threshold = max(self._floor, self._threshold - self._step)
        elif current_rate > self._target + 0.1:
            # Too many hits → raise threshold to be more selective
            self._threshold = min(self._ceiling, self._threshold + self._step)
        # else: within acceptable band, don't change

    def stats(self) -> dict:
        """Return current adaptive threshold statistics."""
        window_hits = sum(self._window) if self._window else 0
        window_size = len(self._window)
        return {
            "base_threshold": self._base,
            "current_threshold": round(self._threshold, 4),
            "target_hit_rate": self._target,
            "window_hit_rate": round(window_hits / window_size, 4) if window_size else 0.0,
            "window_size": window_size,
            "total_lookups": self._total_count,
            "total_hits": self._hit_count,
            "floor": self._floor,
            "ceiling": self._ceiling,
        }

    def reset(self) -> None:
        """Reset to the base threshold and clear history."""
        self._threshold = self._base
        self._window.clear()
        self._hit_count = 0
        self._total_count = 0
