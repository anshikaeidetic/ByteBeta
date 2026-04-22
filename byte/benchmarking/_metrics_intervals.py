"""Confidence-interval and percentile helpers for benchmark summaries."""

from __future__ import annotations

import math
import statistics

_Z_SCORES = {
    0.8: 1.2816,
    0.9: 1.6449,
    0.95: 1.96,
    0.98: 2.3263,
    0.99: 2.5758,
}
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def percentile(values: list[float], quantile: float) -> float:
    """Return a deterministic interpolated percentile for latency series."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def proportion_confidence_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Return a Wilson interval for proportion-style benchmark metrics."""
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    z_score = _z_score(confidence_level)
    observed = float(successes) / float(total)
    denominator = 1.0 + (z_score * z_score) / float(total)
    center = (observed + (z_score * z_score) / (2.0 * float(total))) / denominator
    margin = (
        z_score
        * math.sqrt(
            (observed * (1.0 - observed) + (z_score * z_score) / (4.0 * float(total)))
            / float(total)
        )
        / denominator
    )
    return {
        "low": round(max(0.0, center - margin), 4),
        "high": round(min(1.0, center + margin), 4),
    }


def mean_confidence_interval(
    values: list[float],
    *,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Return a t-interval for replicate-level metrics."""
    if not values:
        return {"mean": 0.0, "stddev": 0.0, "low": 0.0, "high": 0.0, "count": 0}
    mean_value = float(statistics.mean(values))
    if len(values) == 1:
        rounded = round(mean_value, 4)
        return {
            "mean": rounded,
            "stddev": 0.0,
            "low": rounded,
            "high": rounded,
            "count": 1,
        }
    stddev_value = float(statistics.stdev(values))
    critical = _t_critical(confidence_level, len(values) - 1)
    margin = critical * stddev_value / math.sqrt(len(values))
    return {
        "mean": round(mean_value, 4),
        "stddev": round(stddev_value, 4),
        "low": round(mean_value - margin, 4),
        "high": round(mean_value + margin, 4),
        "count": len(values),
    }


def _z_score(confidence_level: float) -> float:
    normalized = round(float(confidence_level or 0.95), 2)
    return _Z_SCORES.get(normalized, 1.96)


def _t_critical(confidence_level: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        return 0.0
    normalized = round(float(confidence_level or 0.95), 2)
    if normalized != 0.95:
        return _z_score(normalized)
    if degrees_of_freedom in _T_CRITICAL_95:
        return _T_CRITICAL_95[degrees_of_freedom]
    if degrees_of_freedom > 30:
        return 1.96
    nearest = max(key for key in _T_CRITICAL_95 if key <= degrees_of_freedom)
    return _T_CRITICAL_95[nearest]


__all__ = [
    "mean_confidence_interval",
    "percentile",
    "proportion_confidence_interval",
]
