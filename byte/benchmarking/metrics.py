from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

_REPLICATE_METRICS = (
    "forced_answer_accuracy",
    "selective_accuracy",
    "coverage",
    "safe_answer_rate",
    "abstention_precision",
    "accuracy_ratio",
    "avg_latency_ms",
    "cost_usd",
    "false_reuse_rate",
    "fallback_trigger_rate",
    "confidence_score_accuracy",
    "confidence_ece",
    "deterministic_output_rate",
    "prompt_token_reduction_ratio",
    "faithfulness_pass_rate",
)
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


def summarize_records(
    records: list[dict[str, Any]],
    *,
    scorecard_mode: str = "dual",
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    latencies = [float(record.get("latency_ms", 0.0) or 0.0) for record in records]
    eligible_records = [record for record in records if bool(record.get("coverage_eligible", True))]
    answered_records = [record for record in eligible_records if _record_answered(record)]
    abstained_records = [record for record in eligible_records if _record_abstained(record)]
    total = len(eligible_records)
    answered_total = len(answered_records)
    abstained_total = len(abstained_records)
    unsafe_total = sum(1 for record in eligible_records if not bool(record.get("reuse_safe", False)))
    safe_total = sum(1 for record in eligible_records if bool(record.get("reuse_safe", False)))
    fallback_total = sum(1 for record in eligible_records if bool(record.get("must_fallback", False)))
    deterministic_total = sum(
        1 for record in eligible_records if bool(record.get("deterministic_expected", False))
    )
    workflow_total_steps = sum(
        int(record.get("workflow_total_steps", 0) or 0) for record in eligible_records
    )
    workflow_steps_skipped = sum(
        int(record.get("workflow_steps_skipped", 0) or 0) for record in eligible_records
    )
    original_prompt_tokens = sum(
        int(record.get("original_prompt_tokens", 0) or 0) for record in eligible_records
    )
    distilled_prompt_tokens = sum(
        int(record.get("distilled_prompt_tokens", 0) or 0) for record in eligible_records
    )
    original_prompt_chars = sum(
        int(record.get("original_prompt_chars", 0) or 0) for record in eligible_records
    )
    distilled_prompt_chars = sum(
        int(record.get("distilled_prompt_chars", 0) or 0) for record in eligible_records
    )
    forced_correct = sum(
        bool(record.get("forced_answer_correct", bool(record.get("output_correct", False))))
        for record in eligible_records
    )
    selective_correct = sum(
        bool(record.get("selective_correct", bool(record.get("output_correct", False))))
        for record in answered_records
    )
    safe_answer_correct = sum(
        bool(record.get("output_correct", False)) and bool(record.get("policy_adherent", False))
        for record in answered_records
    )
    abstention_correct = sum(_abstention_correct(record) for record in abstained_records)
    confidence_intervals = {
        "forced_answer_accuracy": proportion_confidence_interval(
            forced_correct,
            total,
            confidence_level=confidence_level,
        ),
        "selective_accuracy": proportion_confidence_interval(
            selective_correct,
            answered_total,
            confidence_level=confidence_level,
        ),
        "coverage": proportion_confidence_interval(
            answered_total,
            total,
            confidence_level=confidence_level,
        ),
        "safe_answer_rate": proportion_confidence_interval(
            safe_answer_correct,
            total,
            confidence_level=confidence_level,
        ),
    }
    contamination_counts = Counter(
        str(record.get("contamination_status", "") or "unknown") for record in eligible_records
    )
    lane_counts = Counter(str(record.get("benchmark_lane", "") or "unknown") for record in eligible_records)
    summary = {
        "request_count": len(records),
        "sample_size": total,
        "answered_count": answered_total,
        "abstained_count": abstained_total,
        "error_count": sum(1 for record in records if int(record.get("status_code", 0) or 0) != 200),
        "scorecard_mode": scorecard_mode,
        "forced_answer_accuracy": _ratio(forced_correct, total),
        "selective_accuracy": _ratio(selective_correct, answered_total),
        "coverage": _ratio(answered_total, total),
        "safe_answer_rate": _ratio(safe_answer_correct, total),
        "abstention_precision": _ratio(abstention_correct, abstained_total),
        "accuracy_ratio": _ratio(forced_correct, total),
        "scorecard_explanation": _scorecard_explanation(total, answered_total, abstained_total),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "tokens": {
            "prompt_tokens": sum(
                int(record.get("tokens", {}).get("prompt_tokens", 0) or 0) for record in records
            ),
            "cached_prompt_tokens": sum(
                int(record.get("tokens", {}).get("cached_prompt_tokens", 0) or 0) for record in records
            ),
            "completion_tokens": sum(
                int(record.get("tokens", {}).get("completion_tokens", 0) or 0) for record in records
            ),
        },
        "cost_usd": round(sum(float(record.get("cost_usd", 0.0) or 0.0) for record in records), 8),
        "upstream_calls": sum(int(record.get("upstream_calls", 0) or 0) for record in records),
        "actual_reuse_rate": _ratio(
            sum(bool(record.get("actual_reuse", False)) for record in eligible_records),
            total,
        ),
        "false_reuse_rate": _ratio(
            sum(
                bool(record.get("actual_reuse", False)) and not bool(record.get("reuse_safe", False))
                for record in eligible_records
            ),
            unsafe_total,
        ),
        "safe_reuse_recall": _ratio(
            sum(
                bool(record.get("actual_reuse", False)) and bool(record.get("reuse_safe", False))
                for record in eligible_records
            ),
            safe_total,
        ),
        "fallback_trigger_rate": _ratio(
            sum(
                bool(record.get("fallback_taken", False)) and bool(record.get("must_fallback", False))
                for record in eligible_records
            ),
            fallback_total,
        ),
        "confidence_score_accuracy": _ratio(
            sum(
                (float(record.get("reuse_confidence", 0.0) or 0.0) >= 0.80)
                == bool(record.get("reuse_safe", False))
                for record in eligible_records
            ),
            total,
        ),
        "confidence_ece": expected_calibration_error(eligible_records),
        "policy_adherence_rate": _ratio(
            sum(bool(record.get("policy_adherent", False)) for record in eligible_records),
            total,
        ),
        "deterministic_output_rate": _ratio(
            sum(
                bool(record.get("deterministic_output", False))
                for record in eligible_records
                if bool(record.get("deterministic_expected", False))
            ),
            deterministic_total,
        ),
        "original_prompt_tokens": original_prompt_tokens,
        "distilled_prompt_tokens": distilled_prompt_tokens,
        "prompt_token_reduction_ratio": _delta_ratio(
            original_prompt_tokens,
            distilled_prompt_tokens,
        ),
        "compression_ratio": _delta_ratio(
            original_prompt_chars,
            distilled_prompt_chars,
        ),
        "faithfulness_pass_rate": _ratio(
            sum(bool(record.get("faithfulness_pass", False)) for record in eligible_records),
            total,
        ),
        "faithfulness_score": round(
            sum(float(record.get("faithfulness_score", 0.0) or 0.0) for record in eligible_records)
            / max(total, 1),
            4,
        ),
        "entity_preservation_rate": round(
            sum(float(record.get("entity_preservation_rate", 1.0) or 1.0) for record in eligible_records)
            / max(total, 1),
            4,
        ),
        "schema_preservation_rate": round(
            sum(float(record.get("schema_preservation_rate", 1.0) or 1.0) for record in eligible_records)
            / max(total, 1),
            4,
        ),
        "module_reuse_rate": _ratio(
            sum(int(record.get("module_hits", 0) or 0) > 0 for record in eligible_records),
            total,
        ),
        "distillation_fallback_rate": _ratio(
            sum(bool(record.get("distillation_fallback", False)) for record in eligible_records),
            total,
        ),
        "workflow_step_reduction": _ratio(workflow_steps_skipped, workflow_total_steps),
        "served_via_counts": dict(
            Counter(str(record.get("served_via", "") or "unknown") for record in records)
        ),
        "contamination_status": _dominant_value(contamination_counts),
        "contamination_status_counts": dict(contamination_counts),
        "benchmark_lane": _dominant_value(lane_counts),
        "lane_counts": dict(lane_counts),
        "scorecards": {
            "forced": {
                "accuracy": _ratio(forced_correct, total),
                "sample_size": total,
                "coverage": 1.0 if total else 0.0,
                "ci": confidence_intervals["forced_answer_accuracy"],
            },
            "selective": {
                "accuracy": _ratio(selective_correct, answered_total),
                "sample_size": answered_total,
                "coverage": _ratio(answered_total, total),
                "ci": confidence_intervals["selective_accuracy"],
            },
        },
        "confidence_interval": confidence_intervals,
        "confidence_level": confidence_level,
    }
    if math.isclose(confidence_level, 0.95, rel_tol=0.0, abs_tol=1e-9):
        summary["ci_95"] = confidence_intervals
    summary["monthly_cost_simulation"] = {
        "100k": round(summary["cost_usd"] * (100_000 / max(total, 1)), 2),
        "1m": round(summary["cost_usd"] * (1_000_000 / max(total, 1)), 2),
        "10m": round(summary["cost_usd"] * (10_000_000 / max(total, 1)), 2),
    }
    return summary


def expected_calibration_error(records: list[dict[str, Any]], buckets: int = 10) -> float:
    if not records:
        return 0.0
    total = len(records)
    bucket_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        confidence = min(max(float(record.get("reuse_confidence", 0.0) or 0.0), 0.0), 1.0)
        index = min(int(confidence * buckets), buckets - 1)
        bucket_records[index].append(record)
    error = 0.0
    for bucket in bucket_records.values():
        if not bucket:
            continue
        avg_confidence = sum(float(record.get("reuse_confidence", 0.0) or 0.0) for record in bucket) / len(bucket)
        avg_accuracy = sum(bool(record.get("reuse_safe", False)) for record in bucket) / len(bucket)
        error += abs(avg_confidence - avg_accuracy) * (len(bucket) / total)
    return round(error, 4)


def group_records(records: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(field, "") or "")].append(record)
    return dict(grouped)


def compare_summaries(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_cost = float(baseline.get("cost_usd", 0.0) or 0.0)
    baseline_latency = float(baseline.get("avg_latency_ms", 0.0) or 0.0)
    baseline_accuracy = float(baseline.get("accuracy_ratio", 0.0) or 0.0)
    baseline_prompt_tokens = int((baseline.get("tokens", {}) or {}).get("prompt_tokens", 0) or 0)
    return {
        "cost_reduction_ratio": _delta_ratio(baseline_cost, float(summary.get("cost_usd", 0.0) or 0.0)),
        "latency_improvement_ratio": _delta_ratio(
            baseline_latency,
            float(summary.get("avg_latency_ms", 0.0) or 0.0),
        ),
        "accuracy_delta": round(
            float(summary.get("accuracy_ratio", 0.0) or 0.0) - baseline_accuracy,
            4,
        ),
        "token_reduction_ratio": _delta_ratio(
            baseline_prompt_tokens,
            int((summary.get("tokens", {}) or {}).get("prompt_tokens", 0) or 0),
        ),
        "coverage_delta": round(
            float(summary.get("coverage", 0.0) or 0.0) - float(baseline.get("coverage", 0.0) or 0.0),
            4,
        ),
        "selective_accuracy_delta": round(
            float(summary.get("selective_accuracy", 0.0) or 0.0)
            - float(baseline.get("selective_accuracy", 0.0) or 0.0),
            4,
        ),
    }


def cold_warm_gain(cold_summary: dict[str, Any], warm_summary: dict[str, Any]) -> float:
    return _delta_ratio(
        float(cold_summary.get("cost_usd", 0.0) or 0.0),
        float(warm_summary.get("cost_usd", 0.0) or 0.0),
    )


def degradation_accuracy_delta(byte_summary: dict[str, Any], direct_summary: dict[str, Any]) -> float:
    return round(
        float(byte_summary.get("accuracy_ratio", 0.0) or 0.0)
        - float(direct_summary.get("accuracy_ratio", 0.0) or 0.0),
        4,
    )


def aggregate_replicate_summaries(
    summaries: list[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for metric_name in _REPLICATE_METRICS:
        values = [float(summary.get(metric_name, 0.0) or 0.0) for summary in summaries]
        stats[metric_name] = mean_confidence_interval(values, confidence_level=confidence_level)
    return stats


def paired_delta_stats(
    observed_summaries: list[dict[str, Any]],
    baseline_summaries: list[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    pairs = list(zip(observed_summaries, baseline_summaries))
    if not pairs:
        return {}
    delta_metrics: dict[str, list[float]] = defaultdict(list)
    for observed, baseline in pairs:
        comparison = compare_summaries(observed, baseline)
        for metric_name, metric_value in comparison.items():
            delta_metrics[metric_name].append(float(metric_value or 0.0))
    return {
        metric_name: mean_confidence_interval(values, confidence_level=confidence_level)
        for metric_name, values in delta_metrics.items()
    }


def proportion_confidence_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float]:
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
    if not values:
        return {"mean": 0.0, "stddev": 0.0, "low": 0.0, "high": 0.0, "count": 0}
    mean_value = float(statistics.mean(values))
    if len(values) == 1:
        return {
            "mean": round(mean_value, 4),
            "stddev": 0.0,
            "low": round(mean_value, 4),
            "high": round(mean_value, 4),
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


def _record_answered(record: dict[str, Any]) -> bool:
    if "answered" in record:
        return bool(record.get("answered"))
    if int(record.get("status_code", 0) or 0) != 200:
        return False
    response_text = str(record.get("response_text", "") or "").strip()
    if response_text:
        return True
    return any(key in record for key in ("output_correct", "forced_answer_correct", "selective_correct"))


def _record_abstained(record: dict[str, Any]) -> bool:
    if "abstained" in record:
        return bool(record.get("abstained"))
    return not _record_answered(record)


def _abstention_correct(record: dict[str, Any]) -> bool:
    if not _record_abstained(record):
        return False
    return bool(record.get("must_fallback", False) or not bool(record.get("reuse_safe", False)))


def _scorecard_explanation(total: int, answered_total: int, abstained_total: int) -> str:
    if total <= 0:
        return ""
    if abstained_total <= 0:
        return ""
    return (
        f"Selective accuracy is computed over {answered_total} answered cases; "
        f"{abstained_total} of {total} eligible cases abstained."
    )


def _dominant_value(counter: Counter[str]) -> str:
    if not counter:
        return ""
    if len(counter) == 1:
        return next(iter(counter))
    return "mixed"


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _delta_ratio(baseline: float, observed: float) -> float:
    if baseline == 0:
        return 0.0
    return round((baseline - observed) / baseline, 4)


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
