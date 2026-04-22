"""Shared annotation and reporting helpers for benchmark suite execution."""

from __future__ import annotations

from typing import Any

from byte.benchmarking.contracts import RunPhase
from byte.benchmarking.corpus import FAMILY_ORDER
from byte.benchmarking.metrics import (
    cold_warm_gain,
    compare_summaries,
    degradation_accuracy_delta,
    paired_delta_stats,
)


def annotate_record(
    record: dict[str, Any],
    *,
    family_metadata: dict[str, dict[str, Any]],
    scorecard_mode: str,
    judge_mode: str,
    run_id: str,
    replicate_id: int,
) -> dict[str, Any]:
    """Attach report metadata to a single benchmark record."""
    payload = dict(record or {})
    family_name = str(payload.get("family", "") or "")
    metadata = dict(family_metadata.get(family_name, {}) or {})
    answered = record_answered(payload)
    payload["scorecard_mode"] = scorecard_mode
    payload["answered"] = answered
    payload["abstained"] = not answered
    payload["coverage_eligible"] = True
    payload["forced_answer_correct"] = bool(payload.get("output_correct", False)) if answered else False
    payload["selective_correct"] = bool(payload.get("output_correct", False)) if answered else False
    payload["replicate_id"] = replicate_id
    payload["run_id"] = run_id
    payload["confidence_interval"] = {}
    payload["contamination_status"] = str(metadata.get("contamination_status", "") or "unknown")
    payload["benchmark_lane"] = str(metadata.get("family_lane", "") or "unknown")
    payload["judge_mode"] = judge_mode
    payload["live_cutoff_date"] = metadata.get("live_cutoff_date")
    return payload


def record_answered(record: dict[str, Any]) -> bool:
    """Return whether a record contributed an answer to the scorecard."""
    if int(record.get("status_code", 0) or 0) != 200:
        return False
    response_text = str(record.get("response_text", "") or "").strip()
    if response_text:
        return True
    return any(key in record for key in ("output_correct", "forced_answer_correct", "selective_correct"))


def attach_comparisons(results: dict[str, Any]) -> None:
    """Attach direct-baseline deltas to every comparable provider/system/phase."""
    for provider, provider_payload in (results.get("providers", {}) or {}).items():
        direct = (provider_payload.get("systems", {}) or {}).get("direct", {})
        byte_system = (provider_payload.get("systems", {}) or {}).get("byte", {})
        if not byte_system:
            continue
        warm_100 = (byte_system.get("phases", {}) or {}).get(RunPhase.WARM_100.value)
        warm_1000 = (byte_system.get("phases", {}) or {}).get(RunPhase.WARM_1000.value)
        cold = (byte_system.get("phases", {}) or {}).get(RunPhase.COLD.value)
        if cold and warm_100:
            provider_payload["cold_to_warm_gain_100"] = cold_warm_gain(
                cold["summary"],
                warm_100["summary"],
            )
        if cold and warm_1000:
            provider_payload["cold_to_warm_gain_1000"] = cold_warm_gain(
                cold["summary"],
                warm_1000["summary"],
            )
        if not direct:
            continue
        for phase_name, byte_phase in (byte_system.get("phases", {}) or {}).items():
            direct_phase = (direct.get("phases", {}) or {}).get(phase_name)
            if not direct_phase:
                continue
            comparison = compare_summaries(byte_phase["summary"], direct_phase["summary"])
            comparison["paired_deltas"] = paired_delta_stats(
                [payload["summary"] for payload in byte_phase.get("replicates", [])],
                [payload["summary"] for payload in direct_phase.get("replicates", [])],
                confidence_level=float(results.get("confidence_level", 0.95) or 0.95),
            )
            byte_phase["comparison_to_direct"] = comparison
            for family_name in results.get("family_order", []) or FAMILY_ORDER:
                direct_family = (direct_phase.get("families", {}) or {}).get(family_name)
                byte_family = (byte_phase.get("families", {}) or {}).get(family_name)
                if not direct_family or not byte_family:
                    continue
                byte_family["comparison_to_direct"] = compare_summaries(
                    byte_family["summary"],
                    direct_family["summary"],
                )
        cold_direct = (direct.get("phases", {}) or {}).get(RunPhase.COLD.value)
        cold_byte = (byte_system.get("phases", {}) or {}).get(RunPhase.COLD.value)
        if cold_direct and cold_byte:
            degradation_family_direct = (cold_direct.get("families", {}) or {}).get("degradation_unseen")
            degradation_family_byte = (cold_byte.get("families", {}) or {}).get("degradation_unseen")
            if degradation_family_direct and degradation_family_byte:
                provider_payload["degradation_accuracy_delta"] = degradation_accuracy_delta(
                    degradation_family_byte["summary"],
                    degradation_family_direct["summary"],
                )


def attach_highlights(results: dict[str, Any]) -> None:
    """Derive release-facing narrative highlights from provider summaries."""
    highlights: list[str] = []
    for provider, provider_payload in (results.get("providers", {}) or {}).items():
        byte_phase = preferred_phase(provider_payload, "byte")
        direct_phase = preferred_phase(provider_payload, "direct")
        if not byte_phase or not direct_phase:
            continue
        comparison = dict(byte_phase.get("comparison_to_direct", {}) or {})
        summary = dict(byte_phase.get("summary", {}) or {})
        forced_ci = dict((summary.get("ci_95", {}) or {}).get("forced_answer_accuracy", {}) or {})
        selective_ci = dict((summary.get("ci_95", {}) or {}).get("selective_accuracy", {}) or {})
        highlights.append(
            f"{provider}: Byte forced accuracy {float(summary.get('forced_answer_accuracy', 0.0) or 0.0):.4f} "
            f"(95% CI {float(forced_ci.get('low', 0.0) or 0.0):.4f}-{float(forced_ci.get('high', 0.0) or 0.0):.4f}), "
            f"selective accuracy {float(summary.get('selective_accuracy', 0.0) or 0.0):.4f} at coverage "
            f"{float(summary.get('coverage', 0.0) or 0.0):.4f}."
        )
        highlights.append(
            f"{provider}: Byte cost reduction {float(comparison.get('cost_reduction_ratio', 0.0) or 0.0):.4f}, "
            f"latency improvement {float(comparison.get('latency_improvement_ratio', 0.0) or 0.0):.4f}, "
            f"forced accuracy delta {float(comparison.get('accuracy_delta', 0.0) or 0.0):.4f}."
        )
        highlights.append(
            f"{provider}: contamination {summary.get('contamination_status', '')}, false reuse rate "
            f"{float(summary.get('false_reuse_rate', 0.0) or 0.0):.4f}, fallback trigger rate "
            f"{float(summary.get('fallback_trigger_rate', 0.0) or 0.0):.4f}, confidence accuracy "
            f"{float(summary.get('confidence_score_accuracy', 0.0) or 0.0):.4f}."
        )
        if "prompt_token_reduction_ratio" in summary:
            highlights.append(
                f"{provider}: prompt reduction {float(summary.get('prompt_token_reduction_ratio', 0.0) or 0.0):.4f}, "
                f"faithfulness {float(summary.get('faithfulness_pass_rate', 0.0) or 0.0):.4f}, module reuse "
                f"{float(summary.get('module_reuse_rate', 0.0) or 0.0):.4f}."
            )
    results["highlights"] = highlights


def attach_dollar_impact(results: dict[str, Any]) -> None:
    """Compute simple volume-normalized dollar impact projections."""
    impact: dict[str, dict[str, float]] = {}
    direct_total = 0.0
    byte_total = 0.0
    request_count = 0
    for provider_payload in (results.get("providers", {}) or {}).values():
        direct_phase = preferred_phase(provider_payload, "direct")
        byte_phase = preferred_phase(provider_payload, "byte")
        if not direct_phase or not byte_phase:
            continue
        direct_total += float((direct_phase.get("summary", {}) or {}).get("cost_usd", 0.0) or 0.0)
        byte_total += float((byte_phase.get("summary", {}) or {}).get("cost_usd", 0.0) or 0.0)
        request_count += int((direct_phase.get("summary", {}) or {}).get("sample_size", 0) or 0)
    if request_count <= 0:
        results["dollar_impact"] = impact
        return
    for label, volume in (("100k", 100_000), ("1m", 1_000_000), ("10m", 10_000_000)):
        direct_cost = direct_total * (volume / request_count)
        byte_cost = byte_total * (volume / request_count)
        impact[label] = {
            "direct_cost_usd": round(direct_cost, 2),
            "byte_cost_usd": round(byte_cost, 2),
            "savings_usd": round(direct_cost - byte_cost, 2),
        }
    results["dollar_impact"] = impact


def dedupe_lists(results: dict[str, Any]) -> None:
    """Normalize duplicated provider/system execution lists after the run."""
    results["providers_executed"] = sorted(set(results.get("providers_executed", []) or []))
    results["systems_executed"] = sorted(set(results.get("systems_executed", []) or []))


def preferred_phase(provider_payload: dict[str, Any], system_name: str) -> dict[str, Any] | None:
    """Return the warmest available phase for a given provider/system pair."""
    phases = (
        ((provider_payload.get("systems", {}) or {}).get(system_name, {}) or {}).get("phases", {})
        or {}
    )
    for candidate in (RunPhase.WARM_100.value, RunPhase.WARM_1000.value, RunPhase.COLD.value):
        if candidate in phases:
            return phases[candidate]
    return None


def release_track(profile_data: dict[str, Any], max_items_per_family: int | None) -> str:
    """Describe whether the suite was a partial run or a release-track execution."""
    if max_items_per_family is not None:
        return "partial"
    if str(profile_data.get("benchmark_track", "") or "") == "provider_local":
        return "provider_local_release"
    return "platform_global_release"


__all__ = [
    "annotate_record",
    "attach_comparisons",
    "attach_dollar_impact",
    "attach_highlights",
    "dedupe_lists",
    "preferred_phase",
    "record_answered",
    "release_track",
]
