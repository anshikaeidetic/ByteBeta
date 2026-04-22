from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from datetime import datetime
from typing import Any
from uuid import uuid4

from byte.benchmarking.contracts import BenchmarkItem, RunPhase
from byte.benchmarking.corpus import FAMILY_ORDER, load_profile, validate_profile
from byte.benchmarking.integrity import (
    BENCHMARK_CONTRACT_VERSION,
    BENCHMARK_REPORT_VERSION,
    BENCHMARK_SCORING_VERSION,
    BENCHMARK_TRUST_POLICY_VERSION,
)
from byte.benchmarking.metrics import (
    aggregate_replicate_summaries,
    cold_warm_gain,
    compare_summaries,
    degradation_accuracy_delta,
    group_records,
    paired_delta_stats,
    summarize_records,
)
from byte.benchmarking.reporting import write_report_bundle
from byte.benchmarking.systems import ExecutableSystem, build_systems


def run_suite(
    *,
    profile: str = "tier1",
    providers: list[str] | None = None,
    systems: list[str] | None = None,
    phases: list[str] | None = None,
    out_dir: str,
    fail_on_thresholds: bool = True,
    max_items_per_family: int | None = None,
    concurrency: int = 4,
    scorecard_mode: str = "dual",
    replicates: int = 1,
    confidence_level: float = 0.95,
    judge_mode: str = "disabled",
    contamination_check: bool = False,
    live_cutoff_date: str = "",
    release_gate: bool = False,
) -> dict[str, Any]:
    phase_names = [str(phase) for phase in (phases or [phase.value for phase in RunPhase])]
    profile_data = load_profile(
        profile,
        providers=providers,
        max_items_per_family=max_items_per_family,
    )
    warmup_profile = load_profile(
        profile,
        providers=providers,
        max_items_per_family=None,
    )
    corpus_summary = validate_profile(profile_data)
    executables = build_systems(profile_data["providers"], systems)
    provider_items = _group_items_by_provider(profile_data["items"])
    warmup_provider_items = _group_items_by_provider(warmup_profile["items"])
    release_track = _release_track(profile_data, max_items_per_family)
    run_id = f"{profile}-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    results: dict[str, Any] = {
        "schema_version": BENCHMARK_REPORT_VERSION,
        "run_id": run_id,
        "profile": profile,
        "profile_base": profile_data.get("profile_base", profile),
        "benchmark_track": profile_data.get("benchmark_track", "platform_global"),
        "execution_mode": release_track,
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat(),
        "corpus_version": profile_data.get("corpus_version", ""),
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "scoring_version": BENCHMARK_SCORING_VERSION,
        "trust_policy_version": BENCHMARK_TRUST_POLICY_VERSION,
        "scorecard_mode": scorecard_mode,
        "replicates": max(1, int(replicates or 1)),
        "confidence_level": float(confidence_level or 0.95),
        "judge_mode": str(judge_mode or "disabled"),
        "contamination_check": bool(contamination_check),
        "live_cutoff_date": str(live_cutoff_date or ""),
        "release_gate": bool(release_gate),
        "manifest_versions": list(profile_data.get("manifests", []) or []),
        "providers_requested": list(profile_data["providers"]),
        "providers_executed": [],
        "systems_requested": list(systems or []),
        "systems_executed": [],
        "phase_order": phase_names,
        "family_order": list(profile_data.get("families", []) or []),
        "corpus_summary": corpus_summary,
        "providers": {},
        "gate_results": [],
        "highlights": [],
        "dollar_impact": {},
    }
    for executable in executables:
        available, reason = executable.is_available()
        provider_block = results["providers"].setdefault(
            executable.spec.provider,
            {"systems": {}, "skipped_systems": {}},
        )
        if not available:
            provider_block["skipped_systems"][executable.spec.system] = reason
            continue
        items = list(provider_items.get(executable.spec.provider, []))
        if not items:
            provider_block["skipped_systems"][executable.spec.system] = "No workload items for provider."
            continue
        results["providers_executed"].append(executable.spec.provider)
        results["systems_executed"].append(executable.spec.system)
        provider_block["systems"][executable.spec.system] = {"phases": {}}
        for phase_name in phase_names:
            phase_payload = _run_phase(
                executable=executable,
                items=items,
                phase_name=phase_name,
                warmup_items=warmup_provider_items.get(executable.spec.provider, []),
                family_metadata=profile_data.get("family_metadata", {}) or {},
                scorecard_mode=scorecard_mode,
                confidence_level=float(confidence_level or 0.95),
                judge_mode=str(judge_mode or "disabled"),
                replicates=max(1, int(replicates or 1)),
                run_id=run_id,
                concurrency=concurrency,
            )
            provider_block["systems"][executable.spec.system]["phases"][phase_name] = phase_payload
    _dedupe_lists(results)
    _attach_comparisons(results)
    _attach_highlights(results)
    _attach_dollar_impact(results)
    results["gate_results"] = _evaluate_gates(results)
    results["artifacts"] = write_report_bundle(results, out_dir)
    results["failed_thresholds"] = fail_on_thresholds and any(
        not gate["passed"] for gate in results["gate_results"]
    )
    return results


def _run_phase(
    *,
    executable: ExecutableSystem,
    items: list[BenchmarkItem],
    phase_name: str,
    warmup_items: list[BenchmarkItem],
    family_metadata: dict[str, dict[str, Any]],
    scorecard_mode: str,
    confidence_level: float,
    judge_mode: str,
    replicates: int,
    run_id: str,
    concurrency: int,
) -> dict[str, Any]:
    phase_payload: dict[str, Any] = {
        "summary": {},
        "replicate_stats": {},
        "families": {},
        "records": [],
        "replicates": [],
    }
    warmup_selection = _warmup_items_for_phase(
        items,
        phase_name,
        candidate_items=warmup_items,
    )
    for replicate_id in range(1, replicates + 1):
        executable.begin_phase(warmup_selection)
        try:
            records = _run_records(executable, items, phase_name, concurrency=concurrency)
        finally:
            executable.end_phase()
        annotated_records = [
            _annotate_record(
                record,
                family_metadata=family_metadata,
                scorecard_mode=scorecard_mode,
                judge_mode=judge_mode,
                run_id=run_id,
                replicate_id=replicate_id,
            )
            for record in records
        ]
        replicate_payload = {
            "replicate_id": replicate_id,
            "run_id": run_id,
            "summary": summarize_records(
                annotated_records,
                scorecard_mode=scorecard_mode,
                confidence_level=confidence_level,
            ),
            "families": {},
            "records": annotated_records,
        }
        for family_name, family_records in group_records(annotated_records, "family").items():
            replicate_payload["families"][family_name] = {
                "summary": summarize_records(
                    family_records,
                    scorecard_mode=scorecard_mode,
                    confidence_level=confidence_level,
                ),
            }
        phase_payload["replicates"].append(replicate_payload)
        phase_payload["records"].extend(annotated_records)
    phase_payload["summary"] = summarize_records(
        phase_payload["records"],
        scorecard_mode=scorecard_mode,
        confidence_level=confidence_level,
    )
    phase_payload["replicate_stats"] = aggregate_replicate_summaries(
        [payload["summary"] for payload in phase_payload["replicates"]],
        confidence_level=confidence_level,
    )
    for family_name, family_records in group_records(phase_payload["records"], "family").items():
        replicate_summaries = [
            dict((payload.get("families", {}) or {}).get(family_name, {}).get("summary", {}) or {})
            for payload in phase_payload["replicates"]
            if family_name in (payload.get("families", {}) or {})
        ]
        phase_payload["families"][family_name] = {
            "summary": summarize_records(
                family_records,
                scorecard_mode=scorecard_mode,
                confidence_level=confidence_level,
            ),
            "replicate_stats": aggregate_replicate_summaries(
                replicate_summaries,
                confidence_level=confidence_level,
            )
            if replicate_summaries
            else {},
            "records": family_records,
        }
    return phase_payload


def _run_records(
    executable: ExecutableSystem,
    items: list[BenchmarkItem],
    phase_name: str,
    *,
    concurrency: int,
) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(concurrency or 1))) as pool:
        futures = [pool.submit(executable.run_item, item, phase=phase_name) for item in items]
        return [future.result() for future in futures]


def _warmup_items_for_phase(
    items: list[BenchmarkItem],
    phase_name: str,
    *,
    candidate_items: list[BenchmarkItem] | None = None,
) -> list[BenchmarkItem]:
    if phase_name == RunPhase.COLD.value:
        return []
    target = 100 if phase_name == RunPhase.WARM_100.value else 1000
    return _select_warmup_items(items, target=target, candidate_items=candidate_items)


def _select_warmup_items(
    scored_items: list[BenchmarkItem],
    *,
    target: int,
    candidate_items: list[BenchmarkItem] | None = None,
) -> list[BenchmarkItem]:
    pool = list(candidate_items or [])
    if pool:
        excluded_ids = {item.item_id for item in scored_items}
        disjoint_pool = [item for item in pool if item.item_id not in excluded_ids]
        if disjoint_pool:
            pool = disjoint_pool
    else:
        pool = list(scored_items)
    if not pool:
        return []
    if target <= len(pool):
        return pool[:target]
    warmup = []
    cursor = 0
    while len(warmup) < target:
        warmup.append(pool[cursor % len(pool)])
        cursor += 1
    return warmup


def _group_items_by_provider(items: list[BenchmarkItem]) -> dict[str, list[BenchmarkItem]]:
    grouped: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for item in items:
        grouped[item.provider_track].append(item)
    return dict(grouped)


def _annotate_record(
    record: dict[str, Any],
    *,
    family_metadata: dict[str, dict[str, Any]],
    scorecard_mode: str,
    judge_mode: str,
    run_id: str,
    replicate_id: int,
) -> dict[str, Any]:
    payload = dict(record or {})
    family_name = str(payload.get("family", "") or "")
    metadata = dict(family_metadata.get(family_name, {}) or {})
    answered = _record_answered(payload)
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


def _record_answered(record: dict[str, Any]) -> bool:
    if int(record.get("status_code", 0) or 0) != 200:
        return False
    response_text = str(record.get("response_text", "") or "").strip()
    if response_text:
        return True
    return any(key in record for key in ("output_correct", "forced_answer_correct", "selective_correct"))


def _attach_comparisons(results: dict[str, Any]) -> None:
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


def _attach_highlights(results: dict[str, Any]) -> None:
    highlights: list[str] = []
    for provider, provider_payload in (results.get("providers", {}) or {}).items():
        byte_phase = _preferred_phase(provider_payload, "byte")
        direct_phase = _preferred_phase(provider_payload, "direct")
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


def _attach_dollar_impact(results: dict[str, Any]) -> None:
    impact: dict[str, dict[str, float]] = {}
    direct_total = 0.0
    byte_total = 0.0
    request_count = 0
    for provider_payload in (results.get("providers", {}) or {}).values():
        direct_phase = _preferred_phase(provider_payload, "direct")
        byte_phase = _preferred_phase(provider_payload, "byte")
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


def _evaluate_gates(results: dict[str, Any]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    providers_executed = set(results.get("providers_executed", []) or [])
    expected_providers = set(results.get("providers_requested", []) or [])
    if "release" in str(results.get("execution_mode", "") or ""):
        gates.append(
            {
                "name": "required_provider_tracks",
                "passed": providers_executed == expected_providers,
                "value": len(providers_executed),
                "threshold": len(expected_providers),
            }
        )
        gates.append(
            {
                "name": "scorecard_labels_present",
                "passed": bool(str(results.get("scorecard_mode", "") or "").strip()),
                "value": 1 if bool(str(results.get("scorecard_mode", "") or "").strip()) else 0,
                "threshold": 1,
            }
        )
    for provider, provider_payload in (results.get("providers", {}) or {}).items():
        byte_phase = _preferred_phase(provider_payload, "byte")
        direct_phase = _preferred_phase(provider_payload, "direct")
        if not byte_phase:
            continue
        summary = dict(byte_phase.get("summary", {}) or {})
        gates.extend(
            [
                _gate(f"{provider}_false_reuse_rate", summary.get("false_reuse_rate", 0.0), 0.01, less_than=True),
                _gate(f"{provider}_fallback_trigger_rate", summary.get("fallback_trigger_rate", 0.0), 0.95),
                _gate(f"{provider}_confidence_score_accuracy", summary.get("confidence_score_accuracy", 0.0), 0.90),
                _gate(f"{provider}_confidence_ece", summary.get("confidence_ece", 0.0), 0.08, less_than=True),
                _gate(f"{provider}_deterministic_output_rate", summary.get("deterministic_output_rate", 0.0), 0.98),
                {
                    "name": f"{provider}_coverage_and_ci_present",
                    "passed": bool(
                        summary.get("sample_size", 0)
                        and "coverage" in summary
                        and "ci_95" in summary
                    ),
                    "value": 1 if bool(
                        summary.get("sample_size", 0)
                        and "coverage" in summary
                        and "ci_95" in summary
                    ) else 0,
                    "threshold": 1,
                },
                {
                    "name": f"{provider}_contamination_status_present",
                    "passed": bool(str(summary.get("contamination_status", "") or "").strip()),
                    "value": 1 if bool(str(summary.get("contamination_status", "") or "").strip()) else 0,
                    "threshold": 1,
                },
            ]
        )
        if float(summary.get("forced_answer_accuracy", 0.0) or 0.0) >= 1.0:
            gates.append(
                {
                    "name": f"{provider}_perfect_accuracy_labeled",
                    "passed": bool(summary.get("coverage") is not None and summary.get("ci_95")),
                    "value": 1 if bool(summary.get("coverage") is not None and summary.get("ci_95")) else 0,
                    "threshold": 1,
                }
            )
        divergence = abs(
            float(summary.get("forced_answer_accuracy", 0.0) or 0.0)
            - float(summary.get("selective_accuracy", 0.0) or 0.0)
        )
        gates.append(
            {
                "name": f"{provider}_scorecard_divergence_explained",
                "passed": divergence < 0.0001 or bool(str(summary.get("scorecard_explanation", "") or "").strip()),
                "value": round(divergence, 4),
                "threshold": 0.0,
            }
        )
        if str(results.get("profile", "") or "").startswith("prompt_distillation"):
            gates.extend(
                [
                    _gate(
                        f"{provider}_prompt_token_reduction_ratio",
                        summary.get("prompt_token_reduction_ratio", 0.0),
                        0.40,
                    ),
                    _gate(
                        f"{provider}_entity_preservation_rate",
                        summary.get("entity_preservation_rate", 0.0),
                        0.995,
                    ),
                    _gate(
                        f"{provider}_schema_preservation_rate",
                        summary.get("schema_preservation_rate", 0.0),
                        0.995,
                    ),
                ]
            )
        for family_name, family_payload in (byte_phase.get("families", {}) or {}).items():
            family_summary = dict(family_payload.get("summary", {}) or {})
            if family_name in {"wrong_reuse_detection", "degradation_unseen", "real_world_chaos"}:
                gates.append(
                    _gate(
                        f"{provider}_{family_name}_false_reuse_rate",
                        family_summary.get("false_reuse_rate", 0.0),
                        0.0025 if family_name in {"wrong_reuse_detection", "real_world_chaos"} else 0.01,
                        less_than=True,
                    )
                )
            if str(results.get("profile", "") or "").startswith("prompt_distillation") and family_name in {
                "long_context_retrieval",
                "policy_bloat",
                "codebase_context",
                "compression_faithfulness",
                "selective_augmentation",
                "distillation_injection_resilience",
            }:
                gates.extend(
                    [
                        _gate(
                            f"{provider}_{family_name}_prompt_token_reduction_ratio",
                            family_summary.get("prompt_token_reduction_ratio", 0.0),
                            0.40,
                        ),
                        _gate(
                            f"{provider}_{family_name}_entity_preservation_rate",
                            family_summary.get("entity_preservation_rate", 0.0),
                            0.995,
                        ),
                        _gate(
                            f"{provider}_{family_name}_schema_preservation_rate",
                            family_summary.get("schema_preservation_rate", 0.0),
                            0.995,
                        ),
                    ]
                )
        if direct_phase:
            for family_name, byte_family in (byte_phase.get("families", {}) or {}).items():
                direct_family = (direct_phase.get("families", {}) or {}).get(family_name)
                if not direct_family:
                    continue
                delta = float(
                    (byte_family.get("comparison_to_direct", {}) or {}).get("accuracy_delta", 0.0) or 0.0
                )
                threshold = 0.0 if family_name in {"wrong_reuse_detection", "degradation_unseen"} else -0.01
                gates.append(
                    {
                        "name": f"{provider}_{family_name}_accuracy_delta",
                        "passed": delta >= threshold,
                        "value": round(delta, 4),
                        "threshold": threshold,
                    }
                )
    return gates


def _gate(name: str, value: float, threshold: float, *, less_than: bool = False) -> dict[str, Any]:
    numeric_value = float(value or 0.0)
    passed = numeric_value <= threshold if less_than else numeric_value >= threshold
    return {
        "name": name,
        "passed": passed,
        "value": round(numeric_value, 4),
        "threshold": threshold,
    }


def _dedupe_lists(results: dict[str, Any]) -> None:
    results["providers_executed"] = sorted(set(results.get("providers_executed", []) or []))
    results["systems_executed"] = sorted(set(results.get("systems_executed", []) or []))


def _preferred_phase(provider_payload: dict[str, Any], system_name: str) -> dict[str, Any] | None:
    phases = (
        ((provider_payload.get("systems", {}) or {}).get(system_name, {}) or {}).get("phases", {})
        or {}
    )
    for candidate in (RunPhase.WARM_100.value, RunPhase.WARM_1000.value, RunPhase.COLD.value):
        if candidate in phases:
            return phases[candidate]
    return None


def _release_track(profile_data: dict[str, Any], max_items_per_family: int | None) -> str:
    if max_items_per_family is not None:
        return "partial"
    if str(profile_data.get("benchmark_track", "") or "") == "provider_local":
        return "provider_local_release"
    return "platform_global_release"
