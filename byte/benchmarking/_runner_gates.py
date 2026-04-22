"""Release-gate evaluation helpers for benchmark reports."""

from __future__ import annotations

from typing import Any

from byte.benchmarking._runner_reporting import preferred_phase


def evaluate_gates(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate release and trust gates for the completed benchmark bundle."""
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
        byte_phase = preferred_phase(provider_payload, "byte")
        direct_phase = preferred_phase(provider_payload, "direct")
        if not byte_phase:
            continue
        summary = dict(byte_phase.get("summary", {}) or {})
        gates.extend(
            [
                gate(f"{provider}_false_reuse_rate", summary.get("false_reuse_rate", 0.0), 0.01, less_than=True),
                gate(f"{provider}_fallback_trigger_rate", summary.get("fallback_trigger_rate", 0.0), 0.95),
                gate(f"{provider}_confidence_score_accuracy", summary.get("confidence_score_accuracy", 0.0), 0.90),
                gate(f"{provider}_confidence_ece", summary.get("confidence_ece", 0.0), 0.08, less_than=True),
                gate(f"{provider}_deterministic_output_rate", summary.get("deterministic_output_rate", 0.0), 0.98),
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
                    gate(
                        f"{provider}_prompt_token_reduction_ratio",
                        summary.get("prompt_token_reduction_ratio", 0.0),
                        0.40,
                    ),
                    gate(
                        f"{provider}_entity_preservation_rate",
                        summary.get("entity_preservation_rate", 0.0),
                        0.995,
                    ),
                    gate(
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
                    gate(
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
                        gate(
                            f"{provider}_{family_name}_prompt_token_reduction_ratio",
                            family_summary.get("prompt_token_reduction_ratio", 0.0),
                            0.40,
                        ),
                        gate(
                            f"{provider}_{family_name}_entity_preservation_rate",
                            family_summary.get("entity_preservation_rate", 0.0),
                            0.995,
                        ),
                        gate(
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


def gate(name: str, value: float, threshold: float, *, less_than: bool = False) -> dict[str, Any]:
    """Create a normalized gate result payload."""
    numeric_value = float(value or 0.0)
    passed = numeric_value <= threshold if less_than else numeric_value >= threshold
    return {
        "name": name,
        "passed": passed,
        "value": round(numeric_value, 4),
        "threshold": threshold,
    }


__all__ = ["evaluate_gates", "gate"]
