"""Shared workload manifest helpers for benchmark workload generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from byte.benchmarking.contracts import BenchmarkItem
from byte.benchmarking.integrity import BENCHMARK_CONTRACT_VERSION

SYSTEM_PROMPT = "Follow the output contract exactly. Return only the final answer."
PROVIDERS = ("openai", "anthropic", "deepseek")
WORKLOAD_DIR = Path(__file__).resolve().parent / "workloads"
WORKLOAD_GENERATOR_VERSION = "byte-benchmarking-v2"
FAMILY_LANES = {
    "real_world_chaos": "objective_release",
    "wrong_reuse_detection": "objective_release",
    "fuzzy_similarity": "objective_release",
    "generalization": "objective_release",
    "long_horizon_agents": "objective_release",
    "degradation_unseen": "objective_release",
    "prompt_module_reuse": "objective_release",
    "long_context_retrieval": "objective_release",
    "policy_bloat": "objective_release",
    "codebase_context": "objective_release",
    "compression_faithfulness": "objective_release",
    "selective_augmentation": "objective_release",
    "distillation_injection_resilience": "objective_release",
}
FAMILY_CONTAMINATION_STATUS = {
    "real_world_chaos": "controlled_synthetic",
    "wrong_reuse_detection": "controlled_synthetic",
    "fuzzy_similarity": "controlled_paraphrase_holdout",
    "generalization": "controlled_grounded_holdout",
    "long_horizon_agents": "controlled_workflow_holdout",
    "degradation_unseen": "synthetic_unseen_holdout",
    "prompt_module_reuse": "controlled_prompt_module_holdout",
    "long_context_retrieval": "controlled_retrieval_holdout",
    "policy_bloat": "controlled_policy_holdout",
    "codebase_context": "controlled_codebase_holdout",
    "compression_faithfulness": "controlled_faithfulness_holdout",
    "selective_augmentation": "controlled_augmentation_holdout",
    "distillation_injection_resilience": "controlled_injection_holdout",
}
FAMILY_REFERENCE_SET = dict.fromkeys(FAMILY_LANES, "reference_holdout")


def payload(
    prompt: str,
    *,
    context_payload: dict[str, Any] | None = None,
    max_tokens: int = 16,
) -> dict[str, Any]:
    """Build the common chat payload used by generated workload items."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "context_payload": dict(context_payload or {}),
        "max_tokens": max_tokens,
    }


def for_all_providers(item: BenchmarkItem) -> list[BenchmarkItem]:
    """Expand one canonical workload item into provider-specific copies."""
    outputs = []
    for provider in PROVIDERS:
        model_hint = (
            "deepseek-chat"
            if provider == "deepseek"
            else "gpt-4o-mini"
            if provider == "openai"
            else "claude-3-5-sonnet-latest"
        )
        outputs.append(
            BenchmarkItem(
                item_id=item.item_id,
                provider_track=provider,
                family=item.family,
                scenario=item.scenario,
                seed_id=item.seed_id,
                variant_id=item.variant_id,
                input_payload=dict(item.input_payload),
                output_contract=item.output_contract,
                expected_value=item.expected_value,
                tolerance=item.tolerance,
                reuse_safe=item.reuse_safe,
                must_fallback=item.must_fallback,
                tags=tuple(item.tags),
                deterministic_expected=item.deterministic_expected,
                workflow_total_steps=item.workflow_total_steps,
                model_hint=model_hint,
                metadata=contract_metadata(**dict(item.metadata)),
            )
        )
    return outputs


def contract_metadata(recipe: str = "", **payload_fields: Any) -> dict[str, Any]:
    """Attach versioned contract metadata to a generated item."""
    metadata = dict(payload_fields)
    metadata["contract_recipe"] = str(recipe or metadata.get("contract_recipe", "") or "")
    metadata["contract_version"] = BENCHMARK_CONTRACT_VERSION
    return metadata


def format_percentage(value: float) -> str:
    """Render numeric targets in a stable compact percentage format."""
    rounded = round(float(value or 0.0), 2)
    if abs(rounded - round(rounded)) < 0.005:
        return f"{int(round(rounded))}%"
    return f"{rounded:.2f}".rstrip("0").rstrip(".") + "%"


def grounded_context(seed: int, expected: str) -> dict[str, Any]:
    """Build shared grounded context blocks for retrieval-style tasks."""
    return {
        "byte_document_context": (
            f"invoice identifier is INV-{seed + 1000:04d}. "
            f"queue identifier is queue-{seed:02d}-dispatch. "
            f"policy label is POLICY_{seed:02d}. "
            f"owner label is TEAM_{seed:02d}. "
            f"follow-up due date is 2026-04-{10 + (seed % 18):02d}. "
            f"prescribed action label is {expected}."
        ),
        "byte_repo_summary": {
            "services": [f"svc-{seed:02d}", f"svc-{seed + 1:02d}"],
            "queue": f"queue-{seed:02d}-dispatch",
            "policy_label": f"POLICY_{seed:02d}",
        },
    }


def long_policy_block(seed: int, expected: str) -> str:
    """Build large policy-context blocks for prompt-bloat scenarios."""
    return (
        f"Byte policy scaffold {seed:02d}. "
        f"prescribed policy label is {expected}. "
        + " ".join(
            f"Rule {index}: preserve billing, shipping, support, and ledger controls."
            for index in range(1, 15)
        )
    )


def long_schema_block(seed: int) -> str:
    """Build schema-heavy prompt blocks for contract-preservation tests."""
    return (
        f"Schema block {seed:02d}. "
        + " ".join(
            f"Field field_{index}: string; validation required; emit stable JSON when asked."
            for index in range(1, 18)
        )
    )


def long_tool_block(seed: int) -> str:
    """Build tool-catalog context for workflow-heavy reuse tests."""
    return (
        f"Tool catalog {seed:02d}. "
        + " ".join(
            f"tool_{index} accepts queue, owner, invoice_id, and audit_trail arguments."
            for index in range(1, 16)
        )
    )


def noise_paragraph(seed: int, label: str) -> str:
    """Build deterministic filler text for augmentation and noise tests."""
    return " ".join(
        f"{label} note {seed:02d}-{index:02d}: retain audit metadata, replicate backlog status, and record observer context."
        for index in range(1, 14)
    )


def long_policy_catalog(seed: int) -> str:
    """Build a larger catalog of policy noise for long-prompt scenarios."""
    return " ".join(
        f"Policy section {seed:02d}-{index:02d}: approval windows differ by channel, geography, queue, and ledger class."
        for index in range(1, 20)
    )


__all__ = [
    "FAMILY_CONTAMINATION_STATUS",
    "FAMILY_LANES",
    "FAMILY_REFERENCE_SET",
    "PROVIDERS",
    "WORKLOAD_DIR",
    "WORKLOAD_GENERATOR_VERSION",
    "contract_metadata",
    "for_all_providers",
    "format_percentage",
    "grounded_context",
    "long_policy_block",
    "long_policy_catalog",
    "long_schema_block",
    "long_tool_block",
    "noise_paragraph",
    "payload",
]
