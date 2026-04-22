from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from byte.benchmarking.contracts import BenchmarkItem
from byte.benchmarking.integrity import (
    BENCHMARK_CONTRACT_VERSION,
    BENCHMARK_CORPUS_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    BENCHMARK_SCORING_VERSION,
    validate_item_contract,
)
from byte.benchmarking.workload_generator import (
    FAMILY_CONTAMINATION_STATUS,
    FAMILY_LANES,
    WORKLOAD_GENERATOR_VERSION,
)

WORKLOAD_DIR = Path(__file__).resolve().parent / "workloads"
PROFILE_FAMILIES = {
    "tier1": [
        "real_world_chaos",
        "wrong_reuse_detection",
        "fuzzy_similarity",
        "generalization",
        "long_horizon_agents",
        "degradation_unseen",
    ],
    "prompt_distillation": [
        "prompt_module_reuse",
        "long_context_retrieval",
        "policy_bloat",
        "codebase_context",
        "compression_faithfulness",
        "selective_augmentation",
        "distillation_injection_resilience",
    ],
    "tier1_v2_deepseek": [
        "real_world_chaos",
        "wrong_reuse_detection",
        "fuzzy_similarity",
        "generalization",
        "long_horizon_agents",
        "degradation_unseen",
    ],
    "prompt_distillation_v2_deepseek": [
        "prompt_module_reuse",
        "long_context_retrieval",
        "policy_bloat",
        "codebase_context",
        "compression_faithfulness",
        "selective_augmentation",
        "distillation_injection_resilience",
    ],
}
FAMILY_ORDER = PROFILE_FAMILIES["tier1"]
FAMILY_SIZES = {
    "real_world_chaos": 400,
    "wrong_reuse_detection": 320,
    "fuzzy_similarity": 300,
    "generalization": 240,
    "long_horizon_agents": 120,
    "degradation_unseen": 200,
    "prompt_module_reuse": 80,
    "long_context_retrieval": 80,
    "policy_bloat": 80,
    "codebase_context": 80,
    "compression_faithfulness": 80,
    "selective_augmentation": 80,
    "distillation_injection_resilience": 80,
}
PROFILE_PROVIDERS = {
    "tier1": ("openai", "anthropic", "deepseek"),
    "prompt_distillation": ("openai", "anthropic", "deepseek"),
    "tier1_v2_deepseek": ("deepseek",),
    "prompt_distillation_v2_deepseek": ("deepseek",),
}
PROFILE_BASE = {
    "tier1": "tier1",
    "prompt_distillation": "prompt_distillation",
    "tier1_v2_deepseek": "tier1",
    "prompt_distillation_v2_deepseek": "prompt_distillation",
}
PROFILE_TRACKS = {
    "tier1": "platform_global",
    "prompt_distillation": "platform_global",
    "tier1_v2_deepseek": "provider_local",
    "prompt_distillation_v2_deepseek": "provider_local",
}


def manifest_paths(profile: str = "tier1") -> list[Path]:
    if profile not in PROFILE_FAMILIES:
        raise ValueError(f"Unsupported benchmark profile: {profile}")
    base = PROFILE_BASE[profile]
    return [WORKLOAD_DIR / f"{family}.json" for family in PROFILE_FAMILIES[base]]


def load_profile(
    profile: str = "tier1",
    *,
    providers: list[str] | None = None,
    families: list[str] | None = None,
    max_items_per_family: int | None = None,
) -> dict[str, Any]:
    selected_providers = tuple(providers or PROFILE_PROVIDERS[profile])
    if profile not in PROFILE_FAMILIES:
        raise ValueError(f"Unsupported benchmark profile: {profile}")
    base_profile = PROFILE_BASE[profile]
    profile_families = PROFILE_FAMILIES[profile]
    selected_families = set(families or profile_families)
    manifests = []
    family_metadata: dict[str, dict[str, Any]] = {}
    items: list[BenchmarkItem] = []
    for path in manifest_paths(profile):
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema_version = str(payload.get("schema_version", "") or "")
        corpus_version = str(payload.get("corpus_version", "") or "")
        contract_version = str(payload.get("contract_version", "") or "")
        scoring_version = str(payload.get("scoring_version", "") or "")
        generator_version = str(payload.get("generator_version", "") or "")
        if schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema_version {schema_version or '<missing>'}; "
                f"expected {BENCHMARK_SCHEMA_VERSION}."
            )
        if corpus_version != BENCHMARK_CORPUS_VERSION:
            raise ValueError(
                f"{path} has corpus_version {corpus_version or '<missing>'}; "
                f"expected {BENCHMARK_CORPUS_VERSION}."
            )
        if contract_version != BENCHMARK_CONTRACT_VERSION:
            raise ValueError(
                f"{path} has contract_version {contract_version or '<missing>'}; "
                f"expected {BENCHMARK_CONTRACT_VERSION}."
            )
        if scoring_version != BENCHMARK_SCORING_VERSION:
            raise ValueError(
                f"{path} has scoring_version {scoring_version or '<missing>'}; "
                f"expected {BENCHMARK_SCORING_VERSION}."
            )
        if generator_version != WORKLOAD_GENERATOR_VERSION:
            raise ValueError(
                f"{path} has generator_version {generator_version or '<missing>'}; "
                f"expected {WORKLOAD_GENERATOR_VERSION}."
            )
        manifests.append(
            {
                "path": str(path),
                "family": payload["family"],
                "profile": payload["profile"],
                "schema_version": schema_version,
                "corpus_version": corpus_version,
                "contract_version": contract_version,
                "scoring_version": scoring_version,
                "generator_version": generator_version,
                "family_lane": str(payload.get("family_lane", "") or FAMILY_LANES.get(payload["family"], "")),
                "contamination_status": str(
                    payload.get("contamination_status", "")
                    or FAMILY_CONTAMINATION_STATUS.get(payload["family"], "")
                ),
                "generated_at": str(payload.get("generated_at", "") or ""),
                "live_cutoff_date": payload.get("live_cutoff_date"),
            }
        )
        family = str(payload["family"])
        family_metadata[family] = {
            "family_lane": str(payload.get("family_lane", "") or FAMILY_LANES.get(family, "")),
            "contamination_status": str(
                payload.get("contamination_status", "")
                or FAMILY_CONTAMINATION_STATUS.get(family, "")
            ),
            "reference_set": str(payload.get("reference_set", "") or ""),
            "generated_at": str(payload.get("generated_at", "") or ""),
            "live_cutoff_date": payload.get("live_cutoff_date"),
        }
        if family not in selected_families:
            continue
        family_items = [
            BenchmarkItem.from_manifest(item)
            for item in (payload.get("items") or [])
            if str(item.get("provider_track")) in selected_providers
        ]
        if max_items_per_family is not None:
            sliced: list[BenchmarkItem] = []
            counts: dict[str, int] = dict.fromkeys(selected_providers, 0)
            for item in family_items:
                track = item.provider_track
                if counts.get(track, 0) >= int(max_items_per_family):
                    continue
                sliced.append(item)
                counts[track] = counts.get(track, 0) + 1
            family_items = sliced
        items.extend(family_items)
    return {
        "profile": profile,
        "profile_base": base_profile,
        "benchmark_track": PROFILE_TRACKS.get(profile, "platform_global"),
        "providers": list(selected_providers),
        "families": [family for family in profile_families if family in selected_families],
        "manifests": manifests,
        "family_metadata": family_metadata,
        "corpus_version": BENCHMARK_CORPUS_VERSION,
        "items": items,
    }


def validate_profile(profile_data: dict[str, Any]) -> dict[str, Any]:
    items: list[BenchmarkItem] = list(profile_data.get("items") or [])
    providers = list(profile_data.get("providers") or [])
    family_counts: dict[str, dict[str, int]] = {}
    label_counts: dict[str, dict[str, int]] = {}
    contract_versions: set[str] = set()
    contract_recipes: dict[str, set[str]] = {}
    contamination_statuses: dict[str, str] = {}
    family_lanes: dict[str, str] = {}
    profile_key = str(profile_data.get("profile", "tier1"))
    for family in PROFILE_FAMILIES.get(profile_key, FAMILY_ORDER):
        family_counts[family] = dict.fromkeys(providers, 0)
        label_counts[family] = {"reuse_safe": 0, "must_fallback": 0}
        contract_recipes[family] = set()
        family_meta = dict((profile_data.get("family_metadata", {}) or {}).get(family, {}) or {})
        contamination_statuses[family] = str(
            family_meta.get("contamination_status", "")
            or FAMILY_CONTAMINATION_STATUS.get(family, "")
        )
        family_lanes[family] = str(family_meta.get("family_lane", "") or FAMILY_LANES.get(family, ""))
    for item in items:
        family_counts[item.family][item.provider_track] += 1
        if item.reuse_safe:
            label_counts[item.family]["reuse_safe"] += 1
        if item.must_fallback:
            label_counts[item.family]["must_fallback"] += 1
    for item in items:
        if not item.item_id:
            raise ValueError("Benchmark item id is required.")
        if not item.seed_id:
            raise ValueError(f"Missing seed_id for {item.item_id}.")
        if not item.variant_id:
            raise ValueError(f"Missing variant_id for {item.item_id}.")
        if not item.input_payload.get("messages"):
            raise ValueError(f"Missing messages for {item.item_id}.")
        integrity = validate_item_contract(item)
        contract_versions.add(str(integrity["contract_version"]))
        contract_recipes[item.family].add(str(integrity["contract_recipe"]))
    return {
        "family_counts": family_counts,
        "label_counts": label_counts,
        "contract_versions": sorted(contract_versions),
        "contract_recipes": {key: sorted(value) for key, value in contract_recipes.items()},
        "contamination_statuses": contamination_statuses,
        "family_lanes": family_lanes,
    }
