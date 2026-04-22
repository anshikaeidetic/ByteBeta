"""Registry for deterministic benchmark workload family builders."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import import_module

from byte.benchmarking.contracts import BenchmarkItem

FamilyBuilder = Callable[[], list[BenchmarkItem]]

FAMILY_MODULES: tuple[tuple[str, str], ...] = (
    ("real_world_chaos", "byte.benchmarking.workload_families.real_world_chaos"),
    ("wrong_reuse_detection", "byte.benchmarking.workload_families.wrong_reuse_detection"),
    ("fuzzy_similarity", "byte.benchmarking.workload_families.fuzzy_similarity"),
    ("generalization", "byte.benchmarking.workload_families.generalization"),
    ("long_horizon_agents", "byte.benchmarking.workload_families.long_horizon_agents"),
    ("degradation_unseen", "byte.benchmarking.workload_families.degradation_unseen"),
    ("prompt_module_reuse", "byte.benchmarking.workload_families.prompt_module_reuse"),
    ("long_context_retrieval", "byte.benchmarking.workload_families.long_context_retrieval"),
    ("policy_bloat", "byte.benchmarking.workload_families.policy_bloat"),
    ("codebase_context", "byte.benchmarking.workload_families.codebase_context"),
    ("compression_faithfulness", "byte.benchmarking.workload_families.compression_faithfulness"),
    ("selective_augmentation", "byte.benchmarking.workload_families.selective_augmentation"),
    (
        "distillation_injection_resilience",
        "byte.benchmarking.workload_families.distillation_injection_resilience",
    ),
)


def iter_family_builders() -> Iterator[tuple[str, FamilyBuilder]]:
    """Yield benchmark workload family builders in report-stable order."""

    for family_name, module_name in FAMILY_MODULES:
        module = import_module(module_name)
        yield family_name, module.build


__all__ = ["FAMILY_MODULES", "FamilyBuilder", "iter_family_builders"]
