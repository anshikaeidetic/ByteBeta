"""Canonical benchmark workload-plan entrypoints."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PLAN_EXPORTS = {
    "build_deepseek_reasoning_reuse_plan": (
        "byte.benchmarking.plans.deepseek_reasoning_reuse",
        "build_workload_plan",
    ),
    "build_deepseek_runtime_plan": (
        "byte.benchmarking.plans.deepseek_runtime_optimization",
        "build_workload_plan",
    ),
    "build_openai_mixed_100_plan": (
        "byte.benchmarking.plans.openai_mixed_100",
        "build_workload_plan",
    ),
    "build_openai_mixed_1000_plan": (
        "byte.benchmarking.plans.openai_mixed_1000",
        "build_workload_plan",
    ),
    "openai_mixed_100_provider_coverage": (
        "byte.benchmarking.plans.openai_mixed_100",
        "provider_coverage",
    ),
    "openai_mixed_1000_provider_coverage": (
        "byte.benchmarking.plans.openai_mixed_1000",
        "provider_coverage",
    ),
}

__all__ = list(_PLAN_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _PLAN_EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _PLAN_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
