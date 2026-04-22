"""Lazy plan-loader helpers for benchmark workload plan exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PLAN_MODULES = {
    "build_deepseek_reasoning_reuse_plan": "byte.benchmarking.plans.deepseek_reasoning_reuse",
    "build_deepseek_runtime_plan": "byte.benchmarking.plans.deepseek_runtime_optimization",
    "build_openai_mixed_100_plan": "byte.benchmarking.plans.openai_mixed_100",
    "openai_mixed_100_provider_coverage": "byte.benchmarking.plans.openai_mixed_100",
    "build_openai_mixed_1000_plan": "byte.benchmarking.plans.openai_mixed_1000",
    "openai_mixed_1000_provider_coverage": "byte.benchmarking.plans.openai_mixed_1000",
}
_PLAN_EXPORTS = {
    "build_deepseek_reasoning_reuse_plan": "build_workload_plan",
    "build_deepseek_runtime_plan": "build_workload_plan",
    "build_openai_mixed_100_plan": "build_workload_plan",
    "openai_mixed_100_provider_coverage": "provider_coverage",
    "build_openai_mixed_1000_plan": "build_workload_plan",
    "openai_mixed_1000_provider_coverage": "provider_coverage",
}


def load_plan_export(name: str) -> Any:
    """Resolve a plan export lazily to avoid optional imports at package import time."""
    module_name = _PLAN_MODULES[name]
    export_name = _PLAN_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, export_name)


__all__ = ["load_plan_export"]
