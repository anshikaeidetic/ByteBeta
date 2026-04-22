from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

WRAPPER_MAP = {
    "scripts/advanced_openai_cost_patterns.py": "byte.benchmarking.programs.advanced_openai_cost_patterns",
    "scripts/deep_deepseek_reasoning_reuse_benchmark.py": "byte.benchmarking.programs.deep_deepseek_reasoning_reuse_benchmark",
    "scripts/deep_deepseek_runtime_optimization_benchmark.py": "byte.benchmarking.programs.deep_deepseek_runtime_optimization_benchmark",
    "scripts/deep_multi_provider_routing_memory.py": "byte.benchmarking.programs.deep_multi_provider_routing_memory",
    "scripts/deep_openai_1000_request_mixed_benchmark.py": "byte.benchmarking.programs.deep_openai_1000_request_mixed_benchmark",
    "scripts/deep_openai_100_request_mixed_benchmark.py": "byte.benchmarking.programs.deep_openai_100_request_mixed_benchmark",
    "scripts/deep_openai_coding_benchmark.py": "byte.benchmarking.programs.deep_openai_coding_benchmark",
    "scripts/deep_openai_comprehensive_workload_benchmark.py": "byte.benchmarking.programs.deep_openai_comprehensive_workload_benchmark",
    "scripts/deep_openai_cost_levers.py": "byte.benchmarking.programs.deep_openai_cost_levers",
    "scripts/deep_openai_prompt_stress_benchmark.py": "byte.benchmarking.programs.deep_openai_prompt_stress_benchmark",
    "scripts/deep_openai_surface_benchmark.py": "byte.benchmarking.programs.deep_openai_surface_benchmark",
    "scripts/deep_openai_unified_router_benchmark.py": "byte.benchmarking.programs.deep_openai_unified_router_benchmark",
}


def test_benchmark_scripts_are_thin_package_wrappers() -> None:
    for relative_path, module_name in WRAPPER_MAP.items():
        contents = (ROOT / relative_path).read_text(encoding="utf-8")
        assert f"from {module_name} import main" in contents
        assert len(contents.splitlines()) <= 7
