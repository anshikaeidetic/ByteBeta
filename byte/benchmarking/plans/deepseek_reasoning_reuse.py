"""Package entrypoint for the DeepSeek reasoning reuse workload plan."""

from byte.benchmarking.programs import deep_deepseek_reasoning_reuse_benchmark as _script

build_workload_plan = _script.build_workload_plan

__all__ = ["build_workload_plan"]
