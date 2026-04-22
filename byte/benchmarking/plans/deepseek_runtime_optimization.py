"""Package entrypoint for the DeepSeek runtime optimization workload plan."""

from byte.benchmarking.programs import deep_deepseek_runtime_optimization_benchmark as _script

build_workload_plan = _script.build_workload_plan

__all__ = ["build_workload_plan"]
