"""Thin entrypoint for the DeepSeek runtime optimization benchmark."""

from __future__ import annotations

from typing import Any

from byte.benchmarking._program_entry import (
    program_dir,
    proxy_program_attribute,
    run_program,
)
from byte.benchmarking._program_models import BenchmarkProgramSpec

PROGRAM_SPEC = BenchmarkProgramSpec(
    name="deep_deepseek_runtime_optimization_benchmark",
    implementation_module=(
        "byte.benchmarking._program_impl.deep_deepseek_runtime_optimization_benchmark"
    ),
    description="DeepSeek runtime optimization and Byte reuse benchmark.",
)


def main() -> int:
    """Run the benchmark CLI."""

    return run_program(PROGRAM_SPEC)


def __getattr__(name: str) -> Any:
    return proxy_program_attribute(PROGRAM_SPEC, name)


def __dir__() -> list[str]:
    return program_dir(PROGRAM_SPEC, ("build_workload_plan", "run_benchmark"))


__all__ = ["PROGRAM_SPEC", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
