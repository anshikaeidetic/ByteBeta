"""Thin entrypoint for the multi-provider routing and memory benchmark."""

from __future__ import annotations

from typing import Any

from byte.benchmarking._program_entry import (
    program_dir,
    proxy_program_attribute,
    run_program,
)
from byte.benchmarking._program_models import BenchmarkProgramSpec

PROGRAM_SPEC = BenchmarkProgramSpec(
    name="deep_multi_provider_routing_memory",
    implementation_module="byte.benchmarking._program_impl.deep_multi_provider_routing_memory",
    description="Multi-provider routing and memory benchmark.",
)


def main() -> int:
    """Run the benchmark CLI."""

    return run_program(PROGRAM_SPEC)


def __getattr__(name: str) -> Any:
    return proxy_program_attribute(PROGRAM_SPEC, name)


def __dir__() -> list[str]:
    return program_dir(PROGRAM_SPEC)


__all__ = ["PROGRAM_SPEC", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
