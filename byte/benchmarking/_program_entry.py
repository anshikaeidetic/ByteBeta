"""Lazy dispatch helpers for thin benchmark program entry modules."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from byte.benchmarking._program_models import BenchmarkProgramSpec


def load_program_module(spec: BenchmarkProgramSpec) -> ModuleType:
    """Import the private implementation module for a benchmark program."""

    return importlib.import_module(spec.implementation_module)


def run_program(spec: BenchmarkProgramSpec) -> int:
    """Run a benchmark program's private ``main`` function."""

    main = load_program_module(spec).main
    return int(main())


def proxy_program_attribute(spec: BenchmarkProgramSpec, name: str) -> Any:
    """Resolve compatibility attributes from the private implementation module."""

    if name == "PROGRAM_SPEC":
        return spec
    return getattr(load_program_module(spec), name)


def program_dir(spec: BenchmarkProgramSpec, public_names: tuple[str, ...] = ()) -> list[str]:
    """Return public wrapper names plus the implementation module names."""

    names = {"PROGRAM_SPEC", "main", *public_names}
    names.update(dir(load_program_module(spec)))
    return sorted(names)


__all__ = [
    "load_program_module",
    "program_dir",
    "proxy_program_attribute",
    "run_program",
]
