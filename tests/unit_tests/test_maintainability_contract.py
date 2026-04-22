from __future__ import annotations

from pathlib import Path

from byte._devtools import check_maintainability_contract as contract
from byte._devtools.verification_targets import (
    BENCHMARK_ARCHITECTURE_TARGETS,
    MAINTAINABILITY_TARGETS,
    STRICT_TYPECHECK_TARGETS,
)


def _write_module(root: Path, rel_path: str, source: str) -> Path:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_analyze_module_flags_missing_docstrings_and_annotations(tmp_path) -> None:
    path = _write_module(
        tmp_path,
        "pkg/service.py",
        "def public_helper(value):\n    return value\n",
    )

    metrics = contract.analyze_module(
        path,
        root=tmp_path,
        target_paths={"pkg/service.py"},
    )

    assert metrics.target_issues == ["pkg/service.py: missing module docstring"]
    assert metrics.function_issues[0].name == "public_helper"
    assert metrics.function_issues[0].reason == "missing complete annotations"


def test_analyze_module_accepts_documented_boundary_catches(tmp_path) -> None:
    path = _write_module(
        tmp_path,
        "pkg/provider.py",
        '"""Provider boundary."""\n\n'
        "def call_provider(value: int) -> int:\n"
        '    """Call provider and preserve provider failures."""\n'
        "    try:\n"
        "        return value + 1\n"
        "    except Exception as exc:  # provider boundary\n"
        "        raise RuntimeError('provider failed') from exc\n",
    )

    metrics = contract.analyze_module(
        path,
        root=tmp_path,
        target_paths={"pkg/provider.py"},
    )

    assert metrics.broad_exceptions == 1
    assert metrics.justified_broad_exceptions == 1
    assert not metrics.broad_exception_issues


def test_analyze_module_rejects_unclassified_broad_catches(tmp_path) -> None:
    path = _write_module(
        tmp_path,
        "pkg/service.py",
        '"""Service."""\n\n'
        "def parse(value: str) -> str:\n"
        '    """Parse a value."""\n'
        "    try:\n"
        "        return value.strip()\n"
        "    except Exception:\n"
        "        return ''\n",
    )

    metrics = contract.analyze_module(
        path,
        root=tmp_path,
        target_paths={"pkg/service.py"},
    )

    assert metrics.broad_exception_issues
    assert metrics.broad_exception_issues[0].line == 7


def test_expand_target_paths_expands_directories(tmp_path) -> None:
    first = _write_module(tmp_path, "pkg/a.py", '"""A."""\n')
    second = _write_module(tmp_path, "pkg/sub/b.py", '"""B."""\n')

    assert contract.expand_target_paths(tmp_path, ["pkg"]) == (first, second)


def test_contract_failures_enforce_target_issues() -> None:
    metrics = contract.ModuleMetrics(
        path="byte/example.py",
        category="strict_target",
        module_docstring=False,
        functions=1,
        fully_annotated_functions=1,
        target_issues=["byte/example.py: missing module docstring"],
    )

    failures = contract.contract_failures([metrics])

    assert "byte/example.py: missing module docstring" in failures


def test_maintainability_targets_include_strict_and_benchmark_surfaces() -> None:
    assert set(STRICT_TYPECHECK_TARGETS).issubset(MAINTAINABILITY_TARGETS)
    assert set(BENCHMARK_ARCHITECTURE_TARGETS).issubset(MAINTAINABILITY_TARGETS)
    assert "byte/_devtools/check_maintainability_contract.py" in MAINTAINABILITY_TARGETS
    assert "scripts/check_maintainability_contract.py" in MAINTAINABILITY_TARGETS
