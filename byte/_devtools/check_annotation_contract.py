"""Validate return-annotation coverage for Byte production code."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script

sys.dont_write_bytecode = True

PRODUCTION_ROOTS = ("byte", "byte_server", "byte_inference", "byte_memory")
SCRIPT_ROOT = "scripts"
BLOCKING_CATEGORIES = frozenset({"production", "scripts", "tests", "other"})
REPORT_ONLY_CATEGORIES: frozenset[str] = frozenset()
EXCLUDED_PATH_PARTS = {
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".venv-base-check",
    "__pycache__",
    "build",
    "dist",
}


@dataclass(frozen=True)
class MissingReturnAnnotation:
    """A function definition missing an explicit return annotation."""

    path: str
    line: int
    name: str
    category: str


@dataclass(frozen=True)
class AnnotationModuleMetrics:
    """Return-annotation metrics for one Python module."""

    path: str
    category: str
    functions: int
    missing_returns: int


@dataclass(frozen=True)
class AnnotationContractReport:
    """Aggregated return-annotation contract report."""

    modules: tuple[AnnotationModuleMetrics, ...]
    missing: tuple[MissingReturnAnnotation, ...]

    @property
    def ok(self) -> bool:
        return not any(item.category in BLOCKING_CATEGORIES for item in self.missing)


def main(*, reexec: bool = False) -> int:
    """Run the return-annotation contract and print a deterministic report."""

    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    report = run_contract(ROOT)
    print(format_report(report))
    return 0 if report.ok else 1


def run_contract(root: Path) -> AnnotationContractReport:
    """Analyze tracked Python files and return the annotation contract report."""

    modules: list[AnnotationModuleMetrics] = []
    missing: list[MissingReturnAnnotation] = []
    for path in tracked_python_files(root):
        category = classify_path(path.relative_to(root).as_posix())
        metrics, issues = analyze_module(path, root=root, category=category)
        modules.append(metrics)
        missing.extend(issues)
    return AnnotationContractReport(modules=tuple(modules), missing=tuple(missing))


def tracked_python_files(root: Path) -> tuple[Path, ...]:
    """Return tracked Python files relevant to the return-annotation contract."""

    try:
        payload = subprocess.check_output(
            ["git", "ls-files", "*.py"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        paths = [root / line for line in payload.splitlines() if line]
    except (FileNotFoundError, subprocess.CalledProcessError):
        paths = list(root.rglob("*.py"))
    return tuple(sorted(path for path in paths if not should_exclude(path)))


def analyze_module(
    path: Path,
    *,
    root: Path,
    category: str,
) -> tuple[AnnotationModuleMetrics, tuple[MissingReturnAnnotation, ...]]:
    """Compute return-annotation issues for a single module."""

    rel_path = path.relative_to(root).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
    function_count = 0
    missing: list[MissingReturnAnnotation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        function_count += 1
        if node.returns is None:
            missing.append(
                MissingReturnAnnotation(
                    path=rel_path,
                    line=node.lineno,
                    name=node.name,
                    category=category,
                )
            )
    metrics = AnnotationModuleMetrics(
        path=rel_path,
        category=category,
        functions=function_count,
        missing_returns=len(missing),
    )
    return metrics, tuple(missing)


def classify_path(rel_path: str) -> str:
    """Classify a path for contract reporting."""

    top_level = rel_path.split("/", 1)[0]
    if top_level in PRODUCTION_ROOTS:
        return "production"
    if top_level == SCRIPT_ROOT:
        return "scripts"
    if top_level == "tests":
        return "tests"
    return "other"


def should_exclude(path: Path) -> bool:
    """Return whether a path belongs to generated or local-only output."""

    return any(part in EXCLUDED_PATH_PARTS for part in path.parts)


def format_report(report: AnnotationContractReport) -> str:
    """Format the return-annotation report for local and CI logs."""

    category_counts = Counter(module.category for module in report.modules)
    lines = [
        "Return annotation contract report",
        f"- modules: {len(report.modules)} ({dict(sorted(category_counts.items()))})",
    ]
    for category, modules in grouped_modules(report.modules):
        function_count = sum(module.functions for module in modules)
        missing_count = sum(module.missing_returns for module in modules)
        coverage = 1.0 - (missing_count / function_count if function_count else 0.0)
        lines.append(
            f"- {category}: {function_count - missing_count}/{function_count} "
            f"functions annotated ({coverage:.1%}); missing={missing_count}"
        )
    blocking = [item for item in report.missing if item.category in BLOCKING_CATEGORIES]
    if blocking:
        lines.append("Blocking missing return annotations in tracked Python files:")
        lines.extend(
            f"- {item.path}:{item.line}: {item.name} missing return annotation"
            for item in blocking[:200]
        )
        if len(blocking) > 200:
            lines.append(f"- ... {len(blocking) - 200} additional blocking issues omitted")
    else:
        lines.append("Contract passed.")
    return "\n".join(lines)


def grouped_modules(
    modules: Iterable[AnnotationModuleMetrics],
) -> tuple[tuple[str, tuple[AnnotationModuleMetrics, ...]], ...]:
    """Group modules by category while preserving deterministic category ordering."""

    module_list = tuple(modules)
    categories = ("production", "scripts", "tests", "other")
    return tuple(
        (
            category,
            tuple(module for module in module_list if module.category == category),
        )
        for category in categories
        if any(module.category == category for module in module_list)
    )


__all__ = [
    "BLOCKING_CATEGORIES",
    "REPORT_ONLY_CATEGORIES",
    "AnnotationContractReport",
    "AnnotationModuleMetrics",
    "MissingReturnAnnotation",
    "analyze_module",
    "classify_path",
    "format_report",
    "grouped_modules",
    "main",
    "run_contract",
    "tracked_python_files",
]


if __name__ == "__main__":
    raise SystemExit(main(reexec=True))
