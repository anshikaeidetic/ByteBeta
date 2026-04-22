"""Validate Byte's maintained typing, documentation, and exception boundaries."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script
from byte._devtools.verification_targets import MAINTAINABILITY_TARGETS

sys.dont_write_bytecode = True

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
PRODUCTION_ROOTS = {"byte", "byte_server", "byte_inference", "byte_memory"}
LEGACY_BENCHMARK_PREFIXES = (
    "byte/benchmarking/_program_impl/",
    "byte/benchmarking/_workload_impl/",
)
BOUNDARY_COMMENT_MARKERS = (
    "provider boundary",
    "callback boundary",
    "boundary cleanup",
    "boundary path",
    "telemetry",
    "middleware",
    "route normalization",
    "best-effort",
    "defensive",
    "byte-boundary",
)
BOUNDARY_BODY_MARKERS = (
    "boundary",
    "provider",
    "telemetry",
    "middleware",
    "callback",
    "cleanup",
)
MAINTAINABILITY_RUFF_RULES = (
    "ANN001",
    "ANN201",
    "ANN202",
    "D100",
    "D103",
    "BLE001",
)

MIN_PRODUCTION_MODULE_DOCSTRING_RATIO = 0.50
MIN_PRODUCTION_ANNOTATION_RATIO = 0.55
MAX_PRODUCTION_BROAD_EXCEPTIONS = 160


@dataclass(frozen=True)
class FunctionIssue:
    """A public helper that violates the maintained target contract."""

    path: str
    line: int
    name: str
    reason: str


@dataclass(frozen=True)
class BroadExceptionIssue:
    """A broad exception handler that is not documented as a boundary catch."""

    path: str
    line: int
    reason: str


@dataclass
class ModuleMetrics:
    """AST-derived maintainability metrics for one Python module."""

    path: str
    category: str
    module_docstring: bool
    functions: int = 0
    function_docstrings: int = 0
    fully_annotated_functions: int = 0
    broad_exceptions: int = 0
    justified_broad_exceptions: int = 0
    target_issues: list[str] = field(default_factory=list)
    function_issues: list[FunctionIssue] = field(default_factory=list)
    broad_exception_issues: list[BroadExceptionIssue] = field(default_factory=list)


@dataclass(frozen=True)
class MaintainabilityReport:
    """Aggregated maintainability metrics and contract failures."""

    modules: tuple[ModuleMetrics, ...]
    failures: tuple[str, ...]
    ruff_returncode: int

    @property
    def ok(self) -> bool:
        return not self.failures and self.ruff_returncode == 0


class _ParentVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: dict[ast.AST, ast.AST] = {}

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
        super().generic_visit(node)


def main(*, reexec: bool = False) -> int:
    """Run the maintainability contract and print a concise report."""

    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    report = run_contract(ROOT)
    print(format_report(report))
    return 0 if report.ok else 1


def run_contract(root: Path) -> MaintainabilityReport:
    """Analyze the repository and run the focused Ruff maintainability profile."""

    target_paths = expand_target_paths(root, MAINTAINABILITY_TARGETS)
    target_rel_paths = {path.relative_to(root).as_posix() for path in target_paths}
    modules = tuple(analyze_module(path, root=root, target_paths=target_rel_paths) for path in python_files(root))
    failures = list(contract_failures(modules))
    ruff_returncode = run_ruff_profile(root, MAINTAINABILITY_TARGETS)
    if ruff_returncode != 0:
        failures.append("focused Ruff maintainability profile failed")
    return MaintainabilityReport(modules=modules, failures=tuple(failures), ruff_returncode=ruff_returncode)


def expand_target_paths(root: Path, targets: Sequence[str]) -> tuple[Path, ...]:
    """Resolve files covered by a target list while preserving deterministic order."""

    resolved: list[Path] = []
    seen: set[str] = set()
    for target in targets:
        path = root / target
        candidates = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.exists() or should_exclude(candidate):
                continue
            rel_path = candidate.relative_to(root).as_posix()
            if rel_path in seen:
                continue
            seen.add(rel_path)
            resolved.append(candidate)
    return tuple(resolved)


def python_files(root: Path) -> tuple[Path, ...]:
    """Return every tracked-quality Python file under production, scripts, and tests."""

    candidates: list[Path] = []
    for top_level in (*sorted(PRODUCTION_ROOTS), "scripts", "tests"):
        path = root / top_level
        if path.exists():
            candidates.extend(path.rglob("*.py"))
    return tuple(sorted(path for path in candidates if not should_exclude(path)))


def analyze_module(path: Path, *, root: Path, target_paths: set[str]) -> ModuleMetrics:
    """Compute AST metrics and strict-target issues for a single module."""

    rel_path = path.relative_to(root).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parent_visitor = _ParentVisitor()
    parent_visitor.visit(tree)
    exported_names = exported_public_names(tree)
    source_lines = path.read_text(encoding="utf-8").splitlines()
    category = classify_path(rel_path, target_paths)
    is_target = rel_path in target_paths
    metrics = ModuleMetrics(
        path=rel_path,
        category=category,
        module_docstring=ast.get_docstring(tree) is not None,
    )
    if is_target and not metrics.module_docstring:
        metrics.target_issues.append(f"{rel_path}: missing module docstring")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            record_function_metrics(
                metrics,
                node,
                parent_visitor.parents,
                exported_names,
                enforce_target=is_target,
            )
        elif isinstance(node, ast.ExceptHandler) and is_broad_except(node):
            metrics.broad_exceptions += 1
            if is_justified_boundary_handler(node, source_lines):
                metrics.justified_broad_exceptions += 1
            elif is_target:
                metrics.broad_exception_issues.append(
                    BroadExceptionIssue(
                        path=rel_path,
                        line=node.lineno,
                        reason="broad exception handler is not classified as a boundary catch",
                    )
                )
    return metrics


def record_function_metrics(
    metrics: ModuleMetrics,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
    exported_names: set[str],
    *,
    enforce_target: bool,
) -> None:
    """Update function counters and strict-target failures for one function."""

    metrics.functions += 1
    if ast.get_docstring(node) is not None:
        metrics.function_docstrings += 1
    if has_complete_annotations(node):
        metrics.fully_annotated_functions += 1

    if not enforce_target or not is_surface_function(node, parents):
        return
    if not has_complete_annotations(node):
        metrics.function_issues.append(
            FunctionIssue(metrics.path, node.lineno, node.name, "missing complete annotations")
        )
    if requires_docstring(node, parents, exported_names) and ast.get_docstring(node) is None:
        metrics.function_issues.append(
            FunctionIssue(metrics.path, node.lineno, node.name, "missing public helper docstring")
        )


def exported_public_names(tree: ast.Module) -> set[str]:
    """Return literal names declared in a module-level ``__all__``."""

    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            for element in node.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    names.add(element.value)
    return names


def has_complete_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function has argument and return annotations."""

    arguments = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    if node.args.vararg is not None:
        arguments.append(node.args.vararg)
    if node.args.kwarg is not None:
        arguments.append(node.args.kwarg)
    return node.returns is not None and all(
        argument.arg in {"self", "cls"} or argument.annotation is not None
        for argument in arguments
    )


def is_surface_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Return whether a function is a module-level or class-level public helper."""

    if node.name.startswith("_"):
        return False
    parent = parents.get(node)
    return isinstance(parent, (ast.Module, ast.ClassDef))


def requires_docstring(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
    exported_names: set[str],
) -> bool:
    """Require docs for exported or nontrivial top-level public helpers."""

    parent = parents.get(node)
    if not isinstance(parent, ast.Module):
        return False
    return node.name in exported_names or is_nontrivial_function(node)


def is_nontrivial_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function has enough behavior to require a docstring."""

    if len(node.body) > 8:
        return True
    return any(
        isinstance(
            child,
            (
                ast.AsyncFor,
                ast.AsyncWith,
                ast.For,
                ast.Raise,
                ast.Try,
                ast.While,
                ast.With,
                ast.Yield,
                ast.YieldFrom,
            ),
        )
        for child in ast.walk(node)
    )


def is_broad_except(node: ast.ExceptHandler) -> bool:
    """Return whether an exception handler catches all normal exceptions."""

    if node.type is None:
        return True
    if isinstance(node.type, ast.Name):
        return node.type.id in {"Exception", "BaseException"}
    if isinstance(node.type, ast.Tuple):
        return any(
            isinstance(element, ast.Name) and element.id in {"Exception", "BaseException"}
            for element in node.type.elts
        )
    return False


def is_justified_boundary_handler(node: ast.ExceptHandler, source_lines: Sequence[str]) -> bool:
    """Return whether a broad handler is explicitly marked as a boundary catch."""

    line_text = source_lines[node.lineno - 1].lower() if 0 < node.lineno <= len(source_lines) else ""
    if any(marker in line_text for marker in BOUNDARY_COMMENT_MARKERS):
        return True
    body_strings = [
        value.value.lower()
        for value in ast.walk(ast.Module(body=node.body, type_ignores=[]))
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    ]
    body_text = " ".join(body_strings)
    return any(marker in body_text for marker in BOUNDARY_BODY_MARKERS)


def classify_path(rel_path: str, target_paths: set[str]) -> str:
    """Classify a module for maintainability reporting."""

    if rel_path in target_paths:
        return "strict_target"
    if rel_path.startswith("tests/"):
        return "tests"
    if rel_path.startswith("scripts/"):
        return "scripts"
    if any(rel_path.startswith(prefix) for prefix in LEGACY_BENCHMARK_PREFIXES):
        return "legacy_benchmark_impl"
    if rel_path.split("/", 1)[0] in PRODUCTION_ROOTS:
        return "production"
    return "private_internal"


def should_exclude(path: Path) -> bool:
    """Return whether a path belongs to generated or local-only output."""

    return any(part in EXCLUDED_PATH_PARTS for part in path.parts)


def contract_failures(modules: Iterable[ModuleMetrics]) -> tuple[str, ...]:
    """Return maintainability contract failures for the current report."""

    module_list = tuple(modules)
    failures: list[str] = []
    production = [
        module
        for module in module_list
        if module.category in {"production", "strict_target"}
        and not any(module.path.startswith(prefix) for prefix in LEGACY_BENCHMARK_PREFIXES)
    ]
    production_module_ratio = ratio(
        sum(1 for module in production if module.module_docstring),
        len(production),
    )
    production_annotation_ratio = ratio(
        sum(module.fully_annotated_functions for module in production),
        sum(module.functions for module in production),
    )
    production_broad_handlers = sum(module.broad_exceptions for module in production)
    if production_module_ratio < MIN_PRODUCTION_MODULE_DOCSTRING_RATIO:
        failures.append(
            "production module docstring ratio "
            f"{production_module_ratio:.1%} is below {MIN_PRODUCTION_MODULE_DOCSTRING_RATIO:.1%}"
        )
    if production_annotation_ratio < MIN_PRODUCTION_ANNOTATION_RATIO:
        failures.append(
            "production annotation ratio "
            f"{production_annotation_ratio:.1%} is below {MIN_PRODUCTION_ANNOTATION_RATIO:.1%}"
        )
    if production_broad_handlers > MAX_PRODUCTION_BROAD_EXCEPTIONS:
        failures.append(
            "production broad exception handlers "
            f"{production_broad_handlers} exceed {MAX_PRODUCTION_BROAD_EXCEPTIONS}"
        )
    for module in module_list:
        failures.extend(module.target_issues)
        failures.extend(
            f"{issue.path}:{issue.line}: {issue.name} {issue.reason}"
            for issue in module.function_issues
        )
        failures.extend(
            f"{issue.path}:{issue.line}: {issue.reason}"
            for issue in module.broad_exception_issues
        )
    return tuple(failures)


def ratio(numerator: int, denominator: int) -> float:
    """Return a safe ratio for report thresholds."""

    return numerator / denominator if denominator else 1.0


def run_ruff_profile(root: Path, targets: Sequence[str]) -> int:
    """Run Ruff's focused annotation, docstring, and broad-exception checks."""

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-B",
        "-m",
        "ruff",
        "check",
        *targets,
        "--select",
        ",".join(MAINTAINABILITY_RUFF_RULES),
        "--config",
        "ruff.toml",
    ]
    return subprocess.run(command, cwd=root, env=env, check=False).returncode


def format_report(report: MaintainabilityReport) -> str:
    """Format the maintainability report for local and CI logs."""

    category_counts = Counter(module.category for module in report.modules)
    function_count = sum(module.functions for module in report.modules)
    function_docstrings = sum(module.function_docstrings for module in report.modules)
    fully_annotated = sum(module.fully_annotated_functions for module in report.modules)
    module_docstrings = sum(1 for module in report.modules if module.module_docstring)
    broad_handlers = sum(module.broad_exceptions for module in report.modules)
    justified_handlers = sum(module.justified_broad_exceptions for module in report.modules)
    lines = [
        "Maintainability contract report",
        f"- modules: {len(report.modules)} ({dict(sorted(category_counts.items()))})",
        f"- module docstrings: {module_docstrings}/{len(report.modules)} ({ratio(module_docstrings, len(report.modules)):.1%})",
        f"- function docstrings: {function_docstrings}/{function_count} ({ratio(function_docstrings, function_count):.1%})",
        f"- fully annotated functions: {fully_annotated}/{function_count} ({ratio(fully_annotated, function_count):.1%})",
        f"- broad exception handlers: {broad_handlers} ({justified_handlers} justified boundary catches)",
        f"- focused Ruff profile: {'passed' if report.ruff_returncode == 0 else 'failed'}",
    ]
    if report.failures:
        lines.append("Failures:")
        lines.extend(f"- {failure}" for failure in report.failures)
    else:
        lines.append("Contract passed.")
    return "\n".join(lines)


__all__ = [
    "MAINTAINABILITY_RUFF_RULES",
    "BroadExceptionIssue",
    "FunctionIssue",
    "MaintainabilityReport",
    "ModuleMetrics",
    "analyze_module",
    "classify_path",
    "contract_failures",
    "expand_target_paths",
    "format_report",
    "has_complete_annotations",
    "is_broad_except",
    "is_justified_boundary_handler",
    "main",
    "python_files",
    "record_function_metrics",
    "requires_docstring",
    "run_contract",
    "run_ruff_profile",
]


if __name__ == "__main__":
    raise SystemExit(main(reexec=True))
