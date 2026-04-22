"""Validate Byte's optional dependency compatibility policy."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, BinaryIO, cast

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script

sys.dont_write_bytecode = True

EXPECTED_REQUIREMENTS = {
    "torch": ">=2.6,<3",
    "torchaudio": ">=2.6,<3",
    "transformers": ">=4.48,<6",
    "onnxruntime": ">=1.20,<2",
}
LATEST_ALLOWED_SMOKE = {
    "torch": "2.11.0",
    "torchaudio": "2.11.0",
    "transformers": "5.5.4",
    "onnxruntime": "1.24.4",
}
MIN_SUPPORTED_SMOKE = {
    "torch": "2.6.0",
    "torchaudio": "2.6.0",
    "transformers": "4.48.0",
    "onnxruntime": "1.20.0",
}


@dataclass(frozen=True)
class DependencyPolicyIssue:
    """A dependency policy violation."""

    name: str
    reason: str


@dataclass(frozen=True)
class DependencyPolicyReport:
    """Dependency policy validation result."""

    issues: tuple[DependencyPolicyIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def main(*, reexec: bool = False) -> int:
    """Run the dependency policy validation."""

    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    report = run_contract(ROOT)
    print(format_report(report))
    return 0 if report.ok else 1


def run_contract(root: Path) -> DependencyPolicyReport:
    """Validate package metadata for optional ML dependency ranges."""

    pyproject = load_pyproject(root)
    optional = pyproject["project"]["optional-dependencies"]
    requirements = collect_requirements(optional)
    issues: list[DependencyPolicyIssue] = []
    for name, expected in EXPECTED_REQUIREMENTS.items():
        specifier = requirements.get(name)
        if specifier is None:
            issues.append(DependencyPolicyIssue(name, "requirement is missing"))
            continue
        if specifier != expected:
            issues.append(
                DependencyPolicyIssue(
                    name,
                    f"expected specifier {expected}, found {specifier}",
                )
            )
            continue
        issues.extend(validate_smoke_versions(name, specifier))
    return DependencyPolicyReport(issues=tuple(issues))


def load_pyproject(root: Path) -> dict[str, Any]:
    """Load ``pyproject.toml`` as a Python mapping."""

    with (root / "pyproject.toml").open("rb") as handle:
        return toml_load(handle)


def toml_load(handle: BinaryIO) -> dict[str, Any]:
    """Load TOML using stdlib ``tomllib`` with a Python 3.10 fallback."""

    module_name = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    loader = cast(Callable[[BinaryIO], dict[str, Any]], vars(import_module(module_name))["load"])
    return loader(handle)


def collect_requirements(optional_dependencies: dict[str, list[str]]) -> dict[str, str]:
    """Collect optional dependency requirements by normalized package name."""

    collected: dict[str, str] = {}
    for dependencies in optional_dependencies.values():
        for dependency in dependencies:
            name, specifier = parse_requirement(dependency)
            collected[name] = specifier
    return collected


def parse_requirement(dependency: str) -> tuple[str, str]:
    """Return a normalized requirement name and its version specifier."""

    requirement = dependency.split(";", 1)[0].strip()
    for separator in (">=", "==", "~=", "<", ">", "!="):
        if separator in requirement:
            name, specifier = requirement.split(separator, 1)
            return name.strip().lower(), f"{separator}{specifier.strip()}"
    return requirement.strip().lower(), ""


def validate_smoke_versions(name: str, specifier: str) -> tuple[DependencyPolicyIssue, ...]:
    """Validate declared min/latest compatibility versions against a specifier."""

    issues: list[DependencyPolicyIssue] = []
    for label, versions in (
        ("minimum", MIN_SUPPORTED_SMOKE),
        ("latest", LATEST_ALLOWED_SMOKE),
    ):
        version = versions[name]
        if not version_allowed(version, specifier):
            issues.append(
                DependencyPolicyIssue(
                    name,
                    f"{label} smoke version {version} is outside declared range {specifier}",
                )
            )
    return tuple(issues)


def version_allowed(version: str, specifier: str) -> bool:
    """Return whether a dotted numeric version satisfies a simple specifier list."""

    parsed = parse_version(version)
    for piece in specifier.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith(">=") and parsed < parse_version(piece[2:]):
            return False
        if piece.startswith(">") and parsed <= parse_version(piece[1:]):
            return False
        if piece.startswith("<=") and parsed > parse_version(piece[2:]):
            return False
        if piece.startswith("<") and parsed >= parse_version(piece[1:]):
            return False
        if piece.startswith("==") and parsed != parse_version(piece[2:]):
            return False
    return True


def parse_version(version: str) -> tuple[int, ...]:
    """Parse the numeric prefix of a version string."""

    parts: list[int] = []
    for item in version.split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def format_report(report: DependencyPolicyReport) -> str:
    """Format dependency policy validation output."""

    lines = ["Dependency policy contract report"]
    if report.issues:
        lines.append("Failures:")
        lines.extend(f"- {issue.name}: {issue.reason}" for issue in report.issues)
    else:
        lines.append("Contract passed.")
    return "\n".join(lines)


__all__ = [
    "EXPECTED_REQUIREMENTS",
    "LATEST_ALLOWED_SMOKE",
    "MIN_SUPPORTED_SMOKE",
    "DependencyPolicyIssue",
    "DependencyPolicyReport",
    "format_report",
    "main",
    "run_contract",
]


if __name__ == "__main__":
    raise SystemExit(main(reexec=True))
