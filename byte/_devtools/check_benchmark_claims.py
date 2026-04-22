"""Validate that public benchmark claims are backed by proof or demoted."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script

sys.dont_write_bytecode = True

CLAIM_SURFACES = (
    "README.md",
    "CHANGELOG.md",
    "docs",
)
METRIC_PATTERN = re.compile(
    r"(?is)(false\s+reuse|confidence\s+(?:score\s+)?accuracy|deterministic\s+output\s+rate|confidence\s+ece)"
)
PROOF_LANGUAGE_PATTERN = re.compile(
    r"(?is)(independently\s+reproducible|public\s+proof|release\s+sign[- ]off|release\s+artifact)"
)
CHECKPOINT_LANGUAGE_PATTERN = re.compile(
    r"(?is)(checkpoint|partial|not\s+(?:a\s+)?public\s+proof|not\s+offered\s+as|not\s+presented\s+as|not\s+a\s+third-party\s+reproducibility\s+bundle)"
)
MANIFEST_NAME = "benchmark-release-manifest.json"
PUBLIC_PROOF_STATUS = "public_reproducible"
REQUIRED_MANIFEST_FIELDS = {
    "run_id",
    "commit_sha",
    "profile",
    "execution_mode",
    "providers",
    "systems",
    "raw_records_path",
    "summary_path",
    "checksums",
    "environment",
    "generated_at",
    "release_gate_passed",
    "public_proof_status",
}


@dataclass(frozen=True)
class BenchmarkClaimIssue:
    """A benchmark claim that is not adequately classified or backed."""

    path: str
    line: int
    reason: str


@dataclass(frozen=True)
class BenchmarkClaimReport:
    """Benchmark claim validation result."""

    issues: tuple[BenchmarkClaimIssue, ...]
    manifests: tuple[str, ...]
    public_manifests: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def main(*, reexec: bool = False) -> int:
    """Run the benchmark-claim contract."""

    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    report = run_contract(ROOT)
    print(format_report(report))
    return 0 if report.ok else 1


def run_contract(root: Path) -> BenchmarkClaimReport:
    """Scan public documentation and benchmark manifests for claim issues."""

    manifests = tuple(sorted(path for path in root.rglob(MANIFEST_NAME) if ".git" not in path.parts))
    issues, public_manifests = validate_manifests(root, manifests)
    for path in claim_surface_files(root):
        issues.extend(validate_claim_file(path, root=root, manifests=public_manifests))
    return BenchmarkClaimReport(
        issues=tuple(issues),
        manifests=tuple(path.relative_to(root).as_posix() for path in manifests),
        public_manifests=tuple(path.relative_to(root).as_posix() for path in public_manifests),
    )


def claim_surface_files(root: Path) -> tuple[Path, ...]:
    """Return documentation files that can carry public benchmark claims."""

    paths: list[Path] = []
    for surface in CLAIM_SURFACES:
        path = root / surface
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.suffix.lower() in {".md", ".rst", ".txt"}
            )
    return tuple(sorted(set(paths)))


def validate_manifests(
    root: Path,
    manifests: Iterable[Path],
) -> tuple[list[BenchmarkClaimIssue], tuple[Path, ...]]:
    """Validate release manifest shape when a public proof artifact is present."""

    issues: list[BenchmarkClaimIssue] = []
    public_manifests: list[Path] = []
    for manifest_path in manifests:
        rel_path = manifest_path.relative_to(root).as_posix()
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(BenchmarkClaimIssue(rel_path, exc.lineno, "manifest is invalid JSON"))
            continue
        missing = sorted(REQUIRED_MANIFEST_FIELDS - set(payload))
        if missing:
            issues.append(
                BenchmarkClaimIssue(
                    rel_path,
                    1,
                    f"manifest missing required fields: {', '.join(missing)}",
                )
            )
            continue
        issues.extend(validate_manifest_paths(root, rel_path, payload))
        if payload.get("public_proof_status") == PUBLIC_PROOF_STATUS:
            public_manifests.append(manifest_path)
    return issues, tuple(public_manifests)


def validate_manifest_paths(
    root: Path,
    manifest_rel_path: str,
    payload: dict[str, Any],
) -> tuple[BenchmarkClaimIssue, ...]:
    """Validate manifest artifact paths and checksum declarations."""

    issues: list[BenchmarkClaimIssue] = []
    raw_checksums = payload.get("checksums")
    checksums = raw_checksums if isinstance(raw_checksums, dict) else {}
    if not checksums:
        issues.append(BenchmarkClaimIssue(manifest_rel_path, 1, "manifest checksums must be non-empty"))
    for key in ("raw_records_path", "summary_path"):
        value = str(payload.get(key, "") or "")
        if not value:
            issues.append(BenchmarkClaimIssue(manifest_rel_path, 1, f"manifest {key} is empty"))
            continue
        artifact_path = root / value
        if not artifact_path.exists():
            issues.append(BenchmarkClaimIssue(manifest_rel_path, 1, f"manifest {key} does not exist: {value}"))
            continue
        expected_checksum = str(checksums.get(key, "") or "")
        if not expected_checksum:
            issues.append(BenchmarkClaimIssue(manifest_rel_path, 1, f"manifest checksum missing for {key}"))
            continue
        actual_checksum = _sha256_file(artifact_path)
        if actual_checksum != expected_checksum:
            issues.append(
                BenchmarkClaimIssue(
                    manifest_rel_path,
                    1,
                    f"manifest checksum mismatch for {key}: expected {expected_checksum}, got {actual_checksum}",
                )
            )
    return tuple(issues)


def validate_claim_file(
    path: Path,
    *,
    root: Path,
    manifests: tuple[Path, ...],
) -> tuple[BenchmarkClaimIssue, ...]:
    """Validate benchmark metric language in one public text file."""

    rel_path = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8")
    issues: list[BenchmarkClaimIssue] = []
    has_manifest = bool(manifests)
    has_checkpoint_language = CHECKPOINT_LANGUAGE_PATTERN.search(text) is not None
    for match in METRIC_PATTERN.finditer(text):
        if has_manifest or has_checkpoint_language:
            continue
        issues.append(
            BenchmarkClaimIssue(
                rel_path,
                line_number(text, match.start()),
                "benchmark metric appears without release manifest or checkpoint demotion",
            )
        )
    proof_match = PROOF_LANGUAGE_PATTERN.search(text)
    if proof_match and not has_manifest and not has_checkpoint_language:
        issues.append(
            BenchmarkClaimIssue(
                rel_path,
                line_number(text, proof_match.start()),
                "public proof language appears without a benchmark release manifest",
            )
        )
    return tuple(issues)


def line_number(text: str, offset: int) -> int:
    """Return a 1-based line number for a string offset."""

    return text.count("\n", 0, offset) + 1


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_report(report: BenchmarkClaimReport) -> str:
    """Format benchmark claim validation output."""

    lines = [
        "Benchmark claim contract report",
        f"- release manifests: {len(report.manifests)}",
        f"- public proof manifests: {len(report.public_manifests)}",
    ]
    lines.extend(f"- manifest: {path}" for path in report.manifests)
    if report.issues:
        lines.append("Failures:")
        lines.extend(f"- {issue.path}:{issue.line}: {issue.reason}" for issue in report.issues)
    else:
        lines.append("Contract passed.")
    return "\n".join(lines)


__all__ = [
    "PUBLIC_PROOF_STATUS",
    "BenchmarkClaimIssue",
    "BenchmarkClaimReport",
    "_sha256_file",
    "format_report",
    "main",
    "run_contract",
    "validate_claim_file",
    "validate_manifests",
]


if __name__ == "__main__":
    raise SystemExit(main(reexec=True))
