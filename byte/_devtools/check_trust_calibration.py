"""Validate Byte trust calibration provenance and score ownership."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script

sys.dont_write_bytecode = True

TRUST_SCORE_MODULES = (
    "byte/trust/_scoring.py",
    "byte/trust/_references.py",
    "byte/trust/_risk.py",
    "byte/trust/_contracts.py",
)
ALLOWED_FLOAT_VALUES = {0.0, 1.0}
CALIBRATION_ARTIFACT = "byte/trust/calibration/byte-trust-v2.json"
EXPECTED_CALIBRATION_STATUS = "internal_checkpoint"
INTERNAL_CHECKPOINT_STATUSES = frozenset({EXPECTED_CALIBRATION_STATUS})
PUBLIC_REPRODUCIBLE_STATUS = "public_reproducible"
KNOWN_PUBLIC_PROOF_STATUSES = frozenset({"not_public_proof", PUBLIC_REPRODUCIBLE_STATUS})


@dataclass(frozen=True)
class TrustCalibrationIssue:
    """A trust calibration contract issue."""

    path: str
    line: int
    reason: str


@dataclass(frozen=True)
class TrustCalibrationReport:
    """Aggregated trust calibration contract report."""

    issues: tuple[TrustCalibrationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def main(*, reexec: bool = False) -> int:
    """Run the trust calibration contract."""

    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    report = run_contract(ROOT)
    print(format_report(report))
    return 0 if report.ok else 1


def run_contract(root: Path) -> TrustCalibrationReport:
    """Validate the calibration artifact and score-module literals."""

    issues: list[TrustCalibrationIssue] = []
    artifact = root / CALIBRATION_ARTIFACT
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if str(payload.get("checksum", "")) != calibration_checksum(payload):
        issues.append(
            TrustCalibrationIssue(
                CALIBRATION_ARTIFACT,
                1,
                "calibration checksum does not match canonical payload",
            )
        )
    issues.extend(validate_calibration_proof_status(root, payload))
    for rel_path in TRUST_SCORE_MODULES:
        issues.extend(validate_score_literals(root / rel_path, root=root))
    return TrustCalibrationReport(issues=tuple(issues))


def calibration_checksum(payload: dict[str, Any]) -> str:
    """Return the canonical SHA-256 checksum for a calibration payload."""

    canonical_payload = dict(payload)
    canonical_payload.pop("checksum", None)
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_calibration_proof_status(
    root: Path,
    payload: dict[str, Any],
) -> tuple[TrustCalibrationIssue, ...]:
    """Validate that private checkpoints do not ship public-proof metrics."""

    issues: list[TrustCalibrationIssue] = []
    status = str(payload.get("status", ""))
    source_status = str(payload.get("source_status", ""))
    public_proof_status = str(payload.get("public_proof_status", ""))
    public_proof_manifest = str(payload.get("public_proof_manifest", "")).strip()
    validation_metrics = payload.get("validation_metrics", {})
    if status != EXPECTED_CALIBRATION_STATUS:
        issues.append(
            TrustCalibrationIssue(
                CALIBRATION_ARTIFACT,
                1,
                "unexpected calibration status",
            )
        )
    if public_proof_status not in KNOWN_PUBLIC_PROOF_STATUSES:
        issues.append(
            TrustCalibrationIssue(
                CALIBRATION_ARTIFACT,
                1,
                "unknown public proof status",
            )
        )
    if status in INTERNAL_CHECKPOINT_STATUSES and source_status != "private_internal_checkpoint":
        issues.append(
            TrustCalibrationIssue(
                CALIBRATION_ARTIFACT,
                1,
                "internal checkpoint calibration must declare private source status",
            )
        )
    if not isinstance(validation_metrics, dict):
        issues.append(
            TrustCalibrationIssue(
                CALIBRATION_ARTIFACT,
                1,
                "validation_metrics must be an object",
            )
        )
        validation_metrics = {}
    if validation_metrics and public_proof_status != PUBLIC_REPRODUCIBLE_STATUS:
        issues.append(
            TrustCalibrationIssue(
                CALIBRATION_ARTIFACT,
                1,
                "validation metrics require public proof status and a release manifest",
            )
        )
    if status in INTERNAL_CHECKPOINT_STATUSES and validation_metrics:
        issues.append(
            TrustCalibrationIssue(
                CALIBRATION_ARTIFACT,
                1,
                "internal checkpoint calibration must not expose private validation metrics",
            )
        )
    if public_proof_status == PUBLIC_REPRODUCIBLE_STATUS:
        if not public_proof_manifest:
            issues.append(
                TrustCalibrationIssue(
                    CALIBRATION_ARTIFACT,
                    1,
                    "public reproducible calibration requires public_proof_manifest",
                )
            )
        elif not (root / public_proof_manifest).exists():
            issues.append(
                TrustCalibrationIssue(
                    CALIBRATION_ARTIFACT,
                    1,
                    "public proof manifest does not exist",
                )
            )
    return tuple(issues)


def validate_score_literals(path: Path, *, root: Path) -> tuple[TrustCalibrationIssue, ...]:
    """Reject unexplained float literals in trust score modules."""

    rel_path = path.relative_to(root).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
    issues: list[TrustCalibrationIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            if node.value in ALLOWED_FLOAT_VALUES:
                continue
            issues.append(
                TrustCalibrationIssue(
                    rel_path,
                    node.lineno,
                    f"float literal {node.value!r} must live in the trust calibration artifact",
                )
            )
    return tuple(issues)


def format_report(report: TrustCalibrationReport) -> str:
    """Format trust calibration validation output."""

    lines = ["Trust calibration contract report"]
    if report.issues:
        lines.append("Failures:")
        lines.extend(f"- {issue.path}:{issue.line}: {issue.reason}" for issue in report.issues)
    else:
        lines.append("Contract passed.")
    return "\n".join(lines)


__all__ = [
    "CALIBRATION_ARTIFACT",
    "EXPECTED_CALIBRATION_STATUS",
    "INTERNAL_CHECKPOINT_STATUSES",
    "KNOWN_PUBLIC_PROOF_STATUSES",
    "PUBLIC_REPRODUCIBLE_STATUS",
    "TrustCalibrationIssue",
    "TrustCalibrationReport",
    "calibration_checksum",
    "format_report",
    "main",
    "run_contract",
    "validate_calibration_proof_status",
    "validate_score_literals",
]


if __name__ == "__main__":
    raise SystemExit(main(reexec=True))
