from __future__ import annotations

import textwrap
from pathlib import Path

from byte._devtools import (
    check_annotation_contract,
    check_benchmark_claims,
    check_dependency_policy,
    check_trust_calibration,
)
from byte.trust._calibration import load_trust_calibration


def test_annotation_contract_reports_missing_production_return(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        textwrap.dedent(
            """
            def missing():
                return 1

            def present() -> int:
                return 2
            """
        ),
        encoding="utf-8",
    )

    metrics, issues = check_annotation_contract.analyze_module(
        module,
        root=tmp_path,
        category="production",
    )

    assert metrics.functions == 2
    assert metrics.missing_returns == 1
    assert issues[0].name == "missing"


def test_annotation_contract_blocks_tracked_test_files_too() -> None:
    report = check_annotation_contract.AnnotationContractReport(
        modules=(),
        missing=(
            check_annotation_contract.MissingReturnAnnotation(
                path="tests/unit_tests/test_sample.py",
                line=1,
                name="run",
                category="tests",
            ),
        ),
    )

    assert not report.ok
    assert "tracked Python files" in check_annotation_contract.format_report(report)


def test_benchmark_claim_contract_blocks_undemoted_metric_claim(tmp_path: Path) -> None:
    claim = tmp_path / "README.md"
    claim.write_text("False reuse rate: `0.0000` against target.\n", encoding="utf-8")

    issues = check_benchmark_claims.validate_claim_file(
        claim,
        root=tmp_path,
        manifests=(),
    )

    assert len(issues) == 1
    assert "without release manifest or checkpoint demotion" in issues[0].reason


def test_benchmark_claim_contract_allows_checkpoint_metric_claim(tmp_path: Path) -> None:
    claim = tmp_path / "README.md"
    claim.write_text(
        "False reuse rate: `0.0000`; this is a checkpoint, not a public proof.\n",
        encoding="utf-8",
    )

    assert (
        check_benchmark_claims.validate_claim_file(
            claim,
            root=tmp_path,
            manifests=(),
        )
        == ()
    )


def test_benchmark_claim_contract_rejects_manifest_checksum_mismatch(tmp_path: Path) -> None:
    bundle = tmp_path / "proof"
    bundle.mkdir()
    raw_records = bundle / "raw.json"
    summary = bundle / "summary.json"
    raw_records.write_text('{"rows": []}\n', encoding="utf-8")
    summary.write_text('{"summary": true}\n', encoding="utf-8")

    issues = check_benchmark_claims.validate_manifest_paths(
        tmp_path,
        "proof/benchmark-release-manifest.json",
        {
            "raw_records_path": "proof/raw.json",
            "summary_path": "proof/summary.json",
            "checksums": {
                "raw_records_path": "deadbeef",
                "summary_path": check_benchmark_claims._sha256_file(summary),
            },
        },
    )

    assert len(issues) == 1
    assert "checksum mismatch" in issues[0].reason


def test_trust_calibration_artifact_is_checksum_valid() -> None:
    calibration = load_trust_calibration()

    assert calibration.version == "byte-trust-v2"
    assert calibration.status == "internal_checkpoint"
    assert calibration.source_status == "private_internal_checkpoint"
    assert calibration.public_proof_status == "not_public_proof"
    assert calibration.public_proof_manifest == ""
    assert calibration.validation_metrics == {}
    assert calibration.checksum_valid is True
    assert calibration.confidence_score("reasoning_dynamic_verified") == 0.96
    assert calibration.reference_threshold("curated_policy_block_score") == 8.0


def test_trust_calibration_contract_rejects_internal_validation_metrics(tmp_path: Path) -> None:
    issues = check_trust_calibration.validate_calibration_proof_status(
        tmp_path,
        {
            "status": "internal_checkpoint",
            "source_status": "private_internal_checkpoint",
            "public_proof_status": "not_public_proof",
            "public_proof_manifest": "",
            "validation_metrics": {"confidence_score_accuracy": 0.9722},
        },
    )

    reasons = {issue.reason for issue in issues}
    assert "internal checkpoint calibration must not expose private validation metrics" in reasons
    assert "validation metrics require public proof status and a release manifest" in reasons


def test_trust_calibration_contract_rejects_unowned_float_literal(tmp_path: Path) -> None:
    module = tmp_path / "score_module.py"
    module.write_text("SCORE = 0.93\n", encoding="utf-8")

    issues = check_trust_calibration.validate_score_literals(module, root=tmp_path)

    assert len(issues) == 1
    assert "calibration artifact" in issues[0].reason


def test_dependency_policy_contract_accepts_declared_ml_ranges() -> None:
    report = check_dependency_policy.run_contract(Path.cwd())

    assert report.ok
