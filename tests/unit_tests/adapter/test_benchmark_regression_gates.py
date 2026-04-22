from pathlib import Path

import pytest

REPORTS_DIR = Path(__file__).resolve().parents[3] / "docs" / "reports"
LATEST_SUMMARY = REPORTS_DIR / "latest_release_summary.md"
PUBLIC_PROOF_DIR = REPORTS_DIR / "public-proof"
PUBLIC_PROOF_BUNDLE = PUBLIC_PROOF_DIR / "openai-tier1-local-20260419T154401Z"
PUBLIC_PROOF_MANIFEST = PUBLIC_PROOF_BUNDLE / "benchmark-release-manifest.json"


def _load_summary() -> str:
    return LATEST_SUMMARY.read_text(encoding="utf-8")


def test_reports_directory_retains_curated_summary_and_public_proof_bundle() -> None:
    assert sorted(path.name for path in REPORTS_DIR.iterdir()) == [
        "README.md",
        "latest_release_summary.md",
        "public-proof",
    ]
    assert sorted(path.name for path in PUBLIC_PROOF_DIR.iterdir()) == [
        "README.md",
        "openai-tier1-local-20260419T154401Z",
    ]


@pytest.mark.benchmark_regression
def test_public_proof_bundle_contains_manifest_and_raw_artifacts() -> None:
    manifest_text = PUBLIC_PROOF_MANIFEST.read_text(encoding="utf-8")
    engineering_json = PUBLIC_PROOF_BUNDLE / "local" / "engineering_report.json"

    assert '"public_proof_status": "public_reproducible"' in manifest_text
    assert '"run_id": "tier1-20260419T154401Z-5665dac5"' in manifest_text
    assert '"release_gate_passed": false' in manifest_text
    assert (PUBLIC_PROOF_BUNDLE / "summary.json").exists()
    assert (PUBLIC_PROOF_BUNDLE / "summary.md").exists()
    assert engineering_json.exists()
    assert engineering_json.stat().st_size < 800_000
    assert (PUBLIC_PROOF_BUNDLE / "local" / "engineering_report.md").exists()


@pytest.mark.benchmark_regression
def test_latest_release_summary_records_run_identity() -> None:
    summary = _load_summary()

    assert "tier1_v2_deepseek-20260401T143351Z-3e1bcf1b" in summary
    assert "2026-04-01T14:33:51" in summary
    assert "Profile: `tier1_v2_deepseek`" in summary
    assert "Execution mode: `partial`" in summary
    assert "Providers executed: `deepseek`" in summary
    assert "Systems executed: `byte`" in summary


@pytest.mark.benchmark_regression
def test_latest_release_summary_demotes_core_gate_results_to_checkpoint() -> None:
    summary = _load_summary()

    assert "False reuse rate: `0.0000` against a `0.0100` threshold" in summary
    assert "Confidence score accuracy: `0.9722` against a `0.9000` threshold" in summary
    assert "not offered as a third-party reproducibility bundle" in summary
    assert "focused engineering checkpoint" in summary


@pytest.mark.benchmark_regression
def test_latest_release_summary_states_release_readiness_scope() -> None:
    summary = _load_summary()

    assert "targeted growth hardening run" in summary
    assert "not a full release sign-off" in summary
    assert "release gate remained `false`" in summary
    assert "did not emit a dollar-impact payload" in summary
