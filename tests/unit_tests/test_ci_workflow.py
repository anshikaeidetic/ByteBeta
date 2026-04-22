from __future__ import annotations

import re
from pathlib import Path

WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def _workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _job_block(job_name: str) -> str:
    pattern = rf"(?ms)^  {re.escape(job_name)}:\n(.*?)(?=^  [A-Za-z0-9_.-]+:\n|\Z)"
    match = re.search(pattern, _workflow_text())
    assert match is not None, job_name
    return match.group(1)


def test_ci_pins_node24_compatible_actions() -> None:
    workflow = _workflow_text()

    assert "uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd" in workflow
    assert "uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405" in workflow


def test_ci_runs_posix_bootstrap_via_bash() -> None:
    workflow = _workflow_text()

    assert "run: bash ./bootstrap-dev.sh --skip-pre-commit" in workflow
    assert "run: ./bootstrap-dev.sh --skip-pre-commit" not in workflow


def test_ci_separates_base_and_optional_install_profiles() -> None:
    standard_install = 'python -m pip install -c requirements/dev-ci.txt -e ".[dev,observability,server]"'
    security_install = (
        'python -m pip install -c requirements/security-ci.txt -e ".[dev,security,observability,server]"'
    )
    coverage_install = 'python -m pip install -c requirements/dev-ci.txt -e ".[dev,test,onnx,faiss,observability,server]"'
    coverage_redis_install = "python -m pip install -c requirements/dev-ci.txt redis"
    integration_install = (
        'python -m pip install -c requirements/dev-ci.txt -e ".[dev,test,onnx,observability,server]"'
    )

    for job_name in ("hygiene", "lint", "typecheck", "maintainability", "package", "compile"):
        block = _job_block(job_name)
        assert standard_install in block
        assert security_install not in block
        assert coverage_install not in block

    assert security_install in _job_block("security")
    assert "python -m pip install semgrep==1.157.0" in _job_block("security")
    assert coverage_install in _job_block("coverage")
    assert coverage_redis_install in _job_block("coverage")
    assert integration_install in _job_block("integration-smoke")


def test_ci_includes_optional_feature_unit_lanes() -> None:
    workflow = _workflow_text()

    assert "unit-base-py310:" in workflow
    assert "unit-base-py311:" in workflow
    assert "unit-base-py312:" in workflow
    assert "unit-openai:" in workflow
    assert "unit-groq:" in workflow
    assert "unit-sqlalchemy:" in workflow
    assert "unit-transformers:" in workflow
    assert "unit-langchain:" in workflow
    assert "unit-semantic-cache:" in workflow
    assert "benchmark-regression:" in workflow

    assert 'python -m pip install -c requirements/dev-ci.txt -e ".[dev,openai,onnx,sql,faiss,observability,server]"' in _job_block(
        "unit-openai"
    )
    assert "python -m pip install -c requirements/dev-ci.txt pillow" in _job_block("unit-openai")
    assert "python scripts/run_optional_feature_tests.py openai pillow" in _job_block("unit-openai")
    assert 'python -m pip install -c requirements/dev-ci.txt -e ".[dev,groq,onnx,sql,faiss,observability,server]"' in _job_block(
        "unit-groq"
    )
    assert "python scripts/run_optional_feature_tests.py groq" in _job_block("unit-groq")
    assert 'python -m pip install -c requirements/dev-ci.txt -e ".[dev,sql,observability,server]"' in _job_block(
        "unit-sqlalchemy"
    )
    assert "python scripts/run_optional_feature_tests.py sqlalchemy" in _job_block(
        "unit-sqlalchemy"
    )
    assert 'python -m pip install -c requirements/dev-ci.txt -e ".[dev,huggingface,observability,server]"' in _job_block(
        "unit-transformers"
    )
    assert "python scripts/run_optional_feature_tests.py transformers" in _job_block(
        "unit-transformers"
    )
    assert 'python -m pip install -c requirements/dev-ci.txt -e ".[dev,langchain,onnx,sql,faiss,observability,server]"' in _job_block(
        "unit-langchain"
    )
    assert "python scripts/run_optional_feature_tests.py langchain" in _job_block(
        "unit-langchain"
    )
    assert 'python -m pip install -c requirements/dev-ci.txt -e ".[dev,onnx,sql,faiss,observability,server]"' in _job_block(
        "unit-semantic-cache"
    )
    assert "python scripts/run_optional_feature_tests.py onnx sqlalchemy faiss" in _job_block(
        "unit-semantic-cache"
    )
    benchmark_block = _job_block("benchmark-regression")
    assert "Run benchmark regression tests" in benchmark_block
    assert "tests/unit_tests/benchmarking" in benchmark_block
    assert "test_deepseek_runtime_benchmark_plan.py" in benchmark_block
    assert "- maintainability" in _job_block("unit-tests")


def test_ci_includes_coverage_and_integration_smoke_jobs() -> None:
    workflow = _workflow_text()

    assert "coverage:" in workflow
    assert "integration-smoke:" in workflow
    assert "maintainability:" in workflow
    assert "annotation-contract:" in workflow
    assert "benchmark-claim-contract:" in workflow
    assert "trust-calibration-contract:" in workflow
    assert "dependency-compatibility:" in workflow
    assert "python scripts/run_coverage.py" in _job_block("coverage")
    assert "python scripts/run_integration_smoke.py" in _job_block("integration-smoke")
    assert "python scripts/check_maintainability_contract.py" in _job_block("maintainability")
    assert "python scripts/check_annotation_contract.py" in _job_block("annotation-contract")
    assert "python scripts/check_benchmark_claims.py" in _job_block("benchmark-claim-contract")
    assert "python scripts/check_trust_calibration.py" in _job_block("trust-calibration-contract")
    assert "python scripts/check_dependency_policy.py" in _job_block("dependency-compatibility")
    unit_gate = _job_block("unit-tests")
    assert "- annotation-contract" in unit_gate
    assert "- benchmark-claim-contract" in unit_gate
    assert "- dependency-compatibility" in unit_gate
    assert "- trust-calibration-contract" in unit_gate
    assert 'BYTE_RUN_LIVE_INTEGRATION: "1"' in _job_block("integration-smoke")


def test_package_job_checks_artifact_contents_after_build() -> None:
    package_block = _job_block("package")

    assert "python -m build --no-isolation --sdist --wheel" in package_block
    assert "python scripts/check_package_artifacts.py" in package_block
    assert "python scripts/check_install_parity.py" in package_block
    assert "python -m twine check dist/*" in package_block
