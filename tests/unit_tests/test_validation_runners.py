from __future__ import annotations

import subprocess
from pathlib import Path

from byte._devtools import (
    check_devx_contract,
    check_maintainability_contract,
    run_coverage,
    run_integration_smoke,
    run_lint,
    run_optional_feature_tests,
    run_typecheck,
    run_unit_tests,
)


def test_run_lint_uses_bytecode_safe_python(monkeypatch) -> object:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, *, cwd, env, check) -> object:
        del cwd, check
        calls.append((command, env))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_lint.subprocess, "run", fake_run)

    assert run_lint.main() == 0
    assert calls[0][0] == [
        run_lint.sys.executable,
        "-B",
        "-m",
        "ruff",
        "check",
        *run_lint.LINT_TARGETS,
        "--config",
        "ruff.toml",
    ]
    assert calls[0][1]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_run_typecheck_uses_bytecode_safe_python(monkeypatch) -> object:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, *, cwd, env, check) -> object:
        del cwd, check
        calls.append((command, env))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_typecheck.subprocess, "run", fake_run)

    assert run_typecheck.main() == 0
    assert calls[0][0] == [
        run_typecheck.sys.executable,
        "-B",
        "-m",
        "mypy",
        "--config-file",
        "mypy.ini",
        "--follow-imports=silent",
        *run_typecheck.TYPECHECK_TARGETS,
    ]
    assert calls[0][1]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_run_unit_tests_uses_bytecode_safe_python(monkeypatch) -> object:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_call(command, *, cwd, env) -> object:
        del cwd
        calls.append((command, env))
        return 0

    monkeypatch.setattr(run_unit_tests.subprocess, "call", fake_call)

    assert run_unit_tests.main(["-k", "smoke"]) == 0
    assert calls[0][0] == [
        run_unit_tests.sys.executable,
        "-B",
        "-m",
        "pytest",
        *run_unit_tests.PYTEST_BASE_ARGS,
        *run_unit_tests.DEFAULT_TEST_TARGETS,
        "-k",
        "smoke",
    ]
    assert calls[0][1]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_run_unit_tests_preserves_explicit_targets(monkeypatch) -> object:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_call(command, *, cwd, env) -> object:
        del cwd
        calls.append((command, env))
        return 0

    monkeypatch.setattr(run_unit_tests.subprocess, "call", fake_call)

    assert run_unit_tests.main(["tests/integration_tests/test_mocked_service_tier.py"]) == 0
    assert calls[0][0] == [
        run_unit_tests.sys.executable,
        "-B",
        "-m",
        "pytest",
        *run_unit_tests.PYTEST_BASE_ARGS,
        "tests/integration_tests/test_mocked_service_tier.py",
    ]
    assert calls[0][1]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_run_unit_tests_only_treats_real_paths_as_targets() -> None:
    assert run_unit_tests._pytest_args(["--collect-only"]) == [
        *run_unit_tests.DEFAULT_TEST_TARGETS,
        "--collect-only",
    ]
    assert run_unit_tests._pytest_args(["-m", "smoke"]) == [
        *run_unit_tests.DEFAULT_TEST_TARGETS,
        "-m",
        "smoke",
    ]
    assert run_unit_tests._pytest_args(["tests/unit_tests/test_client.py"]) == [
        "tests/unit_tests/test_client.py"
    ]


def test_run_coverage_uses_repo_owned_threshold(monkeypatch) -> object:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_call(command, *, cwd, env) -> object:
        del cwd
        calls.append((command, env))
        return 0

    monkeypatch.setattr(run_coverage.subprocess, "call", fake_call)

    assert run_coverage.main() == 0
    assert calls[0][0] == [
        run_coverage.sys.executable,
        "-B",
        "-m",
        "pytest",
        *run_coverage.PYTEST_BASE_ARGS,
        *run_coverage.BASE_TEST_EXTRA_ARGS,
        *run_coverage.DEFAULT_TEST_TARGETS,
    ]
    assert calls[0][1]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert calls[0][1]["COVERAGE_FILE"].endswith(".coverage.base")
    assert calls[1][0] == [
        run_coverage.sys.executable,
        "-B",
        "-m",
        "pytest",
        *run_coverage.PYTEST_BASE_ARGS,
        *run_coverage.BACKEND_TEST_TARGETS,
    ]
    assert calls[1][1]["COVERAGE_FILE"].endswith(".coverage.backends")
    assert calls[2][0] == [
        run_coverage.sys.executable,
        "-B",
        "-m",
        "coverage",
        "combine",
        str(run_coverage.ROOT / ".coverage.base"),
        str(run_coverage.ROOT / ".coverage.backends"),
    ]
    assert calls[3][0] == [
        run_coverage.sys.executable,
        "-B",
        "-m",
        "coverage",
        "report",
        "-m",
        f"--fail-under={run_coverage.COVERAGE_THRESHOLD}",
    ]
    assert calls[3][1]["COVERAGE_FILE"].endswith(".coverage")


def test_run_optional_feature_tests_uses_registered_targets(monkeypatch) -> object:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_call(command, *, cwd, env) -> object:
        del cwd
        calls.append((command, env))
        return 0

    monkeypatch.setattr(run_optional_feature_tests.subprocess, "call", fake_call)

    assert run_optional_feature_tests.main(["openai", "pillow", "--", "-q"]) == 0
    assert calls[0][0] == [
        run_optional_feature_tests.sys.executable,
        "-B",
        "-m",
        "pytest",
        *run_optional_feature_tests.PYTEST_BASE_ARGS,
        *run_optional_feature_tests.feature_test_targets("openai", "pillow"),
        "-q",
    ]
    assert calls[0][1]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_run_optional_feature_tests_returns_error_for_unregistered_feature_slice(capsys) -> None:
    assert run_optional_feature_tests.main(["docarray"]) == 1
    assert "No unit tests are registered" in capsys.readouterr().out


def test_check_devx_contract_runs_repo_owned_collection_checks(monkeypatch) -> object:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_call(command, *, cwd, env) -> object:
        del cwd
        calls.append((command, env))
        return 0

    monkeypatch.setattr(check_devx_contract.subprocess, "call", fake_call)
    monkeypatch.setattr(check_devx_contract, "_python_for_checks", lambda: "repo-python")
    monkeypatch.setattr(check_devx_contract, "_pytest_executable", lambda: "repo-pytest")

    assert check_devx_contract.main() == 0
    assert calls[0][0] == [
        "repo-python",
        "-B",
        "-m",
        "pytest",
        "--collect-only",
    ]
    assert calls[1][0] == ["repo-pytest", "--collect-only"]
    assert calls[2][0] == [
        "repo-python",
        "-B",
        str(check_devx_contract.ROOT / "scripts" / "run_unit_tests.py"),
        "--collect-only",
    ]
    assert all(env["PYTHONDONTWRITEBYTECODE"] == "1" for _, env in calls)


def test_check_devx_contract_prefers_bootstrapped_repo_python(monkeypatch, tmp_path) -> None:
    repo_python = tmp_path / "python"
    repo_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(check_devx_contract, "REPO_PYTHON", repo_python)
    monkeypatch.setattr(check_devx_contract, "running_inside_tox", lambda: False)
    monkeypatch.setattr(check_devx_contract.sys, "executable", "/usr/bin/python3")

    assert check_devx_contract._python_for_checks() == str(repo_python)


def test_check_devx_contract_uses_repo_pytest_wrapper_without_resolving_symlink(
    monkeypatch,
) -> object:
    class FakePath:
        def __init__(self, value: str) -> None:
            self._path = Path(value)

        @property
        def parent(self) -> Path:
            return self._path.parent

        def resolve(self) -> Path:
            return Path("/opt/hostedtoolcache/Python/3.11.15/x64/bin/python3.11")

        def __fspath__(self) -> str:
            return str(self._path)

        def __str__(self) -> str:
            return str(self._path)

    monkeypatch.setattr(
        check_devx_contract,
        "_python_for_checks",
        lambda: "/home/runner/work/ByteNew/ByteNew/.venv/bin/python",
    )
    monkeypatch.setattr(check_devx_contract, "Path", FakePath)

    assert (
        check_devx_contract._pytest_executable()
        == str(
            Path("/home/runner/work/ByteNew/ByteNew/.venv/bin")
            / f"pytest{'.exe' if check_devx_contract.os.name == 'nt' else ''}"
        )
    )


def test_check_maintainability_contract_returns_report_status(monkeypatch, capsys) -> None:
    report = check_maintainability_contract.MaintainabilityReport(
        modules=(),
        failures=(),
        ruff_returncode=0,
    )

    monkeypatch.setattr(check_maintainability_contract, "run_contract", lambda root: report)

    assert check_maintainability_contract.main() == 0
    assert "Contract passed." in capsys.readouterr().out


def test_run_integration_smoke_defaults_to_mocked_path(monkeypatch) -> object:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_call(command, *, cwd, env) -> object:
        del cwd
        calls.append((command, env))
        return 0

    monkeypatch.delenv("BYTE_RUN_LIVE_INTEGRATION", raising=False)
    monkeypatch.setattr(run_integration_smoke.subprocess, "call", fake_call)

    assert run_integration_smoke.main() == 0
    assert calls[0][0] == [
        run_integration_smoke.sys.executable,
        "-B",
        "-m",
        "pytest",
        *run_integration_smoke.PYTEST_BASE_ARGS,
        *run_integration_smoke.MOCKED_TARGETS,
    ]
    assert calls[0][1]["PYTHONDONTWRITEBYTECODE"] == "1"
