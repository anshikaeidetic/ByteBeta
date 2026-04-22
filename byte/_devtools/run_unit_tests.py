"""Run repo-owned unit tests without writing source-tree bytecode."""

from __future__ import annotations

import os
import subprocess
import sys

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script

sys.dont_write_bytecode = True

DEFAULT_TEST_TARGETS = ["tests/unit_tests"]
PYTEST_BASE_ARGS = ["-p", "no:cacheprovider"]
PYTEST_VALUE_OPTIONS = {
    "-c",
    "-k",
    "-m",
    "-n",
    "-o",
    "-p",
    "--basetemp",
    "--capture",
    "--confcutdir",
    "--cov",
    "--cov-config",
    "--cov-context",
    "--cov-fail-under",
    "--cov-report",
    "--deselect",
    "--durations",
    "--ignore",
    "--import-mode",
    "--junitxml",
    "--log-cli-level",
    "--log-file",
    "--log-file-level",
    "--log-level",
    "--maxfail",
    "--override-ini",
    "--rootdir",
    "--tb",
}


def _looks_like_test_target(token: str) -> bool:
    if "::" in token:
        return True
    if token.endswith(".py"):
        return True
    return (ROOT / token).exists()


def _pytest_args(argv: list[str]) -> list[str]:
    if not argv:
        return [*DEFAULT_TEST_TARGETS]

    explicit_target = False
    expect_value = False
    passthrough_mode = False

    for token in argv:
        if passthrough_mode:
            explicit_target = True
            break
        if expect_value:
            expect_value = False
            continue
        if token == "--":
            passthrough_mode = True
            continue
        if token in PYTEST_VALUE_OPTIONS:
            expect_value = True
            continue
        if token.startswith("-"):
            continue
        if _looks_like_test_target(token):
            explicit_target = True
            break

    if explicit_target:
        return argv
    return [*DEFAULT_TEST_TARGETS, *argv]


def main(argv: list[str] | None = None, *, reexec: bool = False) -> int:
    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    argv = list([] if argv is None else argv)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.call(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            *PYTEST_BASE_ARGS,
            *_pytest_args(argv),
        ],
        cwd=ROOT,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], reexec=True))
