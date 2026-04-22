"""Run repo-owned coverage checks without writing source-tree bytecode."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script

sys.dont_write_bytecode = True

DEFAULT_TEST_TARGETS = ["tests/unit_tests"]
BACKEND_TEST_TARGETS = [
    "tests/unit_tests/manager/test_object_storage.py",
    "tests/unit_tests/manager/test_redis_eviction_unit.py",
    "tests/unit_tests/manager/test_weaviate_unit.py",
    "tests/unit_tests/manager/test_dynamo_storage.py",
    "tests/unit_tests/manager/test_local_index.py",
    "tests/unit_tests/manager/test_milvusdb.py",
    "tests/unit_tests/manager/test_mongo.py",
    "tests/unit_tests/manager/test_pgvector.py",
    "tests/unit_tests/manager/test_qdrant.py",
    "tests/unit_tests/manager/test_redis.py",
    "tests/unit_tests/manager/test_redis_cache_storage.py",
    "tests/unit_tests/manager/test_usearch.py",
]
BASE_TEST_EXTRA_ARGS = [f"--ignore={target}" for target in BACKEND_TEST_TARGETS]
COVERAGE_THRESHOLD = 72
PYTEST_BASE_ARGS = [
    "-p",
    "no:cacheprovider",
    "--cov=byte",
    "--cov=byte_server",
    "--cov-report=",
]


def _coverage_env(coverage_file: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["COVERAGE_FILE"] = str(coverage_file)
    return env


def _run_command(command: list[str], *, env: dict[str, str]) -> int:
    return subprocess.call(command, cwd=ROOT, env=env)


def _pytest_command(*targets: str, extra_args: list[str] | None = None) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        *PYTEST_BASE_ARGS,
        *(extra_args or []),
        *targets,
    ]


def _delete_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _run_default_combined_coverage() -> int:
    base_file = ROOT / ".coverage.base"
    backend_file = ROOT / ".coverage.backends"
    merged_file = ROOT / ".coverage"
    for path in (base_file, backend_file, merged_file):
        _delete_if_exists(path)

    base_exit = _run_command(
        _pytest_command(*DEFAULT_TEST_TARGETS, extra_args=BASE_TEST_EXTRA_ARGS),
        env=_coverage_env(base_file),
    )
    if base_exit != 0:
        return base_exit

    backend_exit = _run_command(
        _pytest_command(*BACKEND_TEST_TARGETS),
        env=_coverage_env(backend_file),
    )
    if backend_exit != 0:
        return backend_exit

    combine_exit = _run_command(
        [
            sys.executable,
            "-B",
            "-m",
            "coverage",
            "combine",
            str(base_file),
            str(backend_file),
        ],
        env=_coverage_env(merged_file),
    )
    if combine_exit != 0:
        return combine_exit

    return _run_command(
        [
            sys.executable,
            "-B",
            "-m",
            "coverage",
            "report",
            "-m",
            f"--fail-under={COVERAGE_THRESHOLD}",
        ],
        env=_coverage_env(merged_file),
    )


def main(argv: list[str] | None = None, *, reexec: bool = False) -> int:
    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    argv = list([] if argv is None else argv)
    if argv:
        return _run_command(
            _pytest_command(*(argv or DEFAULT_TEST_TARGETS)),
            env=_coverage_env(ROOT / ".coverage"),
        )
    return _run_default_combined_coverage()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], reexec=True))
