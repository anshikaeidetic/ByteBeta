"""Validate Byte's bootstrap-first developer experience contract."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from byte._devtools._repo_python import (
    REPO_PYTHON,
    ROOT,
    maybe_reexec_current_script,
    running_inside_tox,
)

sys.dont_write_bytecode = True


def _pytest_executable() -> str:
    suffix = ".exe" if os.name == "nt" else ""
    return str(Path(_python_for_checks()).parent / f"pytest{suffix}")


def _python_for_checks() -> str:
    if REPO_PYTHON.exists() and not running_inside_tox():
        return str(REPO_PYTHON)
    return sys.executable


def _run_check(command: list[str], *, description: str) -> int:
    print(f"[devx] {description}")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.call(command, cwd=ROOT, env=env)


def main(*, reexec: bool = False) -> int:
    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    python = _python_for_checks()
    checks = [
        (
            "python -m pytest --collect-only",
            [python, "-B", "-m", "pytest", "--collect-only"],
        ),
        (
            "pytest --collect-only",
            [_pytest_executable(), "--collect-only"],
        ),
        (
            "python scripts/run_unit_tests.py --collect-only",
            [python, "-B", str(ROOT / "scripts" / "run_unit_tests.py"), "--collect-only"],
        ),
    ]

    for description, command in checks:
        exit_code = _run_check(command, description=description)
        if exit_code != 0:
            print(f"[devx] failed: {description}")
            return exit_code

    print("[devx] collection contract passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(reexec=True))
