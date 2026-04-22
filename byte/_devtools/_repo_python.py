"""Helpers for running repo-owned validation through the bootstrapped virtualenv."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = "Scripts" if os.name == "nt" else "bin"
EXECUTABLE = "python.exe" if os.name == "nt" else "python"
VENV_ROOT = (ROOT / ".venv").absolute()
REPO_PYTHON = (ROOT / ".venv" / SCRIPTS_DIR / EXECUTABLE).absolute()


def _absolute(path: os.PathLike[str] | str) -> Path:
    return Path(path).expanduser().absolute()


def using_repo_python(
    current_python: os.PathLike[str] | str | None = None,
    current_prefix: os.PathLike[str] | str | None = None,
) -> bool:
    if not REPO_PYTHON.exists():
        return False
    prefix = sys.prefix if current_prefix is None else current_prefix
    if _absolute(prefix) == VENV_ROOT:
        return True
    python = sys.executable if current_python is None else current_python
    return _absolute(python) == _absolute(REPO_PYTHON)


def running_inside_tox(env: dict[str, str] | None = None) -> bool:
    environment = os.environ if env is None else env
    return bool(environment.get("TOX_ENV_NAME") or environment.get("TOX_WORK_DIR"))


def should_reexec_with_repo_python(
    current_python: os.PathLike[str] | str | None = None,
    env: dict[str, str] | None = None,
) -> bool:
    return REPO_PYTHON.exists() and not running_inside_tox(env) and not using_repo_python(
        current_python
    )


def missing_bootstrap_message() -> str:
    return "Repo virtualenv is missing. Run ./bootstrap-dev.sh or .\\bootstrap-dev.ps1 first."


def maybe_reexec_current_script(
    script_path: os.PathLike[str] | str,
    argv: Sequence[str] | None = None,
    *,
    current_python: os.PathLike[str] | str | None = None,
    env: dict[str, str] | None = None,
) -> bool:
    if not should_reexec_with_repo_python(current_python=current_python, env=env):
        return False
    args = list(sys.argv[1:] if argv is None else argv)
    command = [str(REPO_PYTHON), str(_absolute(script_path)), *args]
    if os.name == "nt":
        environment = None if env is None else dict(env)
        raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))
    os.execv(str(REPO_PYTHON), command)
    return True


__all__ = [
    "REPO_PYTHON",
    "ROOT",
    "maybe_reexec_current_script",
    "missing_bootstrap_message",
    "running_inside_tox",
    "should_reexec_with_repo_python",
    "using_repo_python",
]
