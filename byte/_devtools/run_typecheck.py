"""Run repo-owned mypy checks on the maintained verification surface."""

from __future__ import annotations

import os
import subprocess
import sys

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script
from byte._devtools.verification_targets import TYPECHECK_TARGETS

sys.dont_write_bytecode = True


def main(*, reexec: bool = False) -> int:
    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-B",
        "-m",
        "mypy",
        "--config-file",
        "mypy.ini",
        "--follow-imports=silent",
        *TYPECHECK_TARGETS,
    ]
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main(reexec=True))
