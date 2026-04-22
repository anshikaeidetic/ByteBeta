"""Run repo-owned tox checks without writing source-tree bytecode."""

from __future__ import annotations

import os
import subprocess
import sys

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script


def main(argv: list[str] | None = None, *, reexec: bool = False) -> int:
    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    argv = list([] if argv is None else argv)
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return subprocess.call([sys.executable, "-m", "tox", *argv], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], reexec=True))
