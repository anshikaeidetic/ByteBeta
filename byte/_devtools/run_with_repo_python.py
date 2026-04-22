"""Run a repo-owned script with the bootstrapped virtualenv interpreter."""

from __future__ import annotations

import subprocess
import sys

from byte._devtools._repo_python import REPO_PYTHON, ROOT, missing_bootstrap_message

sys.dont_write_bytecode = True


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Expected a script path or module command to run.", file=sys.stderr)
        return 2
    if not REPO_PYTHON.exists():
        print(missing_bootstrap_message(), file=sys.stderr)
        return 1
    result = subprocess.run([str(REPO_PYTHON), *args], cwd=ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
