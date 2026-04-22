"""Run lightweight integration smoke checks without writing source-tree bytecode."""

from __future__ import annotations

import os
import subprocess
import sys

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script

sys.dont_write_bytecode = True

PYTEST_BASE_ARGS = ["-p", "no:cacheprovider"]
MOCKED_TARGETS = ["tests/integration_tests/test_mocked_service_tier.py"]
LIVE_TARGETS = ["tests/integration_tests/test_redis_onnx.py"]


def _default_targets() -> list[str]:
    command = [*MOCKED_TARGETS]
    if os.environ.get("BYTE_RUN_LIVE_INTEGRATION") == "1":
        command.extend(["--run-live-integration", "--integration-stack", "external", *LIVE_TARGETS])
    return command


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
            *(argv or _default_targets()),
        ],
        cwd=ROOT,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], reexec=True))
