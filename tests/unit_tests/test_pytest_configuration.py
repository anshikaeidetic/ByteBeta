from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    "tests/unit_tests/test_bootstrap_dev.py",
    "tests/unit_tests/benchmarking/test_quickstart.py",
    "tests/unit_tests/test_validation_runners.py",
]


def test_pytest_uses_repo_root_configuration_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    commands = [
        [sys.executable, "-m", "pytest", "--collect-only", *TARGETS],
        ["pytest", "--collect-only", *TARGETS],
    ]

    for command in commands:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert f"rootdir: {ROOT}" in result.stdout
        assert "configfile: pyproject.toml" in result.stdout

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'pythonpath = ["."]' not in pyproject
