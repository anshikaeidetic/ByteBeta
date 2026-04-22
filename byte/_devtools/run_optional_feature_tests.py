"""Run the unit-test slice for one or more optional Byte feature stacks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script
from byte._testing.optional_deps import feature_test_targets

sys.dont_write_bytecode = True

PYTEST_BASE_ARGS = ["-p", "no:cacheprovider"]


def _split_passthrough_args(argv: list[str] | None) -> tuple[list[str], list[str]]:
    tokens = list([] if argv is None else argv)
    if "--" not in tokens:
        return tokens, []
    split_index = tokens.index("--")
    return tokens[:split_index], tokens[split_index + 1 :]


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    feature_args, pytest_args = _split_passthrough_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "features",
        nargs="+",
        help="Optional feature names to satisfy for the targeted test slice.",
    )
    return parser.parse_args(feature_args), pytest_args


def _pytest_targets(features: list[str]) -> list[str]:
    targets = feature_test_targets(*features)
    if not targets:
        joined = ", ".join(features)
        raise ValueError(f"No unit tests are registered for optional feature selection: {joined}")
    return targets


def main(argv: list[str] | None = None, *, reexec: bool = False) -> int:
    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    args, pytest_args = parse_args(argv)

    try:
        targets = _pytest_targets(args.features)
    except ValueError as exc:
        print(str(exc))
        return 1

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        *PYTEST_BASE_ARGS,
        *targets,
        *pytest_args,
    ]
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], reexec=True))
