"""Validate that built package artifacts contain the shipped ByteAI code."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIST_DIR = ROOT / "dist"
REQUIRED_ARTIFACT_MEMBERS = (
    "LICENSE",
    "byte/__init__.py",
    "byte_server/__init__.py",
    "byte_inference/__init__.py",
    "byte_memory/__init__.py",
    "byte/benchmarking/workloads/codebase_context.json",
    "byte/trust/calibration/byte-trust-v2.json",
)


def _artifact_paths(dist_dir: Path) -> tuple[Path, Path]:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if not wheels:
        raise FileNotFoundError("No wheel artifact found in dist/.")
    if not sdists:
        raise FileNotFoundError("No source distribution artifact found in dist/.")
    return wheels[-1], sdists[-1]


def _missing_required_members(names: set[str]) -> list[str]:
    missing: list[str] = []
    for required in REQUIRED_ARTIFACT_MEMBERS:
        if not any(name.endswith(required) for name in names):
            missing.append(required)
    return missing


def validate_wheel_artifact(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as wheel:
        return _missing_required_members(set(wheel.namelist()))


def validate_sdist_artifact(sdist_path: Path) -> list[str]:
    with tarfile.open(sdist_path, "r:gz") as sdist:
        return _missing_required_members(set(sdist.getnames()))


def main(argv: list[str] | None = None) -> int:
    argv = list([] if argv is None else argv)
    dist_dir = Path(argv[0]).resolve() if argv else DEFAULT_DIST_DIR
    try:
        wheel_path, sdist_path = _artifact_paths(dist_dir)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    failures: list[str] = []
    missing_wheel = validate_wheel_artifact(wheel_path)
    if missing_wheel:
        failures.append(
            f"Wheel artifact {wheel_path.name} is missing: {', '.join(sorted(missing_wheel))}"
        )
    missing_sdist = validate_sdist_artifact(sdist_path)
    if missing_sdist:
        failures.append(
            f"Source distribution {sdist_path.name} is missing: {', '.join(sorted(missing_sdist))}"
        )

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(f"Package artifact check passed for {wheel_path.name} and {sdist_path.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
