"""Smoke-check wheel install parity without relying on the repo working directory."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIST_DIR = ROOT / "dist"
TOP_LEVEL_PACKAGES = ("byte", "byte_server", "byte_inference", "byte_memory")


def _wheel_path(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError("No wheel artifact found in dist/.")
    return wheels[-1]


def _venv_python(venv_dir: Path) -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return venv_dir / scripts_dir / executable


def _run(command: list[str], *, cwd: Path, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def _probe_command() -> list[str]:
    packages = ", ".join(repr(name) for name in TOP_LEVEL_PACKAGES)
    return [
        "-c",
        (
            "import importlib.util; "
            f"packages = ({packages},); "
            "missing = [name for name in packages if importlib.util.find_spec(name) is None]; "
            "print('\\n'.join(missing)); "
            "raise SystemExit(1 if missing else 0)"
        ),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", default=str(DEFAULT_DIST_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dist_dir = Path(args.dist_dir).resolve()
    try:
        wheel_path = _wheel_path(dist_dir)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    with tempfile.TemporaryDirectory(prefix="byte-install-parity-") as temp_dir:
        temp_root = Path(temp_dir)
        venv_dir = temp_root / ".venv"
        create_result = _run(
            [
                sys.executable,
                "-c",
                (
                    "import venv; "
                    f"venv.EnvBuilder(with_pip=True).create(r'{venv_dir}')"
                ),
            ],
            cwd=ROOT,
        )
        if create_result.returncode != 0:
            return create_result.returncode

        venv_python = _venv_python(venv_dir)
        install_result = _run(
            [str(venv_python), "-m", "pip", "install", "--no-deps", str(wheel_path)],
            cwd=ROOT,
        )
        if install_result.returncode != 0:
            return install_result.returncode

        probe_result = _run([str(venv_python), *_probe_command()], cwd=ROOT, capture_output=True)
        if probe_result.returncode != 0:
            missing = probe_result.stdout.strip().splitlines()
            print(
                "Install parity smoke failed. Missing top-level packages: "
                + ", ".join(name for name in missing if name)
            )
            return probe_result.returncode

    print(f"Install parity smoke passed for {wheel_path.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
