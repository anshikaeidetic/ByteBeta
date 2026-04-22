"""Bootstrap a local ByteAI Cache development environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ProbeRunner = Callable[..., subprocess.CompletedProcess[str]]
BASE_DEV_EXTRAS = ".[dev,security,observability,server]"
_VENV_SHIM = """from __future__ import annotations

import subprocess
import sys


def create(
    env_dir,
    system_site_packages=False,
    clear=False,
    symlinks=False,
    with_pip=False,
    prompt=None,
    upgrade=False,
    upgrade_deps=False,
    scm_ignore_files=frozenset(),
):
    command = [sys.executable, "-m", "virtualenv", str(env_dir)]
    if clear:
        command.append("--clear")
    if system_site_packages:
        command.append("--system-site-packages")
    if prompt:
        command.extend(["--prompt", str(prompt)])
    subprocess.run(command, check=True)


class EnvBuilder:
    def __init__(
        self,
        system_site_packages=False,
        clear=False,
        symlinks=False,
        with_pip=False,
        prompt=None,
        upgrade=False,
        upgrade_deps=False,
        scm_ignore_files=frozenset(),
    ):
        self.system_site_packages = system_site_packages
        self.clear = clear
        self.symlinks = symlinks
        self.with_pip = with_pip
        self.prompt = prompt
        self.upgrade = upgrade
        self.upgrade_deps = upgrade_deps
        self.scm_ignore_files = scm_ignore_files

    def create(self, env_dir):
        create(
            env_dir,
            system_site_packages=self.system_site_packages,
            clear=self.clear,
            symlinks=self.symlinks,
            with_pip=self.with_pip,
            prompt=self.prompt,
            upgrade=self.upgrade,
            upgrade_deps=self.upgrade_deps,
            scm_ignore_files=self.scm_ignore_files,
        )
"""


@dataclass(frozen=True)
class ProbeResult:
    command: tuple[str, ...]
    ok: bool
    reason: str = ""


def _default_candidate_commands() -> list[tuple[str, ...]]:
    candidates: list[tuple[str, ...]] = []
    if sys.executable:
        candidates.append((sys.executable,))
    for command in ("python", "python3"):
        resolved = shutil.which(command)
        if resolved:
            candidates.append((resolved,))
    py_launcher = shutil.which("py")
    if py_launcher:
        for version in ("3.12", "3.11", "3.10"):
            candidates.append((py_launcher, f"-{version}"))
        candidates.append((py_launcher,))
    unique: list[tuple[str, ...]] = []
    seen = set()
    for item in candidates:
        key = tuple(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _run_probe(
    command: Sequence[str],
    *,
    require_venv: bool = True,
) -> subprocess.CompletedProcess[str]:
    probe_imports = "import encodings, pip"
    if require_venv:
        probe_imports += ", venv"
    return subprocess.run(
        [*command, "-c", f"{probe_imports}; print('ok')"],
        capture_output=True,
        text=True,
        check=False,
    )


def probe_python_command(
    command: Sequence[str],
    *,
    runner: ProbeRunner = _run_probe,
    require_venv: bool = True,
) -> ProbeResult:
    try:
        result = runner(command, require_venv=require_venv)
    except OSError as exc:
        return ProbeResult(tuple(command), False, reason=str(exc))
    if result.returncode != 0:
        reason = (result.stderr or result.stdout or "").strip() or "probe failed"
        return ProbeResult(tuple(command), False, reason=reason)
    return ProbeResult(tuple(command), True)


def discover_healthy_python(
    *,
    explicit_python: str | None = None,
    candidate_commands: Iterable[Sequence[str]] | None = None,
    allow_virtualenv_fallback: bool = False,
    probe: Callable[..., ProbeResult] = probe_python_command,
) -> tuple[str, ...]:
    if explicit_python:
        explicit = (explicit_python,)
        result = probe(explicit, require_venv=not allow_virtualenv_fallback)
        if result.ok:
            return result.command
        raise RuntimeError(f"Explicit Python interpreter is not usable: {result.reason}")

    commands = list(candidate_commands or _default_candidate_commands())
    failures: list[str] = []
    for command in commands:
        result = probe(command, require_venv=True)
        if result.ok:
            return result.command
        failures.append(f"{' '.join(command)} -> {result.reason}")
    raise RuntimeError(
        "Unable to find a healthy Python interpreter. Checked:\n" + "\n".join(failures)
    )


def _venv_python(venv_dir: Path) -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return venv_dir / scripts_dir / executable


def _venv_precommit(venv_dir: Path) -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "pre-commit.exe" if os.name == "nt" else "pre-commit"
    return venv_dir / scripts_dir / executable


def _venv_tool(venv_python: Path, executable: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    return str(venv_python.parent / f"{executable}{suffix}")


def _constraint_file(cwd: Path, filename: str) -> Path:
    return cwd / "requirements" / filename


def _run_checked(command: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(list(command), cwd=str(cwd), check=True)


def _supports_stdlib_venv(command: Sequence[str], *, cwd: Path) -> bool:
    result = subprocess.run(
        [*command, "-c", "import venv; print('ok')"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _venv_site_packages(venv_python: Path, *, cwd: Path) -> Path:
    result = subprocess.run(
        [str(venv_python), "-c", "import site; print(site.getsitepackages()[0])"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _install_venv_shim(*, venv_python: Path, cwd: Path) -> None:
    shim_path = _venv_site_packages(venv_python, cwd=cwd) / "venv.py"
    shim_path.write_text(_VENV_SHIM, encoding="utf-8")


def create_virtualenv(command: Sequence[str], *, venv_dir: Path, cwd: Path) -> Path:
    supports_stdlib_venv = _supports_stdlib_venv(command, cwd=cwd)
    if supports_stdlib_venv:
        _run_checked([*command, "-m", "venv", str(venv_dir)], cwd=cwd)
    else:
        _run_checked([*command, "-m", "pip", "install", "--upgrade", "virtualenv"], cwd=cwd)
        _run_checked([*command, "-m", "virtualenv", str(venv_dir)], cwd=cwd)
    venv_python = _venv_python(venv_dir)
    if not supports_stdlib_venv:
        _install_venv_shim(venv_python=venv_python, cwd=cwd)
    return venv_python


def install_dev_environment(*, venv_python: Path, cwd: Path) -> None:
    constraint_file = _constraint_file(cwd, "security-ci.txt")
    _run_checked([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], cwd=cwd)
    _run_checked(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "-c",
            str(constraint_file),
            "-e",
            BASE_DEV_EXTRAS,
            "tox",
        ],
        cwd=cwd,
    )
    if _supports_local_semgrep():
        _run_checked(
            [str(venv_python), "-m", "pip", "install", "semgrep==1.157.0"],
            cwd=cwd,
        )


def _supports_local_semgrep() -> bool:
    return os.name != "nt"


def _detect_secrets_version_command(venv_python: Path) -> list[str]:
    return [
        str(venv_python),
        "-c",
        "from importlib.metadata import version; print(version('detect-secrets'))",
    ]


def _piptools_version_command(venv_python: Path) -> list[str]:
    return [
        str(venv_python),
        "-c",
        "from importlib.metadata import version; print(version('pip-tools'))",
    ]


def run_smoke_checks(*, venv_python: Path, cwd: Path) -> None:
    smoke_commands = [
        [str(venv_python), "-m", "tox", "--version"],
        _piptools_version_command(venv_python),
        [str(venv_python), "-m", "ruff", "--version"],
        [str(venv_python), "-m", "mypy", "--version"],
        [str(venv_python), "-m", "pytest", "--version"],
        [str(venv_python), "-c", "import byte, byte_server; print('ok')"],
        [str(venv_python), "-m", "build", "--version"],
        [str(venv_python), "-m", "twine", "--version"],
        [str(venv_python), "-m", "bandit", "--version"],
        [str(venv_python), "-m", "pip_audit", "--version"],
        _detect_secrets_version_command(venv_python),
    ]
    if _supports_local_semgrep():
        smoke_commands.append([_venv_tool(venv_python, "semgrep"), "--version"])
    for command in smoke_commands:
        _run_checked(command, cwd=cwd)


def install_precommit_hooks(*, venv_dir: Path, cwd: Path) -> None:
    if not (cwd / ".git").exists() or shutil.which("git") is None:
        print("Skipping pre-commit hook installation because git is unavailable.")
        return
    precommit = _venv_precommit(venv_dir)
    _run_checked(
        [
            str(precommit),
            "install",
            "--hook-type",
            "pre-commit",
            "--hook-type",
            "pre-push",
        ],
        cwd=cwd,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="explicit_python", default=None)
    parser.add_argument("--venv-dir", default=".venv")
    parser.add_argument("--skip-pre-commit", action="store_true")
    parser.add_argument("--skip-smoke-checks", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    venv_dir = (repo_root / args.venv_dir).resolve()
    command = discover_healthy_python(
        explicit_python=args.explicit_python,
        allow_virtualenv_fallback=bool(args.explicit_python),
    )
    venv_python = create_virtualenv(command, venv_dir=venv_dir, cwd=repo_root)
    install_dev_environment(venv_python=venv_python, cwd=repo_root)
    if not args.skip_smoke_checks:
        run_smoke_checks(venv_python=venv_python, cwd=repo_root)
    if not args.skip_pre_commit:
        install_precommit_hooks(venv_dir=venv_dir, cwd=repo_root)
    print(f"Bootstrapped development environment in {venv_dir}")
    print(f"Use {venv_python} for local checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
