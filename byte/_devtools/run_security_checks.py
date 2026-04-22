"""Run repo-owned security checks on the maintained production surfaces."""

from __future__ import annotations

import json
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

from byte._devtools._repo_python import ROOT, maybe_reexec_current_script
from byte._devtools.verification_targets import SECRET_SCAN_TARGETS, SECURITY_CODE_TARGETS

sys.dont_write_bytecode = True

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

SEMGREP_CONFIGS = ("p/security-audit",)
DETECT_SECRETS_ALLOWLIST = {
    ("byte/trust/calibration/byte-trust-v2.json", "Hex High Entropy String"),
}


def _run(
    command: list[str],
    *,
    capture_output: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=capture_output,
        env=env,
    )


def _bandit_commands() -> list[list[str]]:
    return [
        [sys.executable, "-m", "bandit", "-q", str(ROOT / target)]
        for target in SECURITY_CODE_TARGETS
    ]


def _pip_audit_command() -> list[str]:
    return [sys.executable, "-m", "pip_audit", "--progress-spinner", "off"]


def _tool_executable(name: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    sibling = Path(sys.executable).with_name(f"{name}{suffix}")
    if sibling.exists():
        return str(sibling)
    resolved = shutil.which(name)
    if resolved:
        return resolved
    return str(sibling)


def _semgrep_command() -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env.setdefault("SEMGREP_SEND_METRICS", "off")
    env.setdefault("SEMGREP_DISABLE_VERSION_CHECK", "1")
    command = [
        _tool_executable("semgrep"),
        "scan",
        "--metrics=off",
        "--error",
    ]
    for config in SEMGREP_CONFIGS:
        command.extend(["--config", config])
    command.extend([str(ROOT / target) for target in SECURITY_CODE_TARGETS])
    return command, env


def _semgrep_available() -> bool:
    semgrep = Path(_tool_executable("semgrep"))
    return semgrep.exists() or shutil.which("semgrep") is not None


def _semgrep_required_locally() -> bool:
    return os.name != "nt"


def _detect_secrets_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "detect_secrets",
        "scan",
        "--all-files",
        "--force-use-all-plugins",
        *[str(ROOT / target) for target in SECRET_SCAN_TARGETS if (ROOT / target).exists()],
    ]


def _run_bandit() -> None:
    for command in _bandit_commands():
        result = _run(command)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def _run_pip_audit() -> None:
    _ensure_venv_module()
    result = _run(_pip_audit_command())
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _run_semgrep() -> None:
    if not _semgrep_available():
        if not _semgrep_required_locally():
            print("Skipping semgrep on Windows; the CI security job enforces it on Linux.")
            return
        print("semgrep is required on this platform but is not installed.", file=sys.stderr)
        raise SystemExit(1)
    command, env = _semgrep_command()
    result = _run(command, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _run_detect_secrets() -> None:
    result = _run(_detect_secrets_command(), capture_output=True)
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)
    payload = json.loads(result.stdout or "{}")
    findings = {}
    for path, entries in (payload.get("results") or {}).items():
        normalized_path = path.replace("\\", "/")
        actionable_entries = [
            entry
            for entry in entries
            if (normalized_path, entry.get("type", "unknown"))
            not in DETECT_SECRETS_ALLOWLIST
        ]
        if actionable_entries:
            findings[path] = actionable_entries
    if findings:
        print("detect-secrets reported potential secrets:")
        for path, entries in sorted(findings.items()):
            kinds = ", ".join(sorted({entry.get("type", "unknown") for entry in entries}))
            print(f" - {path}: {kinds}")
        raise SystemExit(1)


def _ensure_venv_module() -> None:
    try:
        import venv  # noqa: F401
    except ModuleNotFoundError:
        for raw_site_dir in site.getsitepackages():
            site_dir = Path(raw_site_dir)
            if not site_dir.exists():
                continue
            shim_path = site_dir / "venv.py"
            if not shim_path.exists():
                shim_path.write_text(_VENV_SHIM, encoding="utf-8")
            return
        print("Unable to install a venv compatibility shim for pip-audit.", file=sys.stderr)
        raise SystemExit(1) from None


def main(*, reexec: bool = False) -> int:
    if reexec:
        maybe_reexec_current_script(sys.argv[0] or __file__)
    _run_bandit()
    _run_pip_audit()
    _run_semgrep()
    _run_detect_secrets()
    print("Security checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(reexec=True))
