from __future__ import annotations

import subprocess
from pathlib import Path

from byte._devtools import bootstrap_dev


def test_probe_python_command_reports_stderr_on_failure() -> object:
    def failing_runner(command, *, require_venv=True) -> object:
        return subprocess.CompletedProcess(command, 1, "", "boom")

    result = bootstrap_dev.probe_python_command(("python",), runner=failing_runner)

    assert result.ok is False
    assert result.reason == "boom"


def test_discover_healthy_python_prefers_first_healthy_candidate() -> object:
    seen: list[tuple[tuple[str, ...], bool]] = []

    def probe(command, *, require_venv=True) -> object:
        seen.append((tuple(command), require_venv))
        return bootstrap_dev.ProbeResult(tuple(command), tuple(command) == ("python3",))

    result = bootstrap_dev.discover_healthy_python(
        candidate_commands=[("python",), ("python3",), ("py", "-3.11")],
        probe=probe,
    )

    assert result == ("python3",)
    assert seen == [(("python",), True), (("python3",), True)]


def test_discover_healthy_python_rejects_bad_explicit_python() -> object:
    def probe(command, *, require_venv=True) -> object:
        return bootstrap_dev.ProbeResult(tuple(command), False, reason="broken")

    try:
        bootstrap_dev.discover_healthy_python(explicit_python="broken-python", probe=probe)
    except RuntimeError as exc:
        assert "broken" in str(exc)
    else:
        raise AssertionError("Expected a RuntimeError for an unusable explicit interpreter.")


def test_discover_healthy_python_allows_explicit_virtualenv_fallback() -> object:
    seen: list[tuple[tuple[str, ...], bool]] = []

    def probe(command, *, require_venv=True) -> object:
        seen.append((tuple(command), require_venv))
        return bootstrap_dev.ProbeResult(tuple(command), True)

    result = bootstrap_dev.discover_healthy_python(
        explicit_python="embedded-python",
        allow_virtualenv_fallback=True,
        probe=probe,
    )

    assert result == ("embedded-python",)
    assert seen == [(("embedded-python",), False)]


def test_run_smoke_checks_covers_packaging_tools(tmp_path, monkeypatch) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run_checked(command, *, cwd) -> None:
        commands.append(tuple(command))

    monkeypatch.setattr(bootstrap_dev, "_run_checked", fake_run_checked)
    monkeypatch.setattr(bootstrap_dev, "_supports_local_semgrep", lambda: False)
    bootstrap_dev.run_smoke_checks(venv_python=tmp_path / "python", cwd=tmp_path)

    assert any("pip-tools" in " ".join(command) for command in commands)
    assert any(command[1:] == ("-c", "import byte, byte_server; print('ok')") for command in commands)
    assert any(command[2:] == ("build", "--version") for command in commands)
    assert any(command[2:] == ("twine", "--version") for command in commands)
    assert any(command[2:] == ("bandit", "--version") for command in commands)
    assert any(command[2:] == ("pip_audit", "--version") for command in commands)
    assert any("detect-secrets" in " ".join(command) for command in commands)
    assert all(command[2:] != ("semgrep", "--version") for command in commands)


def test_run_smoke_checks_includes_semgrep_when_supported(tmp_path, monkeypatch) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run_checked(command, *, cwd) -> None:
        commands.append(tuple(command))

    monkeypatch.setattr(bootstrap_dev, "_run_checked", fake_run_checked)
    monkeypatch.setattr(bootstrap_dev, "_supports_local_semgrep", lambda: True)
    bootstrap_dev.run_smoke_checks(venv_python=tmp_path / "python", cwd=tmp_path)

    assert any(
        Path(command[0]).name in {"semgrep", "semgrep.exe"} and command[1:] == ("--version",)
        for command in commands
    )


def test_detect_secrets_version_command_uses_importlib_metadata(tmp_path) -> None:
    command = bootstrap_dev._detect_secrets_version_command(tmp_path / "python")

    assert command[1] == "-c"
    assert "importlib.metadata" in command[2]
    assert "detect-secrets" in command[2]


def test_piptools_version_command_uses_importlib_metadata(tmp_path) -> None:
    command = bootstrap_dev._piptools_version_command(tmp_path / "python")

    assert command[1] == "-c"
    assert "importlib.metadata" in command[2]
    assert "pip-tools" in command[2]


def test_install_dev_environment_uses_security_extra(tmp_path, monkeypatch) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run_checked(command, *, cwd) -> None:
        commands.append(tuple(command))

    monkeypatch.setattr(bootstrap_dev, "_run_checked", fake_run_checked)
    monkeypatch.setattr(bootstrap_dev, "_supports_local_semgrep", lambda: True)

    bootstrap_dev.install_dev_environment(venv_python=tmp_path / "python", cwd=tmp_path)

    assert commands == [
        (str(tmp_path / "python"), "-m", "pip", "install", "--upgrade", "pip"),
        (
            str(tmp_path / "python"),
            "-m",
            "pip",
            "install",
            "-c",
            str(tmp_path / "requirements" / "security-ci.txt"),
            "-e",
            bootstrap_dev.BASE_DEV_EXTRAS,
            "tox",
        ),
        (str(tmp_path / "python"), "-m", "pip", "install", "semgrep==1.157.0"),
    ]

def test_install_dev_environment_does_not_install_optional_runtime_stacks(tmp_path, monkeypatch) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run_checked(command, *, cwd) -> None:
        commands.append(tuple(command))

    monkeypatch.setattr(bootstrap_dev, "_run_checked", fake_run_checked)
    monkeypatch.setattr(bootstrap_dev, "_supports_local_semgrep", lambda: False)

    bootstrap_dev.install_dev_environment(venv_python=tmp_path / "python", cwd=tmp_path)

    assert commands == [
        (str(tmp_path / "python"), "-m", "pip", "install", "--upgrade", "pip"),
        (
            str(tmp_path / "python"),
            "-m",
            "pip",
            "install",
            "-c",
            str(tmp_path / "requirements" / "security-ci.txt"),
            "-e",
            bootstrap_dev.BASE_DEV_EXTRAS,
            "tox",
        ),
    ]


def test_install_precommit_hooks_skips_when_git_is_unavailable(tmp_path, monkeypatch) -> None:
    invoked = []

    def fake_run_checked(command, *, cwd) -> None:
        invoked.append(tuple(command))

    monkeypatch.setattr(bootstrap_dev, "_run_checked", fake_run_checked)
    monkeypatch.setattr(bootstrap_dev.shutil, "which", lambda name: None)

    bootstrap_dev.install_precommit_hooks(venv_dir=tmp_path / ".venv", cwd=tmp_path)

    assert invoked == []


def test_install_precommit_hooks_installs_pre_commit_and_pre_push(tmp_path, monkeypatch) -> None:
    invoked: list[tuple[str, ...]] = []

    def fake_run_checked(command, *, cwd) -> None:
        invoked.append(tuple(command))

    monkeypatch.setattr(bootstrap_dev, "_run_checked", fake_run_checked)
    monkeypatch.setattr(bootstrap_dev.shutil, "which", lambda name: "git.exe")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    venv_dir = tmp_path / ".venv"

    bootstrap_dev.install_precommit_hooks(venv_dir=venv_dir, cwd=tmp_path)

    assert invoked == [
        (
            str(bootstrap_dev._venv_precommit(venv_dir)),
            "install",
            "--hook-type",
            "pre-commit",
            "--hook-type",
            "pre-push",
        )
    ]


def test_create_virtualenv_falls_back_to_virtualenv_when_stdlib_venv_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    commands: list[tuple[str, ...]] = []
    shim_installs: list[tuple[Path, Path]] = []

    def fake_run_checked(command, *, cwd) -> None:
        commands.append(tuple(command))

    def fake_install_venv_shim(*, venv_python, cwd) -> None:
        shim_installs.append((venv_python, cwd))

    monkeypatch.setattr(bootstrap_dev, "_supports_stdlib_venv", lambda command, *, cwd: False)
    monkeypatch.setattr(bootstrap_dev, "_run_checked", fake_run_checked)
    monkeypatch.setattr(bootstrap_dev, "_install_venv_shim", fake_install_venv_shim)

    venv_python = bootstrap_dev.create_virtualenv(
        ("python",),
        venv_dir=tmp_path / ".venv",
        cwd=tmp_path,
    )

    assert commands == [
        ("python", "-m", "pip", "install", "--upgrade", "virtualenv"),
        ("python", "-m", "virtualenv", str(tmp_path / ".venv")),
    ]
    assert venv_python == bootstrap_dev._venv_python(tmp_path / ".venv")
    assert shim_installs == [(venv_python, tmp_path)]
