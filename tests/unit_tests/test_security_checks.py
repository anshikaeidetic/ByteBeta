from __future__ import annotations

import json
import subprocess
from pathlib import Path

from byte._devtools import run_security_checks


def test_bandit_commands_cover_all_security_targets() -> None:
    commands = run_security_checks._bandit_commands()

    assert len(commands) == len(run_security_checks.SECURITY_CODE_TARGETS)
    assert all(command[:4] == [run_security_checks.sys.executable, "-m", "bandit", "-q"] for command in commands)


def test_semgrep_command_disables_metrics_and_targets_repo_files() -> None:
    command, env = run_security_checks._semgrep_command()

    assert Path(command[0]).name in {"semgrep", "semgrep.exe"}
    assert command[1:4] == ["scan", "--metrics=off", "--error"]
    assert env["SEMGREP_SEND_METRICS"] == "off"
    assert env["SEMGREP_DISABLE_VERSION_CHECK"] == "1"


def test_detect_secrets_command_scans_without_git() -> None:
    command = run_security_checks._detect_secrets_command()

    assert "--all-files" in command


def test_run_detect_secrets_reports_findings(monkeypatch, capsys) -> object:
    payload = {"results": {"README.md": [{"type": "Secret Keyword"}]}}

    def fake_run(command, *, capture_output=False, env=None) -> object:
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(run_security_checks, "_run", fake_run)

    try:
        run_security_checks._run_detect_secrets()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("detect-secrets findings should fail the security gate.")

    captured = capsys.readouterr()
    assert "README.md" in captured.out


def test_run_semgrep_skips_on_windows_when_semgrep_is_unavailable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_security_checks, "_semgrep_available", lambda: False)
    monkeypatch.setattr(run_security_checks, "_semgrep_required_locally", lambda: False)

    run_security_checks._run_semgrep()

    captured = capsys.readouterr()
    assert "Skipping semgrep on Windows" in captured.out


def test_run_semgrep_fails_when_required_platform_lacks_semgrep(monkeypatch) -> None:
    monkeypatch.setattr(run_security_checks, "_semgrep_available", lambda: False)
    monkeypatch.setattr(run_security_checks, "_semgrep_required_locally", lambda: True)

    try:
        run_security_checks._run_semgrep()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("semgrep absence should fail on supported platforms.")
