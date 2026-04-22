from __future__ import annotations

import subprocess
from pathlib import Path

from byte._devtools import run_with_repo_python


def test_main_requires_arguments(capsys) -> None:
    exit_code = run_with_repo_python.main([])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Expected a script path" in captured.err


def test_main_requires_bootstrapped_virtualenv(monkeypatch, capsys) -> None:
    missing_python = Path("Z:/missing/python.exe")
    monkeypatch.setattr(run_with_repo_python, "REPO_PYTHON", missing_python)

    exit_code = run_with_repo_python.main(["scripts/run_unit_tests.py"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Repo virtualenv is missing" in captured.err


def test_main_runs_target_with_repo_python(monkeypatch, tmp_path) -> object:
    repo_python = tmp_path / "python.exe"
    repo_python.write_text("", encoding="utf-8")
    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(command, *, cwd, check) -> object:
        calls.append((command, cwd, check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_with_repo_python, "REPO_PYTHON", repo_python)
    monkeypatch.setattr(run_with_repo_python, "ROOT", tmp_path)
    monkeypatch.setattr(run_with_repo_python.subprocess, "run", fake_run)

    exit_code = run_with_repo_python.main(["scripts/run_unit_tests.py", "-k", "smoke"])

    assert exit_code == 0
    assert calls == [
        ([str(repo_python), "scripts/run_unit_tests.py", "-k", "smoke"], tmp_path, False)
    ]
