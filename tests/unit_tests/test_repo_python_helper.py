from __future__ import annotations

from pathlib import Path

from byte._devtools import _repo_python


def test_should_reexec_with_repo_python_when_bootstrapped(monkeypatch, tmp_path) -> None:
    repo_python = tmp_path / "python.exe"
    repo_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(_repo_python, "REPO_PYTHON", repo_python)
    monkeypatch.setattr(_repo_python, "running_inside_tox", lambda env=None: False)
    monkeypatch.setattr(_repo_python, "using_repo_python", lambda current_python=None: False)

    assert _repo_python.should_reexec_with_repo_python() is True


def test_should_not_reexec_inside_tox(monkeypatch, tmp_path) -> None:
    repo_python = tmp_path / "python.exe"
    repo_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(_repo_python, "REPO_PYTHON", repo_python)
    monkeypatch.setattr(_repo_python, "running_inside_tox", lambda env=None: True)

    assert _repo_python.should_reexec_with_repo_python() is False


def test_should_not_reexec_when_already_using_repo_python(monkeypatch, tmp_path) -> None:
    repo_python = tmp_path / "python.exe"
    repo_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(_repo_python, "REPO_PYTHON", repo_python)
    monkeypatch.setattr(_repo_python, "VENV_ROOT", tmp_path)
    monkeypatch.setattr(_repo_python, "running_inside_tox", lambda env=None: False)
    monkeypatch.setattr(_repo_python, "using_repo_python", lambda current_python=None: True)

    assert _repo_python.should_reexec_with_repo_python() is False


def test_using_repo_python_prefers_virtualenv_prefix_over_symlink_target(monkeypatch, tmp_path) -> None:
    repo_python = tmp_path / ".venv" / "bin" / "python"
    repo_python.parent.mkdir(parents=True)
    repo_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(_repo_python, "REPO_PYTHON", repo_python)
    monkeypatch.setattr(_repo_python, "VENV_ROOT", repo_python.parents[1].absolute())

    assert (
        _repo_python.using_repo_python(
            current_python="/opt/hostedtoolcache/Python/3.11.15/x64/bin/python3.11",
            current_prefix="/opt/hostedtoolcache/Python/3.11.15/x64",
        )
        is False
    )
    assert (
        _repo_python.using_repo_python(
            current_python=str(repo_python),
            current_prefix=str(repo_python.parents[1]),
        )
        is True
    )


def test_maybe_reexec_current_script_reports_missing_bootstrap(monkeypatch) -> None:
    monkeypatch.setattr(_repo_python, "should_reexec_with_repo_python", lambda **kwargs: False)

    assert _repo_python.maybe_reexec_current_script(Path("scripts/run_unit_tests.py")) is False
