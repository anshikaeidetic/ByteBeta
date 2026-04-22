from __future__ import annotations

import subprocess
from pathlib import Path

from byte._devtools import check_locks, refresh_locks


def test_refresh_locks_builds_expected_piptools_command(monkeypatch, tmp_path) -> object:
    calls: list[list[str]] = []

    monkeypatch.setattr(refresh_locks, "ROOT", tmp_path)

    def fake_run(command, *, cwd, check) -> object:
        del cwd, check
        calls.append(command)
        output_path = tmp_path / "requirements" / "dev-ci.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "#\n"
            "#    pip-compile --output-file='C:\\Users\\Apollo\\Documents\\ByteAI (9)\\ByteAI\\requirements\\dev-ci.txt' pyproject.toml\n"
            "numpy==2.2.6\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(refresh_locks.subprocess, "run", fake_run)

    refresh_locks._compile_lock(tmp_path / "requirements" / "dev-ci.txt", ("dev", "server"))

    assert calls == [
        [
            refresh_locks.sys.executable,
            "-m",
            "piptools",
            "compile",
            "pyproject.toml",
            "--resolver=backtracking",
            "--upgrade",
            "--strip-extras",
            "--newline",
            "lf",
            "--quiet",
            "--output-file",
            "requirements/dev-ci.txt",
            "--pip-args",
            refresh_locks.LOCK_PIP_ARGS,
            "--extra",
            "dev",
            "--extra",
            "server",
        ]
    ]
    assert "--output-file='requirements/dev-ci.txt'" in (
        tmp_path / "requirements" / "dev-ci.txt"
    ).read_text(encoding="utf-8")


def test_check_locks_reports_stale_files(monkeypatch, tmp_path, capsys) -> None:
    requirements_dir = tmp_path / "requirements"
    requirements_dir.mkdir()
    stale = requirements_dir / "dev-ci.txt"
    stale.write_text("old\n", encoding="utf-8")

    monkeypatch.setattr(check_locks, "ROOT", tmp_path)
    monkeypatch.setattr(refresh_locks, "ROOT", tmp_path)
    monkeypatch.setattr(check_locks, "LOCKS", {stale: ("dev",)})

    def fake_compile(output_path: Path, extras: tuple[str, ...], **kwargs) -> None:
        del kwargs
        del extras
        output_path.write_text("new\n", encoding="utf-8")

    monkeypatch.setattr(check_locks, "_compile_lock", fake_compile)

    assert check_locks.main() == 1
    assert "Refresh the lockfiles" in capsys.readouterr().out


def test_canonicalize_lockfile_contents_strips_platform_specific_transitives() -> None:
    contents = (
        "cryptography==46.0.6\n"
        "    # via\n"
        "    #   google-auth\n"
        "    #   secretstorage\n"
        "\n"
        "colorama==0.4.6\n"
        "    # via click\n"
        "\n"
        "numpy==2.2.6\n"
        "    # via byteai-cache\n"
        "\n"
        "nvidia-cublas==13.1.0.3\n"
        "    # via torch\n"
        "\n"
        "secretstorage==3.5.0\n"
        "    # via keyring\n"
    )

    assert refresh_locks.canonicalize_lockfile_contents(contents) == (
        "cryptography==46.0.6\n"
        "    # via\n"
        "    #   google-auth\n"
        "\n"
        "numpy==2.2.6\n"
        "    # via byteai-cache\n"
        "\n"
    )


def test_check_locks_normalizes_absolute_output_path_to_repo_relative(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "requirements" / "dev-ci.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(check_locks, "ROOT", tmp_path)
    monkeypatch.setattr(refresh_locks, "ROOT", tmp_path)
    absolute_header = (
        "#\n"
        "#    pip-compile --output-file='C:\\Users\\Apollo\\Documents\\ByteAI (9)\\ByteAI\\requirements\\dev-ci.txt' pyproject.toml\n"
        "numpy==2.2.6\n"
    )

    normalized = check_locks._normalize_lockfile_contents(absolute_header, output_path)

    assert "--output-file='requirements/dev-ci.txt'" in normalized
    assert "C:\\Users\\Apollo\\Documents\\ByteAI (9)\\ByteAI" not in normalized
