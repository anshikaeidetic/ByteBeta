from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from byte import __version__

ROOT = Path(__file__).resolve().parents[2]


def test_package_version_matches_runtime_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == __version__


def test_docs_release_matches_runtime_version() -> None:
    contents = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    assert f'release = "{__version__}"' in contents
