from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

from byte._devtools import check_package_artifacts


def _write_wheel(path: Path, members: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        for member in members:
            wheel.writestr(member, "# packaged\n")


def _write_sdist(path: Path, members: list[str]) -> None:
    with tarfile.open(path, "w:gz") as sdist:
        for member in members:
            data = io.BytesIO(b"# packaged\n")
            info = tarfile.TarInfo(member)
            info.size = len(data.getvalue())
            sdist.addfile(info, data)


def test_package_artifact_check_passes_for_real_members(tmp_path, capsys) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    members = [
        f"byteai_cache-1.0.6/{member}"
        for member in check_package_artifacts.REQUIRED_ARTIFACT_MEMBERS
    ]
    _write_wheel(
        dist_dir / "byteai_cache-1.0.6-py3-none-any.whl",
        list(check_package_artifacts.REQUIRED_ARTIFACT_MEMBERS),
    )
    _write_sdist(dist_dir / "byteai_cache-1.0.6.tar.gz", members)

    assert check_package_artifacts.main([str(dist_dir)]) == 0
    assert "Package artifact check passed" in capsys.readouterr().out


def test_package_artifact_check_fails_on_metadata_only_wheel(tmp_path, capsys) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir / "byteai_cache-1.0.6-py3-none-any.whl",
        ["byteai_cache-1.0.6.dist-info/METADATA"],
    )
    _write_sdist(
        dist_dir / "byteai_cache-1.0.6.tar.gz",
        ["byteai_cache-1.0.6/README.md"],
    )

    assert check_package_artifacts.main([str(dist_dir)]) == 1
    output = capsys.readouterr().out
    assert "LICENSE" in output
    assert "byte/__init__.py" in output
    assert "byte_server/__init__.py" in output
