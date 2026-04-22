from __future__ import annotations

from pathlib import Path

from byte._devtools import check_install_parity


def test_wheel_path_requires_built_wheel(tmp_path) -> None:
    try:
        check_install_parity._wheel_path(tmp_path)
    except FileNotFoundError as exc:
        assert "No wheel artifact found" in str(exc)
    else:
        raise AssertionError("Expected a missing-wheel failure.")


def test_probe_command_covers_all_top_level_packages() -> None:
    probe = check_install_parity._probe_command()

    assert probe[0] == "-c"
    for package_name in check_install_parity.TOP_LEVEL_PACKAGES:
        assert repr(package_name) in probe[1]


def test_venv_python_matches_platform_layout(tmp_path) -> None:
    venv_dir = tmp_path / ".venv"
    expected_name = "python.exe" if check_install_parity.os.name == "nt" else "python"

    assert Path(check_install_parity._venv_python(venv_dir)).name == expected_name
