from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import byte


def _pyproject_payload() -> dict:
    with Path("pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_package_version_matches_public_export() -> None:
    pyproject = _pyproject_payload()

    assert pyproject["project"]["version"] == "1.0.6"
    assert byte.__version__ == "1.0.6"
    assert pyproject["project"]["version"] == byte.__version__
    assert pyproject["project"]["description"].startswith("Byte is ")
    assert pyproject["project"]["authors"][0]["name"] == "Byte"
    assert pyproject["project"]["license"] == "LicenseRef-ByteAI-Proprietary"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
    assert Path("LICENSE").exists()
    assert Path(".github/CODEOWNERS").exists()


def test_networkx_is_not_a_base_dependency() -> None:
    pyproject = _pyproject_payload()
    base_dependencies = pyproject["project"]["dependencies"]

    assert not any(str(dep).strip().startswith("networkx") for dep in base_dependencies)
    assert any(
        str(dep).strip().startswith("networkx")
        for dep in pyproject["project"]["optional-dependencies"]["huggingface"]
    )


def test_torch_dependency_policy_uses_next_major_upper_bound() -> None:
    pyproject = _pyproject_payload()
    huggingface = pyproject["project"]["optional-dependencies"]["huggingface"]
    test_extra = pyproject["project"]["optional-dependencies"]["test"]
    all_extra = pyproject["project"]["optional-dependencies"]["all"]

    assert "torch>=2.6,<3" in huggingface
    assert "transformers>=4.48,<6" in huggingface
    assert "torch>=2.6,<3" in test_extra
    assert "torchaudio>=2.6,<3" in test_extra
    assert "transformers>=4.48,<6" in test_extra
    assert "torch>=2.6,<3" in all_extra
    assert "transformers>=4.48,<6" in all_extra


def test_optional_runtime_stacks_are_not_bundled_into_dev_extra() -> None:
    pyproject = _pyproject_payload()
    dev_extra = pyproject["project"]["optional-dependencies"]["dev"]

    assert "onnxruntime>=1.20,<2; python_version>='3.11'" not in dev_extra
    assert "sqlalchemy>=2.0,<3" not in dev_extra
    assert "sqlalchemy>=2.0,<3" in pyproject["project"]["optional-dependencies"]["sql"]
    assert "sqlalchemy>=2.0,<3" in pyproject["project"]["optional-dependencies"]["test"]
    assert "openai>=2.30,<3" in pyproject["project"]["optional-dependencies"]["test"]


def test_pillow_test_extra_uses_patched_major_range() -> None:
    pyproject = _pyproject_payload()

    assert "pillow>=12.1.1,<13" in pyproject["project"]["optional-dependencies"]["test"]
    assert "pillow==12.2.0" in Path("requirements/dev-ci.txt").read_text(encoding="utf-8")
    assert "pillow==12.2.0" in Path("requirements/security-ci.txt").read_text(encoding="utf-8")
