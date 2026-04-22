from __future__ import annotations

from pathlib import Path

from byte._devtools.verification_targets import (
    STRICT_MYPY_MODULES,
    STRICT_TYPECHECK_TARGETS,
    TYPECHECK_TARGETS,
)

ROOT = Path(__file__).resolve().parents[2]


def test_strict_typecheck_targets_are_included_in_repo_typecheck_surface() -> None:
    for target in STRICT_TYPECHECK_TARGETS:
        assert target in TYPECHECK_TARGETS


def test_mypy_strict_sections_match_curated_runtime_surface() -> None:
    contents = (ROOT / "mypy.ini").read_text(encoding="utf-8")
    for module_name in STRICT_MYPY_MODULES:
        assert f"[mypy-{module_name}]" in contents
