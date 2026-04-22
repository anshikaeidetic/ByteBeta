"""Verify repo-owned lockfiles are up to date."""

from __future__ import annotations

import difflib
import tempfile
from pathlib import Path

from byte._devtools.refresh_locks import (
    LOCKS,
    ROOT,
    _compile_lock,
    canonicalize_lockfile_contents,
    normalize_output_file_reference,
)


def _normalize_lockfile_contents(contents: str, output_path: Path) -> str:
    normalized = normalize_output_file_reference(contents, output_path)
    return canonicalize_lockfile_contents(normalized)


def main() -> int:
    mismatches: list[str] = []
    with tempfile.TemporaryDirectory(prefix="byte-lock-check-") as temp_dir:
        temp_root = Path(temp_dir)
        for output_path, extras in LOCKS.items():
            temp_path = temp_root / output_path.name
            _compile_lock(temp_path, extras, header_output_path=output_path)
            committed = _normalize_lockfile_contents(
                output_path.read_text(encoding="utf-8"),
                output_path,
            )
            generated = _normalize_lockfile_contents(
                temp_path.read_text(encoding="utf-8"),
                output_path,
            )
            if committed == generated:
                continue
            diff = "".join(
                difflib.unified_diff(
                    committed.splitlines(keepends=True),
                    generated.splitlines(keepends=True),
                    fromfile=str(output_path.relative_to(ROOT)),
                    tofile=f"{output_path.relative_to(ROOT)} (expected)",
                )
            )
            mismatches.append(diff)
    if mismatches:
        print(
            "Lockfile check failed. Refresh the lockfiles with `python scripts/refresh_locks.py`."
        )
        for diff in mismatches:
            print(diff, end="" if diff.endswith("\n") else "\n")
        return 1
    print("Lockfile check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
