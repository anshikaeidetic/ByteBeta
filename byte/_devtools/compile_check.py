from __future__ import annotations

import compileall
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("byte", "byte_server", "scripts")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="byteai-compile-") as temp_dir:
        sys.pycache_prefix = temp_dir
        success = True
        for target in TARGETS:
            target_path = REPO_ROOT / target
            success = compileall.compile_dir(
                str(target_path),
                quiet=1,
            ) and success
    if not success:
        print("Compile validation failed.")
        return 1
    print("Compile validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
