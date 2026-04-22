"""Thin filesystem entrypoint for the optional-feature unit-test runner."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _main() -> int:
    from byte._devtools.run_optional_feature_tests import main

    return main(sys.argv[1:], reexec=True)


if __name__ == "__main__":
    raise SystemExit(_main())
