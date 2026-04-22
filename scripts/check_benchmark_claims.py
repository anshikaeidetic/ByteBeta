"""Thin filesystem entrypoint for benchmark-claim validation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _main() -> int:
    from byte._devtools.check_benchmark_claims import main

    return main(reexec=True)


if __name__ == "__main__":
    raise SystemExit(_main())
