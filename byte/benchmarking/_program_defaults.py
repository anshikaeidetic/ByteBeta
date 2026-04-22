"""Load static benchmark program defaults from packaged JSON data."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=1)
def _all_program_defaults() -> dict[str, dict[str, Any]]:
    payload = files("byte.benchmarking").joinpath("workloads/program_defaults.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(payload)
    return data if isinstance(data, dict) else {}


def load_program_defaults(name: str) -> dict[str, Any]:
    return dict(_all_program_defaults().get(name, {}))


__all__ = ["load_program_defaults"]
