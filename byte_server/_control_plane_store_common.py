"""Common serialization helpers for the Byte control-plane store."""

from __future__ import annotations

import json
import time
from typing import Any


def _now() -> float:
    return time.time()


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
