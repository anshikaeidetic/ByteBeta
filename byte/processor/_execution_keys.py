"""Hashing and serialization helpers for execution memory."""

import hashlib
import json
from typing import Any

from byte.processor.intent import extract_request_intent


def execution_request_key(
    request_kwargs: dict[str, Any] | None = None,
    *,
    model: str = "",
    scope: str = "",
    repo_fingerprint: str = "",
) -> str:
    request_kwargs = request_kwargs or {}
    intent = extract_request_intent(request_kwargs)
    payload = {
        "scope": scope or "",
        "model": model or str(request_kwargs.get("model", "") or ""),
        "repo": repo_fingerprint or "",
        "canonical_key": intent.canonical_key,
        "route_key": intent.route_key,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


def _short_digest(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
