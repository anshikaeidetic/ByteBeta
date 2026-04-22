"""Shared helpers for benchmark program entry modules."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from byte import Cache
from byte.processor.pre import normalize_text


def make_item(
    prompt: str,
    expected: str,
    group: str,
    variant: str,
    kind: str,
    *,
    max_tokens: int = 12,
    request_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized benchmark request item payload."""
    return {
        "prompt": prompt,
        "expected": expected,
        "group": group,
        "variant": variant,
        "kind": kind,
        "max_tokens": max_tokens,
        "request_overrides": dict(request_overrides or {}),
    }


def usage_fields(usage: Any) -> dict[str, int]:
    """Normalize usage payloads from dict or object responses."""
    if not usage:
        return {"prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0}

    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details", {}) or {}
        cached_prompt_tokens = int(details.get("cached_tokens", 0) or 0)
    else:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        if details is None:
            cached_prompt_tokens = 0
        elif isinstance(details, dict):
            cached_prompt_tokens = int(details.get("cached_tokens", 0) or 0)
        else:
            cached_prompt_tokens = int(getattr(details, "cached_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": min(cached_prompt_tokens, prompt_tokens),
        "completion_tokens": completion_tokens,
    }


def normalized_answer(text: str | None) -> str:
    """Normalize model answers, including fenced code blocks, for exact comparisons."""
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    return normalize_text(candidate)


def call_with_retry(
    func: Callable[[], Any],
    *,
    attempts: int = 3,
    base_sleep_seconds: float = 1.0,
) -> Any:
    """Run a provider call with a simple bounded retry loop."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return func()
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(base_sleep_seconds + attempt)
    if last_error is None:
        raise RuntimeError("call_with_retry exhausted without capturing an exception")
    raise last_error


def p95(values: list[float]) -> float:
    """Return the inclusive p95 for a list of latency values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(round(0.95 * len(ordered) + 0.4999) - 1, len(ordered) - 1))
    return round(ordered[index], 2)


def release_cache_tree(cache_obj: Cache) -> None:
    """Detach chained cache managers so temp directories can be removed on Windows."""
    seen = set()
    current: Cache | None = cache_obj
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current.data_manager = None
        current = getattr(current, "next_cache", None)
