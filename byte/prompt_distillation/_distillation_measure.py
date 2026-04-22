"""Prompt-size measurement helpers for prompt distillation."""

from __future__ import annotations

from typing import Any

from byte.processor.optimization_memory import estimate_tokens
from byte.prompt_distillation._distillation_common import (
    _DISTILLABLE_FIELD_TYPES,
    _RETRIEVAL_LIKE_FIELDS,
)


def measure_request_prompt(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    text = _request_text(request_kwargs)
    return {
        "chars": len(text),
        "tokens": estimate_tokens(text),
        "text": text,
    }

def _retrieval_context_chars(request_kwargs: dict[str, Any]) -> int:
    return sum(
        len(_stringify_value(request_kwargs.get(field)))
        for field in _RETRIEVAL_LIKE_FIELDS
        if request_kwargs.get(field) not in (None, "", [], {})
    )


def _primary_request_chars(request_kwargs: dict[str, Any]) -> int:
    messages = request_kwargs.get("messages") or []
    if messages:
        return len("\n".join(str(message.get("content", "") or "") for message in messages))
    if request_kwargs.get("prompt") is not None:
        return len(str(request_kwargs.get("prompt") or ""))
    if request_kwargs.get("input") is not None:
        return len(str(request_kwargs.get("input") or ""))
    return 0


def _request_text(request_kwargs: dict[str, Any]) -> str:
    parts: list[str] = []
    messages = request_kwargs.get("messages") or []
    if messages:
        parts.append("\n".join(str(message.get("content", "") or "") for message in messages))
    elif request_kwargs.get("prompt") is not None:
        parts.append(str(request_kwargs.get("prompt") or ""))
    elif request_kwargs.get("input") is not None:
        parts.append(str(request_kwargs.get("input") or ""))

    for field, _, _ in _DISTILLABLE_FIELD_TYPES:
        value = request_kwargs.get(field)
        if value in (None, "", [], {}):
            continue
        parts.append(f"{field}: {_stringify_value(value)}")
    return "\n".join(part for part in parts if part)


def _request_chars(request_kwargs: dict[str, Any]) -> int:
    return len(_request_text(request_kwargs))


def _stringify_value(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            rendered = _stringify_value(item)
            if rendered:
                parts.append(f"{key} {rendered}")
        return "\n".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            rendered = _stringify_value(item)
            if rendered:
                parts.append(rendered)
        return "\n".join(parts)
    return str(value)


__all__ = [
    "_primary_request_chars",
    "_request_chars",
    "_request_text",
    "_retrieval_context_chars",
    "_stringify_value",
    "measure_request_prompt",
]
