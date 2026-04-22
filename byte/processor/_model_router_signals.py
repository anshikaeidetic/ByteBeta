
"""Request-size and complexity helpers for model routing."""

from __future__ import annotations

from typing import Any

_COMPLEXITY_KEYWORDS = (
    "step by step",
    "reason carefully",
    "analyze",
    "analysis",
    "architect",
    "debug",
    "optimize",
    "root cause",
    "tradeoff",
    "compare",
    "evaluate",
    "design",
    "refactor",
    "prove",
    "derive",
    "multi step",
    "chain of thought",
)

_CHEAP_CATEGORIES = {
    "classification",
    "translation",
    "exact_answer",
    "extraction",
    "code_explanation",
}

_HARD_CATEGORIES = {
    "code_fix",
    "code_refactor",
    "test_generation",
}

_CODER_CATEGORIES = {
    "code_fix",
    "code_refactor",
    "test_generation",
    "documentation",
}

def _request_size(request_kwargs: dict[str, Any]) -> tuple:
    messages = request_kwargs.get("messages") or []
    total_chars = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total_chars += len(str(item.get("text", "") or ""))
                else:
                    total_chars += len(str(item or ""))
        else:
            total_chars += len(str(content or ""))
    if total_chars == 0:
        if request_kwargs.get("prompt") is not None:
            total_chars = len(str(request_kwargs.get("prompt") or ""))
        elif request_kwargs.get("input") is not None:
            total_chars = len(str(request_kwargs.get("input") or ""))
    return total_chars, len(messages)


def _is_hard_request(
    request_kwargs: dict[str, Any],
    *,
    category: str,
    prompt_chars: int,
    message_count: int,
    long_prompt_chars: int,
    multi_turn_threshold: int,
) -> bool:
    if prompt_chars >= long_prompt_chars:
        return True
    if message_count >= multi_turn_threshold:
        return True
    if category in _HARD_CATEGORIES:
        return True
    if category in {"comparison", "question_answer"} and prompt_chars >= max(
        400, long_prompt_chars // 2
    ):
        return True
    if category == "code_explanation" and prompt_chars >= max(600, long_prompt_chars // 2):
        return True
    if category == "instruction":
        text = _request_text(request_kwargs)
        if any(keyword in text for keyword in _COMPLEXITY_KEYWORDS):
            return True
    return False


def _request_text(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    if messages:
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "") or ""))
                else:
                    parts.append(str(item or ""))
            return " ".join(parts).lower()
        return str(content or "").lower()
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "").lower()
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input") or "").lower()
    return ""


def _slot_count(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, (list, tuple, set)):
        return len([item for item in value if str(item).strip()])
    if isinstance(value, str):
        return len([item for item in value.split("|") if item.strip()])
    return 1


def _pick_route_target(*candidates: tuple) -> tuple:
    for model_name, tier in candidates:
        if model_name:
            return model_name, tier
    return "", "default"

__all__ = [
    "_CHEAP_CATEGORIES",
    "_CODER_CATEGORIES",
    "_HARD_CATEGORIES",
    "_is_hard_request",
    "_pick_route_target",
    "_request_size",
    "_request_text",
    "_slot_count",
]
