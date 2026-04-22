
"""Text normalization, digesting, and lexical helpers for optimization memory."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any

_RELEVANCE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "using",
    "use",
    "question",
    "answer",
    "context",
    "retrieval",
    "available",
    "docs",
    "doc",
    "document",
    "documents",
    "support",
    "article",
    "articles",
    "repo",
    "workspace",
}

def stable_digest(value: Any) -> str:
    normalized = _json_safe(value)
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def estimate_tokens(value: Any) -> int:
    text = compact_text(value, max_chars=20000)
    if not text:
        return 0
    return max(1, round(len(text) / 4.0))


def compact_text(value: Any, *, max_chars: int = 480) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        text = " ".join(value.replace("\r", "\n").split())
    else:
        text = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=True, default=str)
    if len(text) <= max_chars:
        return text
    head = max(48, max_chars // 2)
    tail = max(24, max_chars - head - 5)
    return f"{text[:head].rstrip()} ... {text[-tail:].lstrip()}"

def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    return [value]


def _lexical_tokens(value: Any) -> set:
    text = compact_text(value, max_chars=4000).lower()
    return {
        _normalize_relevance_token(token)
        for token in re.findall(r"[a-z0-9_./:-]+", text)
        if len(token) >= 3
        and not token.isdigit()
        and _normalize_relevance_token(token)
        and _normalize_relevance_token(token) not in _RELEVANCE_STOPWORDS
    }


def _lexical_overlap_score(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection <= 0:
        return 0.0
    return intersection / max(1, min(len(left), len(right)))


def _normalize_relevance_token(token: str) -> str:
    token = str(token or "").strip().lower()
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif (len(token) > 4 and token.endswith("ed")) or (len(token) > 4 and token.endswith("es")):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token.strip("_-./:")


def _success_rate(entry: dict[str, Any]) -> float:
    successes = int(entry.get("successes", 0) or 0)
    failures = int(entry.get("failures", 0) or 0)
    total = successes + failures
    if total <= 0:
        return 0.0
    return successes / total


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, Counter):
        return {str(key): int(val) for key, val in value.items()}
    if isinstance(value, defaultdict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)

__all__ = [
    "_coerce_list",
    "_json_safe",
    "_lexical_overlap_score",
    "_lexical_tokens",
    "_normalize_relevance_token",
    "_success_rate",
    "compact_text",
    "estimate_tokens",
    "stable_digest",
]
