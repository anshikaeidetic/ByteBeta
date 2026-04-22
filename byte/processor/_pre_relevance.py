from __future__ import annotations

"""Relevance-scoring helpers used by context compilation."""

import json
import re
from typing import Any

from byte.processor._pre_canonicalize import _RELEVANCE_STOPWORDS, normalize_text


def _lexical_tokens(value: Any) -> set:
    text = normalize_text(value)
    return {
        _normalize_relevance_token(token)
        for token in re.findall(r"[a-z0-9_./:-]+", text)
        if len(token) >= 3
        and not token.isdigit()
        and _normalize_relevance_token(token)
        and _normalize_relevance_token(token) not in _RELEVANCE_STOPWORDS
    }


def _lexical_overlap(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap <= 0:
        return 0.0
    return overlap / max(1, min(len(left), len(right)))


def _negative_focus_penalty(value: Any) -> float:
    text = normalize_text(value)
    if re.search(
        r"\b(?:unrelated|irrelevant|not related|not about|different topic|ignore this)\b", text
    ):
        return 0.35
    return 0.0


def _normalize_relevance_token(token: str) -> str:
    token = str(token or "").strip().lower()
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif (len(token) > 4 and token.endswith("ed")) or (len(token) > 4 and token.endswith("es")):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token.strip("_-./:")


def _compact_json(value: Any) -> str:
    try:
        serialized = json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        serialized = str(value)
    if len(serialized) <= 512:
        return serialized
    return f"{serialized[:256].rstrip()} ... {serialized[-251:].lstrip()}"


__all__ = [
    "_compact_json",
    "_lexical_overlap",
    "_lexical_tokens",
    "_negative_focus_penalty",
    "_normalize_relevance_token",
]
