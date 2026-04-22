
"""Focus extraction helpers for optimization-memory artifacts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from byte.processor._optimization_text import (
    _coerce_list,
    _lexical_overlap_score,
    _lexical_tokens,
    compact_text,
)


def _artifact_focus_segments(
    artifact_type: str,
    value: Any,
    *,
    query_tokens: set,
    max_chars: int,
    top_k: int,
) -> list[str]:
    budget = max(96, int(max_chars or 96))
    segment_budget = max(72, min(180, int(budget / max(1, min(int(top_k or 1), 3)))))
    ranked = []
    for index, item in enumerate(_coerce_list(value)[:12]):
        candidate = _artifact_focus_candidate(
            artifact_type,
            item,
            query_tokens=query_tokens,
            max_chars=segment_budget,
        )
        if not candidate:
            continue
        tokens = _lexical_tokens(candidate)
        score = _lexical_overlap_score(query_tokens, tokens)
        if score > 0 and _contains_factish_text(candidate):
            score += 0.04
        score -= _negative_focus_penalty(candidate)
        if artifact_type in {"changed_hunks", "code_hunks"} and any(
            marker in candidate.lower()
            for marker in ("error", "test", "fail", "traceback", "assert")
        ):
            score += 0.05
        if score <= 0:
            continue
        ranked.append((score, index, candidate))

    if not ranked:
        return []

    seen_tokens: list[set] = []
    segments: list[str] = []
    remaining = budget
    for _, _, candidate in sorted(ranked, key=lambda item: (item[0], -item[1]), reverse=True):
        candidate_tokens = _lexical_tokens(candidate)
        if any(_lexical_overlap_score(candidate_tokens, prior) >= 0.72 for prior in seen_tokens):
            continue
        fitted = compact_text(candidate, max_chars=min(segment_budget, remaining))
        if not fitted:
            continue
        if len(fitted) > remaining and remaining < 72:
            continue
        segments.append(fitted)
        seen_tokens.append(candidate_tokens)
        remaining -= len(fitted) + 3
        if len(segments) >= max(1, int(top_k or 1)) or remaining < 72:
            break
    return segments


def _artifact_focus_candidate(
    artifact_type: str,
    item: Any,
    *,
    query_tokens: set,
    max_chars: int,
) -> str:
    label = _artifact_item_label(item)
    snippet = _artifact_focus_snippet(
        artifact_type,
        item,
        query_tokens=query_tokens,
        max_chars=max_chars,
    )
    if artifact_type in {"repo_snapshot", "repo_summary", "workspace_summary"} and isinstance(
        item, dict
    ):
        summary_bits = []
        for field in ("repo", "workspace", "branch", "language", "framework"):
            field_value = item.get(field)
            if field_value not in (None, "", [], {}):
                summary_bits.append(f"{field}={field_value}")
        for field in ("files", "paths", "symbols", "exports"):
            values = item.get(field) or []
            if isinstance(values, list) and values:
                matched = _matching_values(values, query_tokens, limit=2)
                if matched:
                    summary_bits.append(f"{field}={', '.join(matched)}")
                else:
                    summary_bits.append(f"{field}={len(values)}")
        joined = ", ".join(summary_bits)
        return compact_text(joined, max_chars=max_chars)
    if artifact_type in {"tools", "tool_schema", "tool_schemas"} and isinstance(item, dict):
        function = item.get("function", {}) or {} if item.get("type") == "function" else item
        name = str(function.get("name") or item.get("name") or label or "").strip()
        description = str(function.get("description") or item.get("description") or "").strip()
        joined = f"{name}: {description}".strip(": ")
        return compact_text(joined, max_chars=max_chars)
    if artifact_type in {"prompt_pieces", "prompt_piece"} and isinstance(item, dict):
        piece_type = str(item.get("type") or item.get("piece_type") or "piece")
        content = item.get("content") if "content" in item else item.get("text")
        snippet = _artifact_focus_snippet(
            artifact_type, content, query_tokens=query_tokens, max_chars=max_chars
        )
        return compact_text(f"{piece_type}: {snippet}".strip(": "), max_chars=max_chars)
    if label and snippet:
        return compact_text(f"{label}: {snippet}", max_chars=max_chars)
    return compact_text(label or snippet, max_chars=max_chars)


def _artifact_focus_snippet(
    artifact_type: str,
    item: Any,
    *,
    query_tokens: set,
    max_chars: int,
) -> str:
    if isinstance(item, dict):
        raw_value = None
        preferred_fields = (
            "snippet",
            "text",
            "content",
            "page_content",
            "summary",
            "body",
            "chunk",
            "hunk",
            "range",
        )
        for field in preferred_fields:
            if item.get(field) not in (None, "", [], {}):
                raw_value = item.get(field)
                break
        if raw_value is None:
            raw_value = item
    else:
        raw_value = item

    raw_text = compact_text(raw_value, max_chars=max(512, max_chars * 4))
    if not raw_text:
        return ""
    if artifact_type in {"changed_hunks", "code_hunks"}:
        units = [line.strip() for line in str(raw_text).splitlines() if line.strip()]
    else:
        units = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+|\n+", str(raw_text))
            if segment and segment.strip()
        ]
    if not units:
        return compact_text(raw_text, max_chars=max_chars)

    ranked = []
    max_segments = (
        3
        if query_tokens.intersection(
            {"refund", "policy", "invoice", "invoice_id", "due_date", "owner", "function", "symbol"}
        )
        else 2
    )
    for index, unit in enumerate(units[:16]):
        unit_tokens = _lexical_tokens(unit)
        score = _lexical_overlap_score(query_tokens, unit_tokens)
        if score > 0 and _contains_factish_text(unit):
            score += 0.03
        score += _fact_priority_bonus(unit, query_tokens)
        score -= _negative_focus_penalty(unit)
        if score <= 0 and index > 1:
            continue
        ranked.append((score, index, compact_text(unit, max_chars=max_chars)))
    if not ranked:
        return compact_text(raw_text, max_chars=max_chars)

    chosen = []
    remaining = max_chars
    for _, _, unit in sorted(ranked, key=lambda row: (row[0], -row[1]), reverse=True):
        if any(unit == existing for existing in chosen):
            continue
        fitted = compact_text(unit, max_chars=remaining)
        if not fitted:
            continue
        chosen.append(fitted)
        remaining -= len(fitted) + 3
        if remaining < 48 or len(chosen) >= max_segments:
            break
    if not chosen:
        return compact_text(raw_text, max_chars=max_chars)
    return compact_text(" / ".join(chosen), max_chars=max_chars)


def _matching_values(values: Sequence[Any], query_tokens: set, *, limit: int = 2) -> list[str]:
    ranked = []
    for index, value in enumerate(values[:24]):
        preview = compact_text(value, max_chars=96)
        score = _lexical_overlap_score(query_tokens, _lexical_tokens(preview))
        ranked.append((score, index, preview))
    ranked = [item for item in ranked if item[0] > 0]
    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [item[2] for item in ranked[: max(1, int(limit or 1))]]


def _contains_factish_text(value: Any) -> bool:
    text = compact_text(value, max_chars=240)
    return bool(re.search(r"\b\d+\b|[/_-]|\b(?:id|ref|path|line|file|status|http)\b", text.lower()))


def _fact_priority_bonus(value: Any, query_tokens: set) -> float:
    text = compact_text(value, max_chars=240)
    lowered = text.lower()
    score = 0.0
    if query_tokens.intersection({"refund", "policy"}):
        if re.search(r"\bwithin\s+\d+\s+days\b", lowered):
            score += 0.18
        if re.search(r"\bday\s+\d+\b", lowered):
            score += 0.18
        if any(label in lowered for label in ("refund_approve", "refund_deny", "escalate")):
            score += 0.12
    if query_tokens.intersection({"invoice", "invoice_id"}) and re.search(r"\binv-\d+\b", lowered):
        score += 0.2
    if query_tokens.intersection({"due_date", "due", "date"}) and re.search(r"\b\d{4}-\d{2}-\d{2}\b", lowered):
        score += 0.18
    if query_tokens.intersection({"owner"}) and "owner" in lowered:
        score += 0.15
    if query_tokens.intersection({"function", "symbol", "normalize", "invoice"}) and re.search(
        r"\b(?:def|function|class)\s+[a-z_][a-z0-9_]*",
        lowered,
    ):
        score += 0.2
    return score


def _negative_focus_penalty(value: Any) -> float:
    text = compact_text(value, max_chars=240).lower()
    if re.search(
        r"\b(?:unrelated|irrelevant|not related|not about|different topic|ignore this)\b", text
    ):
        return 0.35
    return 0.0



def _artifact_item_label(value: Any) -> str:
    if isinstance(value, dict):
        for field in ("title", "name", "path", "source", "id", "url", "file"):
            field_value = value.get(field)
            if field_value not in (None, "", [], {}):
                return str(field_value)
    if value not in (None, "", [], {}):
        return str(value)
    return ""


def _artifact_item_snippet(value: Any) -> str:
    if isinstance(value, dict):
        for field in ("snippet", "text", "content", "page_content", "summary", "body", "chunk"):
            field_value = value.get(field)
            if field_value not in (None, "", [], {}):
                return compact_text(field_value, max_chars=120)
        return compact_text(value, max_chars=120)
    return compact_text(value, max_chars=120)


def _path_like_label(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("path") or value.get("file") or value.get("name") or value)
    return str(value or "")


def _prefix_summary(prefix: str, summary: str, *, max_chars: int) -> str:
    prefix = str(prefix or "").strip()
    summary = str(summary or "").strip()
    if not prefix:
        return compact_text(summary, max_chars=max_chars)
    if not summary:
        return compact_text(prefix, max_chars=max_chars)
    return compact_text(f"{prefix}: {summary}", max_chars=max_chars)

__all__ = [
    "_artifact_focus_segments",
    "_artifact_item_label",
    "_artifact_item_snippet",
    "_path_like_label",
    "_prefix_summary",
]
