"""Faithfulness verification helpers for prompt distillation."""

from __future__ import annotations

import re
from typing import Any

from byte.prompt_distillation._distillation_common import normalize_text
from byte.prompt_distillation._distillation_measure import _request_text

_NUMBER_PATTERN = re.compile(r"\b-?\d+(?:\.\d+)?%?\b")
_CRITICAL_NUMBER_CONTEXT_PATTERN = re.compile(
    r"(?is)\b(?:price|cost|amount|score|window|day|days|due|date|margin|refund|invoice|queue)\b"
    r"[^.\n]{0,24}?(?P<value>-?\d+(?:\.\d+)?%?)"
)
_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_UPPER_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_:-]{2,}\b")
_INVOICE_PATTERN = re.compile(r"\bINV-\d+\b")
_QUEUE_PATTERN = re.compile(r"\bqueue-[A-Za-z0-9_-]+\b")
_FILE_PATTERN = re.compile(r"\b[A-Za-z]:[\\/][^\s]+|\b(?:src|app|lib|tests?|packages?)[\\/][^\s:]+")
_JSON_KEY_PATTERN = re.compile(r'"([A-Za-z0-9_:-]+)"\s*:')
_CODE_SYMBOL_PATTERN = re.compile(r"\b(?:def|class|function|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)")
_INJECTION_PATTERN = re.compile(
    r"(?is)\b(ignore\s+previous|follow\s+these\s+instructions|system\s+message|developer\s+message|do\s+not\s+follow\s+previous)\b"
)
_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_PROMPT_LABEL_PATTERN = re.compile(
    r"(?is)\bprescribed\s+(?:policy|action)\s+label\s+is\s+(?P<value>[A-Z][A-Z0-9_:-]+)\b"
)
_IDENTIFIER_ASSIGNMENT_PATTERN = re.compile(
    r"(?is)\b(?P<key>invoice_id|invoice identifier|due_date|due date|follow-up due date|owner|owner label|queue identifier)"
    r"\s+(?:is|=|to)\s+(?P<value>[A-Za-z0-9_.:-]+)"
)
_JSON_ASSIGNMENT_PATTERN = re.compile(
    r"(?is)\b(?:set|and)\s+(?P<key>invoice_id|due_date|owner|owner label|queue identifier)\s+to\s+(?P<value>[A-Za-z0-9_.:-]+)"
)
_REFUND_WINDOW_PATTERN = re.compile(r"(?is)\brefunds?\s+(?:are\s+)?allowed\s+within\s+(?P<days>\d+)\s+days\b")
_REFUND_DAY_PATTERN = re.compile(
    r"(?is)\b(?:request|customer\s+asked|asked|arrived|came)\s+(?:on\s+)?day\s+(?P<day>\d+)\b"
)
_LABEL_BLOCK_PATTERN = re.compile(
    r"(?is)\blabels?\s*:\s*(?P<labels>[A-Z][A-Z0-9_:-]*(?:\s*,\s*[A-Z][A-Z0-9_:-]*)+)"
)

def verify_request_faithfulness(
    original_request: dict[str, Any],
    distilled_request: dict[str, Any],
) -> dict[str, Any]:
    return _verify_faithfulness(original_request, distilled_request)

def _verify_faithfulness(original: dict[str, Any], distilled: dict[str, Any]) -> dict[str, Any]:
    original_text = _request_text(original)
    distilled_text = _request_text(distilled)
    request_focus = _request_focus_text(original)
    original_entities = _required_entities(original_text, request_focus)
    if not original_entities:
        original_entities = _extract_relevant_entities(original_text, request_focus)
    if not original_entities:
        original_entities = _extract_entities(original_text)
    distilled_entities = _extract_entities(distilled_text)
    entity_preservation = _set_preservation(original_entities, distilled_entities)
    schema_preservation = _schema_preservation(original_text, distilled_text)
    if (original_entities and entity_preservation < 0.995) or schema_preservation < 0.995:
        verifier = "fail"
    else:
        verifier = "pass"
    faithfulness = round((entity_preservation * 0.7) + (schema_preservation * 0.3), 4)
    return {
        "faithfulness_score": faithfulness,
        "entity_preservation_rate": round(entity_preservation, 4),
        "schema_preservation_rate": round(schema_preservation, 4),
        "verifier_result": verifier,
    }


def _request_focus_text(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    for message in reversed(messages):
        content = message.get("content", "")
        if isinstance(content, list):
            rendered = " ".join(
                str(item.get("text", "") or item.get("content", "") or "")
                for item in content
                if isinstance(item, dict)
            ).strip()
            if rendered:
                return rendered
        elif str(content or "").strip():
            return str(content or "").strip()
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input") or "")
    return ""


def _required_entities(original_text: str, request_focus: str) -> set[str]:
    normalized_focus = normalize_text(request_focus)
    required: set[str] = set()

    for match in _PROMPT_LABEL_PATTERN.finditer(original_text or ""):
        value = _clean_required_entity_value(match.group("value"), uppercase=True)
        if value:
            required.add(value)

    if "invoice identifier" in normalized_focus:
        required.update(
            str(match.group(0) or "").strip().upper()
            for match in _INVOICE_PATTERN.finditer(original_text or "")
        )
    if "queue identifier" in normalized_focus:
        required.update(
            str(match.group(0) or "").strip()
            for match in _QUEUE_PATTERN.finditer(original_text or "")
        )
    if "owner" in normalized_focus and "label" in normalized_focus:
        for match in _IDENTIFIER_ASSIGNMENT_PATTERN.finditer(original_text or ""):
            key = normalize_text(match.group("key"))
            value = _clean_required_entity_value(
                match.group("value"),
                uppercase=key in {"owner", "owner label"},
            )
            if key in {"owner", "owner label"} and value:
                required.add(value)
    if "due date" in normalized_focus or "due_date" in normalized_focus:
        required.update(
            str(match.group(0) or "").strip()
            for match in _DATE_PATTERN.finditer(original_text or "")
        )

    for match in _JSON_ASSIGNMENT_PATTERN.finditer(original_text or ""):
        key = normalize_text(match.group("key"))
        value = _clean_required_entity_value(
            match.group("value"),
            uppercase=key in {"owner", "owner label"},
        )
        if key in {"invoice_id", "due_date", "owner", "owner label", "queue identifier"} and value:
            required.add(value)

    if "refund" in normalized_focus:
        window_match = _REFUND_WINDOW_PATTERN.search(original_text or "")
        if window_match:
            required.add(str(window_match.group("days") or "").strip())
        day_match = _REFUND_DAY_PATTERN.search(original_text or "")
        if day_match:
            required.add(str(day_match.group("day") or "").strip())
        label_match = _LABEL_BLOCK_PATTERN.search(original_text or "")
        if label_match:
            required.update(
                token.strip().upper()
                for token in str(label_match.group("labels") or "").split(",")
                if token.strip()
            )

    if any(
        token in normalized_focus
        for token in ("function name", "symbol", "codebase context", "normalizes the invoice")
    ):
        required.update(_relevant_code_symbols(original_text, request_focus))

    return {value for value in required if value}


def _clean_required_entity_value(value: Any, *, uppercase: bool = False) -> str:
    cleaned = str(value or "").strip().strip(".,;:")
    if uppercase:
        cleaned = cleaned.upper()
    return cleaned


def _extract_relevant_entities(original_text: str, request_focus: str) -> set[str]:
    segments = _relevant_segments(original_text, request_focus)
    if not segments:
        return set()
    return _extract_entities(" ".join(segments))


def _relevant_segments(text: str, request_focus: str, *, max_segments: int = 4) -> list[str]:
    segments = [segment.strip() for segment in _SPLIT_PATTERN.split(str(text or "")) if segment.strip()]
    if not segments:
        return []
    query_tokens = set(normalize_text(request_focus).split())
    ranked: list[tuple[float, int, str]] = []
    for index, segment in enumerate(segments):
        score = _segment_score(segment, query_tokens=query_tokens)
        if _preserve_segment(segment):
            score += 0.1
        ranked.append((score, index, segment))
    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [
        (index, segment)
        for score, index, segment in ranked[: max(1, int(max_segments or 1))]
        if score >= 0.15 or _preserve_segment(segment)
    ]
    if not selected:
        selected = [(ranked[0][1], ranked[0][2])]
    selected.sort(key=lambda item: item[0])
    return [segment for _, segment in selected]


def _relevant_code_symbols(text: str, request_focus: str) -> set[str]:
    focus_tokens = set(normalize_text(request_focus).split())
    candidates: dict[str, float] = {}
    segments = _relevant_segments(text, request_focus, max_segments=6)
    search_text = "\n".join(segments) if segments else str(text or "")
    for match in _CODE_SYMBOL_PATTERN.finditer(search_text):
        symbol = str(match.group(1) or "").strip()
        if not symbol:
            continue
        normalized_symbol = normalize_text(symbol)
        score = 0.2
        if "invoice" in normalized_symbol:
            score += 0.45
        if "normalize" in normalized_symbol:
            score += 0.45
        overlap = len(set(normalized_symbol.split()).intersection(focus_tokens))
        score += min(0.2, overlap * 0.1)
        candidates[symbol] = max(candidates.get(symbol, 0.0), score)
    if not candidates:
        return set()
    ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
    best_symbol, best_score = ranked[0]
    if best_score >= 0.6:
        return {best_symbol}
    return set()


def _extract_entities(text: str) -> set[str]:
    entities: set[str] = set()
    window_match = _REFUND_WINDOW_PATTERN.search(text or "")
    if window_match:
        entities.add(_clean_required_entity_value(window_match.group("days")))
    day_match = _REFUND_DAY_PATTERN.search(text or "")
    if day_match:
        entities.add(_clean_required_entity_value(day_match.group("day")))
    for match in _IDENTIFIER_ASSIGNMENT_PATTERN.finditer(text or ""):
        key = normalize_text(match.group("key"))
        value = _clean_required_entity_value(
            match.group("value"),
            uppercase=key in {"owner", "owner label"},
        )
        if value:
            entities.add(value)
    for match in _PROMPT_LABEL_PATTERN.finditer(text or ""):
        value = _clean_required_entity_value(match.group("value"), uppercase=True)
        if value:
            entities.add(value)
    for match in _CRITICAL_NUMBER_CONTEXT_PATTERN.finditer(text or ""):
        value = str(match.group("value") or "").strip()
        if value:
            entities.add(value)
    for pattern in (
        _DATE_PATTERN,
        _UPPER_TOKEN_PATTERN,
        _INVOICE_PATTERN,
        _QUEUE_PATTERN,
        _FILE_PATTERN,
        _JSON_KEY_PATTERN,
        _CODE_SYMBOL_PATTERN,
    ):
        for match in pattern.finditer(text or ""):
            if pattern is _CODE_SYMBOL_PATTERN or pattern is _JSON_KEY_PATTERN:
                value = str(match.group(1) or "").strip()
            else:
                value = str(match.group(0) or "").strip()
            if value:
                entities.add(value)
    return entities


def _schema_preservation(original_text: str, distilled_text: str) -> float:
    original_keys = {str(match.group(1) or "").strip() for match in _JSON_KEY_PATTERN.finditer(original_text or "")}
    if not original_keys:
        return 1.0
    distilled_keys = {str(match.group(1) or "").strip() for match in _JSON_KEY_PATTERN.finditer(distilled_text or "")}
    return _set_preservation(original_keys, distilled_keys)


def _set_preservation(left: set[str], right: set[str]) -> float:
    if not left:
        return 1.0
    return round(len(left.intersection(right)) / max(len(left), 1), 4)


def _segment_score(segment: str, *, query_tokens: set[str]) -> float:
    normalized = normalize_text(segment)
    tokens = set(normalized.split())
    overlap = len(tokens.intersection(query_tokens)) / max(len(query_tokens), 1) if query_tokens else 0.0
    score = overlap
    if _preserve_segment(segment):
        score += 0.45
    if _INJECTION_PATTERN.search(segment or ""):
        score -= 0.35
    if "byte compiled context" in normalized:
        score += 0.2
    return round(score, 4)


def _preserve_segment(segment: str) -> bool:
    text = str(segment or "")
    return bool(
        _NUMBER_PATTERN.search(text)
        or _DATE_PATTERN.search(text)
        or _UPPER_TOKEN_PATTERN.search(text)
        or _INVOICE_PATTERN.search(text)
        or _QUEUE_PATTERN.search(text)
        or _FILE_PATTERN.search(text)
        or _JSON_KEY_PATTERN.search(text)
        or _CODE_SYMBOL_PATTERN.search(text)
    )


__all__ = ["_verify_faithfulness", "verify_request_faithfulness"]
