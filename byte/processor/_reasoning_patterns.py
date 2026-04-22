"""Pattern resolvers for deterministic and grounded reasoning shortcuts."""

from __future__ import annotations

import re
from typing import Any

from byte.processor._reasoning_store import ReasoningShortcut
from byte.processor.pre import normalize_text

_PRICE_PATTERNS = (
    re.compile(
        r"(?is)\b(?:selling|sale|sell|product)\s+price\s*(?:=|:)\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"
    ),
    re.compile(
        r"(?is)\b(?:selling|sale)\s+price(?:\s+is)?\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"
    ),
    re.compile(r"(?is)\bprice\s*(?:=|:)\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"(?is)\bprice(?:\s+is)?\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"(?is)\bsold\b.{0,24}?\bat\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"(?is)\bsells?\b.{0,48}?\bfor\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"(?is)\brevenue\s*(?:=|:|is)?\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"),
)
_COST_PATTERNS = (
    re.compile(
        r"(?is)\b(?:production|marketing|shipping|operating|labor|labour|materials?|support|service|overhead)\s+cost\s*(?:=|:)\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"
    ),
    re.compile(
        r"(?is)\b(?:production|marketing|shipping|operating|labor|labour|materials?|support|service|overhead)\s*=\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"
    ),
)
_POLICY_LABEL_PATTERN = re.compile(r"(?is)\blabels?\s*:\s*(?P<labels>[^\n]+)")
_POLICY_LABEL_SET_PATTERN = re.compile(
    r"(?is)\b(?:one|single)\s+(?:label|action)\s+(?:from|out\s+of)\s*\{(?P<labels>[^}]+)\}"
)
_POLICY_WINDOW_PATTERN = re.compile(
    r"(?is)\brefunds?\s+(?:are\s+)?allowed\s+within\s+(?P<days>\d+)\s+days?"
)
_POLICY_DAY_PATTERN = re.compile(r"(?is)\bday\s*(?P<day>\d+)\b")
_CAPITAL_QUERY_PATTERNS = (
    re.compile(
        r"(?is)\bwhat\s+is\s+the\s+capital(?:\s+city)?\s+of\s+(?P<entity>[a-z][a-z .'-]{1,60})"
    ),
    re.compile(
        r"(?is)\bwhich\s+city\s+is\s+the\s+capital(?:\s+city)?\s+of\s+(?P<entity>[a-z][a-z .'-]{1,60})"
    ),
    re.compile(r"(?is)\bname\s+the\s+capital(?:\s+city)?\s+of\s+(?P<entity>[a-z][a-z .'-]{1,60})"),
)
_POLICY_APPROVE_HINTS = ("approve", "approved", "accept", "allow", "yes", "refund_approve")
_POLICY_DENY_HINTS = ("deny", "denied", "reject", "decline", "no", "refund_deny")
_POLICY_ESCALATE_HINTS = ("escalate", "review", "manual")
_EXACT_TOKEN_PATTERNS = (
    re.compile(
        r"(?is)(?:return|reply|respond|answer)\s+with\s+exactly\s+(?P<token>[A-Za-z0-9_.:-]+)"
    ),
    re.compile(r"(?is)exactly\s+(?P<token>[A-Za-z0-9_.:-]+)\s+and\s+nothing\s+else"),
)
_GENERIC_EXACT_TOKENS = {"one", "label", "labels", "token", "word", "json", "yaml", "csv"}
_AUX_CONTEXT_FIELDS = (
    "byte_retrieval_context",
    "byte_document_context",
    "byte_support_articles",
    "byte_tool_result_context",
    "byte_repo_summary",
    "byte_repo_snapshot",
    "byte_changed_files",
    "byte_changed_hunks",
    "byte_prompt_pieces",
)
_GROUNDED_TARGET_RULES = (
    (
        "primary_service",
        ("validated primary service identifier", "primary service identifier", "primary service"),
    ),
    ("fallback_service", ("fallback service identifier", "fallback service")),
    ("queue_name", ("queue identifier", "queue name", "queue")),
    ("policy_label", ("architecture policy label", "policy label")),
    ("invoice_id", ("invoice identifier", "invoice id")),
    ("amount", ("total amount due", "amount due", "open amount", "total amount")),
    ("owner", ("owner label", "owner")),
    (
        "cause",
        (
            "incident root-cause label",
            "incident root cause label",
            "root-cause label",
            "root cause label",
            "cause label",
        ),
    ),
    ("due_date", ("follow-up due date", "due date")),
    (
        "action_label",
        ("prescribed action label", "final action label", "action label", "next action label"),
    ),
)
_INVOICE_PATTERN = re.compile(r"\bINV-\d+\b")
_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_AMOUNT_PATTERN = re.compile(r"\$\d[\d,]*(?:\.\d+)?")
_OWNER_PATTERNS = (
    re.compile(r"(?is)\bbelongs\s+to\s+owner\s+(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"),
    re.compile(r"(?is)\bowner(?:\s+label)?\s*(?:is|:)?\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"),
)
_CAUSE_PATTERNS = (
    re.compile(
        r"(?is)\broot[\s-]+cause\s+label(?:\s+for\s+case\s+\d+)?\s*(?:is|:)?\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"
    ),
    re.compile(
        r"(?is)\bcause\s+label(?:\s+for\s+case\s+\d+)?\s*(?:is|:)\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"
    ),
)
_ACTION_PATTERNS = (
    re.compile(r"(?is)\baction\s+label\s+should\s+be\s+(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"),
    re.compile(
        r"(?is)\bprescribed\s+action\s*(?:label\s*)?(?:is|:)?\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"
    ),
    re.compile(
        r"(?is)\bfinal\s+action\s+label(?:\s+should\s+be|\s+is|:)\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"
    ),
)
_PRIMARY_SERVICE_PATTERNS = (
    re.compile(r"(?is)api\s+gateway\s*->\s*(?P<value>[A-Za-z0-9_-]+)\s*->"),
    re.compile(r"(?is)\bfor\s+(?P<value>svc-\d+)\b"),
)
_FALLBACK_SERVICE_PATTERNS = (
    re.compile(r"(?is)\bfallback\s+traffic\s+routes\s+to\s+(?P<value>[A-Za-z0-9_-]+)\b"),
    re.compile(r"(?is)\bdiverts\s+to\s+(?P<value>[A-Za-z0-9_-]+)\b"),
)
_QUEUE_PATTERNS = (
    re.compile(r"(?is)->\s*(?P<value>queue-[A-Za-z0-9_-]+)\b"),
    re.compile(
        r"(?is)\bqueue(?:\s+identifier|\s+name)?\s*(?:is|:)?\s*(?P<value>queue-[A-Za-z0-9_-]+)\b"
    ),
)
_POLICY_LABEL_PATTERNS = (
    re.compile(r"(?is)\bpolicy\s+label\s*(?:is|:)?\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"),
)
_CURATED_CAPITAL_CITIES = {
    "france": "Paris",
    "italy": "Rome",
    "japan": "Tokyo",
    "canada": "Ottawa",
    "australia": "Canberra",
    "spain": "Madrid",
    "germany": "Berlin",
    "portugal": "Lisbon",
    "austria": "Vienna",
    "ireland": "Dublin",
    "norway": "Oslo",
    "sweden": "Stockholm",
    "finland": "Helsinki",
    "denmark": "Copenhagen",
    "belgium": "Brussels",
    "switzerland": "Bern",
    "poland": "Warsaw",
    "greece": "Athens",
    "czech republic": "Prague",
    "hungary": "Budapest",
}

def _solve_profit_margin(raw_text: str, normalized: str) -> str:
    if "profit margin" not in normalized and "margin percentage" not in normalized:
        return ""
    price = None
    for pattern in _PRICE_PATTERNS:
        match = pattern.search(raw_text)
        if match:
            price = _number(match.group("value"))
            break
    if price is None or price <= 0:
        return ""
    costs = []
    for pattern in _COST_PATTERNS:
        for match in pattern.finditer(raw_text):
            value = _number(match.group("value"))
            if value is not None:
                costs.append(value)
    if not costs:
        return ""
    margin = ((price - sum(costs)) / price) * 100.0
    return _format_percentage(margin)


def _solve_refund_policy_label(raw_text: str, normalized: str) -> str:
    if "refund" not in normalized or "day" not in normalized:
        return ""
    labels = _extract_labels(raw_text)
    if not labels:
        return ""
    window_match = _POLICY_WINDOW_PATTERN.search(raw_text)
    if not window_match:
        return ""
    threshold = int(window_match.group("days"))
    days = [int(match.group("day")) for match in _POLICY_DAY_PATTERN.finditer(raw_text)]
    if not days:
        return ""
    request_day = days[-1]
    approve_label = _best_label(labels, _POLICY_APPROVE_HINTS)
    deny_label = _best_label(labels, _POLICY_DENY_HINTS)
    escalate_label = _best_label(labels, _POLICY_ESCALATE_HINTS)
    if approve_label and deny_label:
        return approve_label if request_day <= threshold else deny_label
    if request_day > threshold and deny_label:
        return deny_label
    if request_day <= threshold and approve_label:
        return approve_label
    return escalate_label or ""


def _answer_matches_shortcut(shortcut: ReasoningShortcut, answer_text: str) -> bool:
    answer = " ".join(str(answer_text or "").split()).strip()
    if not answer:
        return False
    if shortcut.constraint == "numeric_answer":
        expected = _number(shortcut.answer.replace("%", ""))
        observed = _extract_percentage(answer)
        if expected is None or observed is None:
            return False
        return abs(expected - observed) <= 0.05
    normalized_answer = normalize_text(answer)
    normalized_expected = normalize_text(shortcut.answer)
    if normalized_answer == normalized_expected:
        return True
    return f" {normalized_expected} " in f" {normalized_answer} "


def _canonical_knowledge_answer(kind: str, answer_text: Any, *, brief_answer: bool = False) -> str:
    answer = " ".join(str(answer_text or "").split()).strip()
    if not answer:
        return ""
    if kind != "capital_city" or not brief_answer:
        return answer
    direct = answer.strip().strip(" .!?")
    if "\n" not in direct and len(direct) <= 32 and len(direct.split()) <= 4:
        return direct
    match = re.search(r"(?is)\b(?:is|:)\s*(?P<value>[A-Za-z][A-Za-z .'-]{1,32})[.!?]?\s*$", answer)
    if not match:
        return ""
    candidate = match.group("value").strip().strip(" .!?")
    if not candidate or len(candidate.split()) > 4:
        return ""
    return candidate


def _extract_labels(raw_text: str) -> list:
    for pattern in (_POLICY_LABEL_PATTERN, _POLICY_LABEL_SET_PATTERN):
        match = pattern.search(raw_text or "")
        if not match:
            continue
        payload = match.group("labels")
        parts = re.split(r"[,|/;]", payload)
        labels = []
        for part in parts:
            cleaned = str(part or "").strip().strip("{}")
            if cleaned:
                labels.append(cleaned)
        if labels:
            return labels
    return []


def _best_label(labels: list, hints: tuple) -> str:
    for label in labels:
        normalized = normalize_text(label)
        if any(hint in normalized for hint in hints):
            return label.strip()
    return ""


def _cleanup_entity(value: str) -> str:
    cleaned = normalize_text(value)
    if not cleaned:
        return ""
    cleaned = re.split(
        r"\b(?:please|reply|answer|return|only|today|currently|now)\b", cleaned, maxsplit=1
    )[0].strip()
    words = [word for word in cleaned.split() if word not in {"the", "a", "an"}]
    if not words or len(words) > 5:
        return ""
    return " ".join(words)


def _prefers_brief_answer(normalized: str) -> bool:
    return not any(
        phrase in normalized
        for phrase in (
            "show your work",
            "step by step",
            "walk me through",
            "why is",
            "explain why",
            "explain how",
        )
    )


def _capital_prefers_brief_answer(normalized: str) -> bool:
    explicit_brief_markers = (
        "return only the city",
        "return only the city name",
        "reply with the city only",
        "answer with only the city",
        "answer with only the city name",
        "city only",
        "only the city",
        "only the city name",
    )
    return any(marker in normalized for marker in explicit_brief_markers)


def _number(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_percentage(value: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", str(value or ""))
    if match:
        return _number(match.group(1))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value or ""))
    if len(numbers) == 1:
        return _number(numbers[0])
    return None


def _format_percentage(value: float) -> str:
    rounded = round(float(value or 0.0), 2)
    if abs(rounded - round(rounded)) < 0.005:
        return f"{int(round(rounded))}%"
    return f"{rounded:.2f}".rstrip("0").rstrip(".") + "%"


def _extract_request_text(request_kwargs: dict[str, Any] | None) -> str:
    request_kwargs = request_kwargs or {}
    messages = request_kwargs.get("messages") or []
    if messages:
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "") or item.get("content", "") or ""))
                else:
                    parts.append(str(item or ""))
            return " ".join(part for part in parts if part)
        return str(content)
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt"))
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input"))
    return ""


def _extract_exact_token(request_text: str) -> str:
    for pattern in _EXACT_TOKEN_PATTERNS:
        match = pattern.search(request_text or "")
        if not match:
            continue
        token = str(match.group("token") or "").strip()
        if token and normalize_text(token) not in _GENERIC_EXACT_TOKENS:
            return token
    return ""


def _has_aux_context(
    request_kwargs: dict[str, Any] | None,
    context_hints: dict[str, Any] | None = None,
) -> bool:
    request_kwargs = request_kwargs or {}
    if any(request_kwargs.get(field) not in (None, "", [], {}) for field in _AUX_CONTEXT_FIELDS):
        return True
    raw = _raw_aux_context(request_kwargs, context_hints)
    return any(raw.get(field) not in (None, "", [], {}) for field in _AUX_CONTEXT_FIELDS)


def _raw_aux_context(
    request_kwargs: dict[str, Any] | None,
    context_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    request_kwargs = request_kwargs or {}
    for field in _AUX_CONTEXT_FIELDS:
        value = request_kwargs.get(field)
        if value not in (None, "", [], {}):
            payload[field] = value
    hints = dict(context_hints or {})
    if isinstance(hints.get("_byte_raw_aux_context"), dict):
        hints = dict(hints["_byte_raw_aux_context"])
    for field in _AUX_CONTEXT_FIELDS:
        value = hints.get(field)
        if value not in (None, "", [], {}):
            payload[field] = value
    return payload


def _requested_grounded_target(request_text: str) -> str:
    normalized = normalize_text(request_text)
    if not normalized:
        return ""
    if not any(
        marker in normalized
        for marker in (
            "return exactly",
            "reply exactly",
            "answer with only",
            "reply with only",
            "return only",
            "final action label",
            "action label",
            "identifier",
        )
    ):
        return ""
    for target, aliases in _GROUNDED_TARGET_RULES:
        if any(alias in normalized for alias in aliases):
            return target
    return ""


def _resolve_grounded_target_value(target: str, raw_aux: dict[str, Any]) -> str:
    candidates = []
    if target == "primary_service":
        candidates.extend(_repo_summary_candidates(raw_aux, ("services",), index=0))
        candidates.extend(_regex_candidates(raw_aux, _PRIMARY_SERVICE_PATTERNS))
    elif target == "fallback_service":
        candidates.extend(_repo_summary_candidates(raw_aux, ("services",), index=1))
        candidates.extend(_regex_candidates(raw_aux, _FALLBACK_SERVICE_PATTERNS))
    elif target == "queue_name":
        candidates.extend(_repo_summary_candidates(raw_aux, ("queue", "queue_name")))
        candidates.extend(_regex_candidates(raw_aux, _QUEUE_PATTERNS))
    elif target == "policy_label":
        candidates.extend(_repo_summary_candidates(raw_aux, ("policy_label",)))
        candidates.extend(_regex_candidates(raw_aux, _POLICY_LABEL_PATTERNS))
    elif target == "invoice_id":
        candidates.extend(_pattern_values(raw_aux, _INVOICE_PATTERN))
    elif target == "amount":
        candidates.extend(_pattern_values(raw_aux, _AMOUNT_PATTERN))
    elif target == "due_date":
        candidates.extend(_pattern_values(raw_aux, _DATE_PATTERN))
    elif target == "owner":
        candidates.extend(_regex_candidates(raw_aux, _OWNER_PATTERNS))
    elif target == "cause":
        candidates.extend(_regex_candidates(raw_aux, _CAUSE_PATTERNS))
    elif target == "action_label":
        grounded_policy = _grounded_refund_policy_label(raw_aux)
        if grounded_policy:
            candidates.append(grounded_policy)
        candidates.extend(_regex_candidates(raw_aux, _ACTION_PATTERNS))
    return _unique_candidate(
        candidates, label_like=target in {"owner", "cause", "action_label", "policy_label"}
    )


def _repo_summary_candidates(
    raw_aux: dict[str, Any], keys: tuple, *, index: int | None = None
) -> list:
    repo_summary = raw_aux.get("byte_repo_summary")
    if not isinstance(repo_summary, dict):
        return []
    values = []
    for key in keys:
        value = repo_summary.get(key)
        if value in (None, "", [], {}):
            continue
        if index is not None and isinstance(value, (list, tuple)):
            if 0 <= index < len(value):
                values.append(str(value[index]))
        else:
            values.append(str(value))
    return values


def _grounded_refund_policy_label(raw_aux: dict[str, Any]) -> str:
    policy_text = _grounded_policy_text(raw_aux)
    if not policy_text:
        return ""
    return _solve_refund_policy_label(policy_text, normalize_text(policy_text))


def _grounded_policy_text(raw_aux: dict[str, Any]) -> str:
    chunks: list[str] = []
    for field in (
        "byte_support_articles",
        "byte_retrieval_context",
        "byte_document_context",
        "byte_tool_result_context",
    ):
        value = raw_aux.get(field)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    title = str(item.get("title", "") or "").strip()
                    snippet = str(item.get("snippet", "") or item.get("content", "") or "").strip()
                    if title:
                        chunks.append(title)
                    if snippet:
                        chunks.append(snippet)
                else:
                    text = str(item or "").strip()
                    if text:
                        chunks.append(text)
        elif isinstance(value, dict):
            for nested in value.values():
                text = str(nested or "").strip()
                if text:
                    chunks.append(text)
        else:
            text = str(value or "").strip()
            if text:
                chunks.append(text)
    return "\n".join(chunk for chunk in chunks if chunk)


def _pattern_values(raw_aux: dict[str, Any], pattern: re.Pattern) -> list:
    values = []
    for blob in _aux_text_blobs(raw_aux):
        values.extend(match.group(0) for match in pattern.finditer(blob))
    return values


def _regex_candidates(raw_aux: dict[str, Any], patterns: tuple) -> list:
    values = []
    for blob in _aux_text_blobs(raw_aux):
        for pattern in patterns:
            for match in pattern.finditer(blob):
                value = str(match.groupdict().get("value") or "").strip()
                if value:
                    values.append(value)
    return values


def _aux_text_blobs(raw_aux: dict[str, Any]) -> list:
    blobs = []
    for field in _AUX_CONTEXT_FIELDS:
        value = raw_aux.get(field)
        if value in (None, "", [], {}):
            continue
        _append_text_blobs(value, blobs)
    return blobs


def _append_text_blobs(value: Any, blobs: list) -> None:
    if value in (None, "", [], {}):
        return
    if isinstance(value, str):
        blobs.append(value)
        return
    if isinstance(value, dict):
        ordered_parts = []
        for key in ("title", "snippet", "text", "content"):
            item = value.get(key)
            if item not in (None, "", [], {}):
                ordered_parts.append(str(item))
        if ordered_parts:
            blobs.append("\n".join(ordered_parts))
        for nested_key, nested_value in value.items():
            if nested_key in {"title", "snippet", "text", "content"}:
                continue
            _append_text_blobs(nested_value, blobs)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_text_blobs(item, blobs)
        return
    blobs.append(str(value))


def _unique_candidate(candidates: list, *, label_like: bool = False) -> str:
    seen = {}
    for candidate in candidates:
        value = str(candidate or "").strip().strip(" .,!?:;")
        if not value:
            continue
        normalized = normalize_text(value)
        if not normalized:
            continue
        if label_like:
            value = value.upper()
            normalized = normalize_text(value)
        seen[normalized] = seen.get(normalized, value)
    if len(seen) != 1:
        return ""
    return next(iter(seen.values()))


__all__ = [
    "_answer_matches_shortcut",
    "_canonical_knowledge_answer",
    "_extract_exact_token",
    "_extract_labels",
    "_extract_request_text",
    "_grounded_policy_text",
    "_has_aux_context",
    "_raw_aux_context",
    "_requested_grounded_target",
    "_resolve_grounded_target_value",
    "_solve_profit_margin",
    "_solve_refund_policy_label",
]
