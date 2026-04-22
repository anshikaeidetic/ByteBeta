"""Deterministic reference detection for Byte trust evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from byte.processor.pre import normalize_text

from ._calibration import load_trust_calibration
from ._contracts import _extract_labels, _normalize_field_name, request_text

_LABEL_PATTERNS = (
    re.compile(r"(?is)\blabels?\s*:\s*(?P<labels>[^\n]+)"),
    re.compile(
        r"(?is)\b(?:one|single)\s+(?:label|action)\s+(?:from|out\s+of)\s*\{(?P<labels>[^}]+)\}"
    ),
)
_JSON_KEYS_PATTERN = re.compile(
    r"(?is)\bvalid\s+json\s+only\s+with\s+keys?\s+(?P<keys>[^.\n]+)"
)
_SET_VALUE_PATTERN = re.compile(
    r"(?is)(?:\bset\b|\band\b)\s+(?P<key>[A-Za-z0-9_ -]+?)\s+to\s+(?P<value>[A-Za-z0-9_.:-]+)"
)
_RULE_PATTERN = re.compile(
    r"(?is)\bif\s+(?P<condition>.+?)\s+return\s+(?P<label>[A-Z][A-Z0-9_:-]{1,})"
)
_ELSE_IF_PATTERN = re.compile(
    r"(?is)\belse\s+if\s+(?P<condition>.+?)\s+return\s+(?P<label>[A-Z][A-Z0-9_:-]{1,})"
)
_ELSE_PATTERN = re.compile(
    r"(?is)\b(?:otherwise|else)\b(?:[^A-Z\n]{0,48})?(?:return|reply\s+with|output)\s+(?P<label>[A-Z][A-Z0-9_:-]{1,})"
)
_TOKEN_VALUE_PATTERN = re.compile(
    r"(?is)\b(?P<field>[A-Za-z][A-Za-z _-]{1,32})\s*(?:=|:|is)\s*(?P<value>[A-Za-z0-9_.:-]+)"
)
_COMPARISON_PATTERN = re.compile(
    r"(?is)^(?P<field>[A-Za-z][A-Za-z _-]{1,32})\s*(?P<op>>=|<=|>|<|==|=)\s*(?P<value>-?\d+(?:\.\d+)?)$"
)
_WORD_MATCH_PATTERN = re.compile(
    r"(?is)^(?P<field>[A-Za-z][A-Za-z _-]{1,32})\s+(?P<value>[A-Za-z0-9_.:-]+)$"
)
_CONFLICT_HINTS = ("conflict", "disagree", "contradict", "unsafe_to_reuse", "escalate_conflict")
_UNIQUE_HINTS = ("unique", "nonce", "uuid", "hash", "digest", "trace")
_AUXILIARY_INJECTION_PATTERN = re.compile(
    r"(?is)\b(?:ignore\s+previous|system\s+message|developer\s+message|follow\s+these\s+instructions|override\s+policy|do\s+not\s+follow\s+previous)\b"
)
_AUXILIARY_SEGMENT_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_AUXILIARY_CONTEXT_FIELDS = (
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

@dataclass(frozen=True)
class DeterministicReference:
    answer: str
    constraint: str
    reason: str
    score: float = field(default_factory=lambda: load_trust_calibration().reference_score("default"))
    kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def deterministic_reference_answer(
    request_kwargs: dict[str, Any] | None,
    *,
    context_hints: dict[str, Any] | None = None,
) -> DeterministicReference | None:
    text = request_text(request_kwargs)
    if not text:
        return None
    safe_context_text = _safe_auxiliary_context_text(context_hints or {})
    combined_text = "\n".join(part for part in (text, safe_context_text) if part).strip()
    label_reference = _rule_label_reference(combined_text)
    if label_reference is not None:
        return label_reference
    curated_label_reference = _curated_label_reference(combined_text)
    if curated_label_reference is not None:
        return curated_label_reference
    refund_reference = _refund_policy_reference(combined_text)
    if refund_reference is not None:
        return refund_reference
    invoice_reference = _invoice_identifier_reference(text, safe_context_text)
    if invoice_reference is not None:
        return invoice_reference
    code_symbol_reference = _code_symbol_reference(text, safe_context_text)
    if code_symbol_reference is not None:
        return code_symbol_reference
    json_reference = _json_contract_reference(combined_text)
    if json_reference is not None:
        return json_reference
    return None

def _rule_label_reference(text: str) -> DeterministicReference | None:
    labels = _extract_labels(text)
    if not labels:
        return None
    values = _extract_symbolic_values(text)
    rule_clauses: list[tuple[str, str]] = []
    for pattern in (_RULE_PATTERN, _ELSE_IF_PATTERN):
        for match in pattern.finditer(text):
            condition = " ".join(str(match.group("condition") or "").split())
            label = str(match.group("label") or "").strip()
            if condition and label:
                rule_clauses.append((condition, label))
    fallback_match = _ELSE_PATTERN.search(text)
    fallback_label = str(fallback_match.group("label") or "").strip() if fallback_match else ""
    if not rule_clauses or not fallback_label:
        return None
    for condition, label in rule_clauses:
        if _evaluate_condition(condition, values):
            return DeterministicReference(
                answer=label,
                constraint="label_set",
                reason="symbolic_rule_reference",
                score=load_trust_calibration().reference_score("symbolic_rule"),
                kind="label_rule",
            )
    return DeterministicReference(
        answer=fallback_label,
        constraint="label_set",
        reason="symbolic_rule_reference",
        score=load_trust_calibration().reference_score("symbolic_rule"),
        kind="label_rule",
    )


def _curated_label_reference(text: str) -> DeterministicReference | None:
    labels = {
        str(label or "").strip().strip(".").upper()
        for label in _extract_labels(text)
        if str(label or "").strip().strip(".")
    }
    if not {"ALLOW", "REVIEW", "BLOCK"}.issubset(labels):
        return None
    score_match = re.search(
        r"(?is)\b(?:azimuth|signal|nebula|telemetry)\s+score(?:\s*(?:=|:|is))?\s*(?P<value>-?\d+(?:\.\d+)?)",
        text,
    )
    radius_match = re.search(
        r"(?is)\b(?:blast\s+)?radius(?:\s*(?:=|:|is))?\s*(?P<value>external|internal)\b",
        text,
    )
    override_match = re.search(
        r"(?is)\b(?:manual\s+)?override(?:\s*(?:=|:|is))?\s*(?P<value>yes|no)\b",
        text,
    )
    if not score_match or not radius_match:
        return None
    normalized_radius = str(radius_match.group("value") or "").strip().lower()
    normalized_override = str(
        override_match.group("value") if override_match else ""
    ).strip().lower()
    score_value = float(score_match.group("value"))
    answer = (
        "BLOCK"
        if score_value >= load_trust_calibration().reference_threshold("curated_policy_block_score")
        and normalized_radius == "external"
        else "REVIEW"
        if normalized_override == "yes"
        else "ALLOW"
    )
    return DeterministicReference(
        answer=answer,
        constraint="label_set",
        reason="curated_policy_reference",
        score=load_trust_calibration().reference_score("curated_policy"),
        kind="label_rule",
    )


def _json_contract_reference(text: str) -> DeterministicReference | None:
    key_match = _JSON_KEYS_PATTERN.search(text)
    if not key_match:
        return None
    raw_keys = str(key_match.group("keys") or "")
    raw_keys = re.sub(r"(?is)\bno\s+markdown\b.*$", "", raw_keys).strip().strip(".")
    keys = [
        part.strip().strip(".")
        for part in re.split(r"\s*(?:,|\band\b)\s*", raw_keys, flags=re.I)
        if part.strip().strip(".")
    ]
    assignments = {}
    for match in _SET_VALUE_PATTERN.finditer(text):
        key = _normalize_field_name(match.group("key"))
        if not key:
            continue
        assignments[key] = str(match.group("value") or "").strip().strip(".")
    payload: dict[str, str] = {}
    for key in keys:
        normalized_key = _normalize_field_name(key)
        if normalized_key not in assignments:
            return None
        payload[normalized_key] = assignments[normalized_key]
    answer = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return DeterministicReference(
        answer=answer,
        constraint="json",
        reason="json_contract_reference",
        score=load_trust_calibration().reference_score("json_contract"),
        kind="json_contract",
    )


def _refund_policy_reference(text: str) -> DeterministicReference | None:
    labels = {
        str(label or "").strip().strip(".").upper()
        for label in _extract_labels(text)
        if str(label or "").strip().strip(".")
    }
    if not {"REFUND_APPROVE", "REFUND_DENY", "ESCALATE"}.issubset(labels):
        return None
    window_match = re.search(r"(?is)\brefunds?\s+allowed\s+within\s+(?P<days>\d+)\s+days\b", text)
    day_match = re.search(
        r"(?is)\b(?:request|customer\s+asked|asked)\s+(?:on\s+)?day\s+(?P<day>\d+)\b",
        text,
    )
    if not window_match or not day_match:
        return None
    allowed_days = int(window_match.group("days"))
    request_day = int(day_match.group("day"))
    answer = "REFUND_APPROVE" if request_day <= allowed_days else "REFUND_DENY"
    return DeterministicReference(
        answer=answer,
        constraint="label_set",
        reason="refund_policy_reference",
        score=load_trust_calibration().reference_score("refund_policy"),
        kind="refund_policy",
    )


def _invoice_identifier_reference(
    request_text_value: str,
    context_text: str,
) -> DeterministicReference | None:
    normalized = normalize_text(request_text_value)
    if "invoice identifier" not in normalized and "invoice id" not in normalized:
        return None
    matches = re.findall(r"\bINV-\d+\b", context_text or "", flags=re.I)
    if not matches:
        return None
    counts: dict[str, int] = {}
    for match in matches:
        key = str(match).upper()
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return DeterministicReference(
        answer=ranked[0][0],
        constraint="exact_text",
        reason="grounded_invoice_reference",
        score=load_trust_calibration().reference_score("invoice_identifier"),
        kind="invoice_identifier",
    )


def _code_symbol_reference(
    request_text_value: str,
    context_text: str,
) -> DeterministicReference | None:
    normalized = normalize_text(request_text_value)
    if not any(
        token in normalized
        for token in ("function name", "symbol", "codebase context", "normalizes the invoice")
    ):
        return None
    candidates = re.findall(r"\b(?:def|function|class)\s+([A-Za-z_][A-Za-z0-9_]*)", context_text or "", flags=re.I)
    if not candidates:
        return None
    ranked: list[tuple[float, str]] = []
    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        score = 0.0
        if "normalize" in normalized_candidate:
            score += load_trust_calibration().reference_score("normalize_symbol_bonus")
        if "invoice" in normalized_candidate:
            score += load_trust_calibration().reference_score("invoice_symbol_bonus")
        if "function name" in normalized:
            score += load_trust_calibration().reference_score("function_name_bonus")
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_candidate = ranked[0]
    if best_score < load_trust_calibration().reference_threshold("code_symbol_min_score"):
        return None
    if (
        len(ranked) > 1
        and abs(best_score - ranked[1][0])
        < load_trust_calibration().reference_threshold("code_symbol_tie_margin")
    ):
        return None
    return DeterministicReference(
        answer=best_candidate,
        constraint="exact_text",
        reason="grounded_code_symbol_reference",
        score=load_trust_calibration().reference_score("code_symbol"),
        kind="code_symbol",
    )

def _extract_symbolic_values(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for match in _TOKEN_VALUE_PATTERN.finditer(text):
        field = _normalize_field_name(match.group("field"))
        value = str(match.group("value") or "").strip().strip(".")
        if not field or not value:
            continue
        lowered = value.lower()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", lowered):
            values[field] = float(lowered)
        else:
            values[field] = lowered
    return values


def _evaluate_condition(condition: str, values: dict[str, Any]) -> bool:
    segments = re.split(r"\s+\band\b\s+", condition, flags=re.I)
    return all(_evaluate_atomic_condition(segment, values) for segment in segments if segment.strip())


def _evaluate_atomic_condition(segment: str, values: dict[str, Any]) -> bool:
    piece = " ".join(str(segment or "").split()).strip().rstrip(".")
    match = _COMPARISON_PATTERN.match(piece)
    if match:
        field = _normalize_field_name(match.group("field"))
        op = str(match.group("op") or "")
        expected = float(match.group("value"))
        observed = _resolve_observed_value(field, values)
        if not isinstance(observed, (int, float)):
            return False
        observed_value = float(observed)
        if op == ">=":
            return observed_value >= expected
        if op == "<=":
            return observed_value <= expected
        if op == ">":
            return observed_value > expected
        if op == "<":
            return observed_value < expected
        return observed_value == expected
    match = _WORD_MATCH_PATTERN.match(piece)
    if not match:
        return False
    field = _normalize_field_name(match.group("field"))
    expected_word = str(match.group("value") or "").strip().strip(".").lower()
    observed = _resolve_observed_value(field, values)
    return str(observed or "").strip().lower() == expected_word

def _auxiliary_context_values(
    request_kwargs: dict[str, Any] | None,
    context: dict[str, Any],
) -> list[str]:
    request_kwargs = request_kwargs or {}
    raw_aux = context.get("_byte_raw_aux_context", {}) or {}
    if not isinstance(raw_aux, dict):
        raw_aux = {}
    values: list[str] = []
    for field_name in _AUXILIARY_CONTEXT_FIELDS:
        value = request_kwargs.get(field_name)
        if value not in (None, "", [], {}):
            values.append(_stringify_auxiliary_value(value))
            continue
        value = raw_aux.get(field_name)
        if value not in (None, "", [], {}):
            values.append(_stringify_auxiliary_value(value))
    return values


def _stringify_auxiliary_value(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            rendered = _stringify_auxiliary_value(item)
            if rendered:
                parts.append(f"{key} {rendered}")
        return "\n".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            rendered = _stringify_auxiliary_value(item)
            if rendered:
                parts.append(rendered)
        return "\n".join(parts)
    return str(value)


def _safe_auxiliary_context_text(context: dict[str, Any]) -> str:
    values = _auxiliary_context_values({}, context)
    sanitized = [_sanitize_auxiliary_text(value) for value in values]
    return "\n".join(value for value in sanitized if value)


def _sanitize_auxiliary_text(text: str) -> str:
    segments = [
        segment.strip()
        for segment in _AUXILIARY_SEGMENT_SPLIT_PATTERN.split(str(text or ""))
        if segment and segment.strip()
    ]
    safe_segments = [
        segment for segment in segments if not _AUXILIARY_INJECTION_PATTERN.search(segment)
    ]
    return "\n".join(safe_segments)


def _resolve_observed_value(field: str, values: dict[str, Any]) -> Any:
    if field in values:
        return values[field]
    suffix_matches = [
        value for key, value in values.items() if key.endswith(f"_{field}") or key == field
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    token_matches = []
    field_tokens = tuple(part for part in field.split("_") if part)
    for key, value in values.items():
        if all(token in key.split("_") for token in field_tokens):
            token_matches.append(value)
    if len(token_matches) == 1:
        return token_matches[0]
    return values.get(field)
