"""Contract parsing helpers for Byte trust evaluation."""

from __future__ import annotations

import re
from typing import Any

from byte.processor.pre import normalize_text

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


def request_text(request_kwargs: dict[str, Any] | None) -> str:
    request_kwargs = request_kwargs or {}
    messages = request_kwargs.get("messages") or []
    if messages:
        parts: list[str] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", "") or item.get("content", "") or ""))
                    else:
                        parts.append(str(item or ""))
            else:
                parts.append(str(content or ""))
        return "\n".join(part for part in parts if part).strip()
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input") or "")
    return ""


def extract_contract(request_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    text = request_text(request_kwargs)
    normalized = normalize_text(text)
    labels = _extract_labels(text)
    structured_format = ""
    if "json" in normalized:
        structured_format = "json"
    elif "yaml" in normalized:
        structured_format = "yaml"
    elif "csv" in normalized:
        structured_format = "csv"
    exact_token = _extract_exact_token(text)
    return {
        "exact_token": exact_token,
        "labels": labels,
        "structured_format": structured_format,
        "strict": bool(exact_token or labels or structured_format),
    }

def _extract_labels(text: str) -> list[str]:
    labels: list[str] = []
    for pattern in _LABEL_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        parts = re.split(r"[,/|;\n]", str(match.group("labels") or ""))
        for part in parts:
            token = re.findall(r"[A-Za-z0-9_./:-]+", part.strip().strip("{}[]()"))
            if token and any(ch.isalpha() for ch in token[0]):
                labels.append(token[0].rstrip(".,;:"))
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        key = label.upper()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(label)
    return deduped


def _extract_exact_token(text: str) -> str:
    patterns = (
        re.compile(
            r"(?is)(?:return|reply|respond|answer)\s+with\s+exactly\s+(?P<token>[A-Za-z0-9_.:-]+)"
        ),
        re.compile(r"(?is)exactly\s+(?P<token>[A-Za-z0-9_.:-]+)\s+and\s+nothing\s+else"),
    )
    for pattern in patterns:
        match = pattern.search(text or "")
        if match:
            token = str(match.group("token") or "").strip()
            if token:
                return token
    return ""

def _normalize_field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
