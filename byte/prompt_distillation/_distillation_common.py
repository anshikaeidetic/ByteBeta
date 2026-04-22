"""Shared registry types and common helpers for prompt distillation."""

from __future__ import annotations

# ruff: noqa: F401
import hashlib
import hmac
import json
import re
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

from byte.processor.optimization_memory import compact_text, estimate_tokens

_MODULE_TYPES = {
    "system",
    "tools",
    "tool_schemas",
    "prompt_pieces",
    "retrieval_context",
    "document_context",
    "support_articles",
    "repo_snapshot",
    "repo_summary",
    "changed_files",
    "changed_hunks",
    "tool_results",
}
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
_STABLE_FIELDS = {
    "system": 1.0,
    "tools": 0.98,
    "tool_schemas": 0.98,
    "prompt_pieces": 0.95,
    "retrieval_context": 0.9,
    "document_context": 0.88,
    "support_articles": 0.88,
    "repo_snapshot": 0.82,
    "repo_summary": 0.84,
    "changed_files": 0.86,
    "changed_hunks": 0.9,
    "tool_results": 0.9,
}
_DISTILLABLE_FIELD_TYPES = (
    ("byte_prompt_pieces", "prompt_pieces", 1.0),
    ("byte_retrieval_context", "retrieval_context", 0.98),
    ("byte_document_context", "document_context", 0.95),
    ("byte_support_articles", "support_articles", 0.93),
    ("byte_tool_result_context", "tool_result_context", 0.96),
    ("byte_changed_hunks", "changed_hunks", 0.94),
    ("byte_changed_files", "changed_files", 0.88),
    ("byte_repo_summary", "repo_summary", 0.82),
    ("byte_repo_snapshot", "repo_snapshot", 0.8),
    ("tools", "tools", 0.9),
    ("functions", "tool_schemas", 0.9),
)
_RETRIEVAL_LIKE_FIELDS = {
    "byte_retrieval_context",
    "byte_document_context",
    "byte_support_articles",
    "byte_tool_result_context",
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class PromptDistillationResult:
    request_kwargs: dict[str, Any]
    metadata: dict[str, Any]


class PromptModuleRegistry:
    def __init__(self, *, max_entries: int = 4096, artifact_version: str = "byte-prompt-distill-v1") -> None:
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max(1, int(max_entries or 1))
        self._artifact_version = str(artifact_version or "byte-prompt-distill-v1").strip()
        self._hits = 0
        self._writes = 0
        self._lock = Lock()

    def remember_many(
        self,
        pieces: list[dict[str, Any]],
        *,
        scope: str = "",
        source: str = "runtime",
    ) -> list[dict[str, Any]]:
        remembered: list[dict[str, Any]] = []
        for piece in pieces or []:
            piece_type = str(piece.get("type") or piece.get("piece_type") or "").strip().lower()
            if piece_type not in _MODULE_TYPES:
                continue
            remembered.append(
                self.remember(
                    piece_type,
                    piece.get("content"),
                    scope=scope,
                    source=source,
                    metadata={k: v for k, v in piece.items() if k not in {"content"}},
                )
            )
        return remembered

    def remember(
        self,
        module_type: str,
        content: Any,
        *,
        scope: str = "",
        source: str = "runtime",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        module_type = str(module_type or "").strip().lower()
        module_id = str((metadata or {}).get("module_id", "") or _module_id(module_type, content))
        now = time.time()
        preview = compact_text(content, max_chars=160)
        normalized = normalize_text(preview)
        entry = {
            "module_id": module_id,
            "scope": str(scope or ""),
            "module_type": module_type,
            "source": str(source or "runtime"),
            "preview": preview,
            "normalized_preview": normalized,
            "tokens_estimate": estimate_tokens(content),
            "chars": len(compact_text(content, max_chars=20000)),
            "stability": float(_STABLE_FIELDS.get(module_type, 0.75)),
            "metadata": dict(metadata or {}),
            "artifact_version": self._artifact_version,
            "created_at": now,
            "updated_at": now,
            "hits": 0,
        }
        with self._lock:
            existing = self._entries.get(module_id)
            if existing is not None:
                entry["created_at"] = existing.get("created_at", now)
                entry["hits"] = int(existing.get("hits", 0) or 0) + 1
                self._hits += 1
                self._entries.pop(module_id, None)
            self._entries[module_id] = entry
            self._entries.move_to_end(module_id)
            self._writes += 1
            self._evict_if_needed()
            return _public_module_entry(entry)

    def resolve(self, module_ids: list[str]) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        with self._lock:
            for module_id in module_ids or []:
                entry = self._entries.get(str(module_id or ""))
                if entry is None:
                    continue
                entry["hits"] = int(entry.get("hits", 0) or 0) + 1
                self._hits += 1
                self._entries.move_to_end(module_id)
                resolved.append(_public_module_entry(entry))
        return resolved

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "writes": self._writes,
                "artifact_version": self._artifact_version,
                "module_types": sorted({entry["module_type"] for entry in self._entries.values()}),
            }

    def snapshot(self, *, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            values = list(self._entries.values())
            if limit is not None:
                values = values[-max(0, int(limit or 0)) :]
            return {
                "entries": [_public_module_entry(entry) for entry in values],
                "stats": {
                    "entries": len(self._entries),
                    "hits": self._hits,
                    "writes": self._writes,
                    "artifact_version": self._artifact_version,
                    "module_types": sorted(
                        {entry["module_type"] for entry in self._entries.values()}
                    ),
                },
            }

    def merge(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        payload = payload or {}
        imported = 0
        skipped = 0
        for entry in payload.get("entries") or []:
            module_id = str(entry.get("module_id", "") or "")
            module_type = str(entry.get("module_type", "") or "")
            if not module_id or not module_type:
                skipped += 1
                continue
            imported += 1
            self.remember(
                module_type,
                {"preview": entry.get("preview", ""), "module_id": module_id},
                scope=str(entry.get("scope", "") or ""),
                source=str(entry.get("source", "import") or "import"),
                metadata=dict(entry.get("metadata", {}) or {}),
            )
        return {"imported": imported, "skipped": skipped, "total_entries": len(self._entries)}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._writes = 0

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

def _module_id(module_type: str, content: Any) -> str:
    payload = json.dumps(
        {
            "module_type": str(module_type or "").strip().lower(),
            "content": compact_text(content, max_chars=4000),
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _ratio_delta(baseline: int, observed: int) -> float:
    if baseline <= 0:
        return 0.0
    return round((float(baseline) - float(observed)) / float(baseline), 4)


def _public_module_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "module_id": str(entry.get("module_id", "") or ""),
        "scope": str(entry.get("scope", "") or ""),
        "module_type": str(entry.get("module_type", "") or ""),
        "source": str(entry.get("source", "") or ""),
        "preview": str(entry.get("preview", "") or ""),
        "tokens_estimate": int(entry.get("tokens_estimate", 0) or 0),
        "chars": int(entry.get("chars", 0) or 0),
        "stability": float(entry.get("stability", 0.0) or 0.0),
        "metadata": dict(entry.get("metadata", {}) or {}),
        "artifact_version": str(entry.get("artifact_version", "") or ""),
        "hits": int(entry.get("hits", 0) or 0),
    }


__all__ = [
    "_DISTILLABLE_FIELD_TYPES",
    "_MODULE_TYPES",
    "_RETRIEVAL_LIKE_FIELDS",
    "_STABLE_FIELDS",
    "PromptDistillationResult",
    "PromptModuleRegistry",
    "_module_id",
    "_public_module_entry",
    "_ratio_delta",
    "normalize_text",
]
