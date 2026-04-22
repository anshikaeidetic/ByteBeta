"""Reasoning shortcut datatypes and memory store helpers."""

# ruff: noqa: F401
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

from byte.processor.optimization_memory import stable_digest
from byte.processor.pre import normalize_text
from byte.quantization.vector import (
    compression_text_entry,
    encode_text_payload,
    related_text_score,
)


@dataclass(frozen=True)
class ReasoningShortcut:
    kind: str
    answer: str
    confidence: float
    reason: str
    byte_reason: str
    key: str = ""
    source: str = "deterministic"
    constraint: str = "deterministic_reasoning"
    promotion_state: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "answer": self.answer,
            "confidence": self.confidence,
            "reason": self.reason,
            "byte_reason": self.byte_reason,
            "key": self.key,
            "source": self.source,
            "constraint": self.constraint,
            "promotion_state": self.promotion_state,
        }


class ReasoningMemoryStore:
    """Provider-agnostic memory for stable reasoning answers."""

    def __init__(self, *, max_entries: int = 2000, codec_name: str = "disabled", bits: int = 8) -> None:
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max(1, int(max_entries or 1))
        self._codec_name = str(codec_name or "disabled").strip().lower()
        self._bits = max(1, int(bits or 1))
        self._writes = 0
        self._hits = 0
        self._lock = Lock()

    def remember(
        self,
        *,
        kind: str,
        key: str,
        answer: Any,
        verified: bool = True,
        metadata: dict[str, Any] | None = None,
        source: str = "llm",
    ) -> dict[str, Any]:
        entry_key = str(key or "").strip()
        if not entry_key:
            raise ValueError("reasoning memory key is required")
        answer_text = " ".join(str(answer or "").split()).strip()
        if not answer_text:
            raise ValueError("reasoning memory answer is required")
        now = time.time()
        compression, _ = compression_text_entry(
            f"{entry_key} {answer_text}",
            codec_name=self._codec_name,
            bits=self._bits,
        )
        entry = {
            "key": entry_key,
            "kind": str(kind or "").strip() or "generic",
            "answer": answer_text,
            "verified": bool(verified),
            "compression": compression,
            "compression_payload": encode_text_payload(
                f"{entry_key} {answer_text}",
                codec_name=self._codec_name,
                bits=self._bits,
            ),
            "metadata": dict(metadata or {}),
            "source": str(source or "llm"),
            "created_at": now,
            "updated_at": now,
            "hits": 0,
        }
        with self._lock:
            existing = self._entries.get(entry_key)
            if existing is not None:
                entry["created_at"] = existing.get("created_at", now)
                entry["hits"] = int(existing.get("hits", 0) or 0)
                entry["metadata"] = _merge_reasoning_metadata(
                    dict(existing.get("metadata", {}) or {}),
                    dict(entry.get("metadata", {}) or {}),
                )
                self._entries.pop(entry_key, None)
            else:
                entry["metadata"] = _merge_reasoning_metadata(
                    {},
                    dict(entry.get("metadata", {}) or {}),
                )
            self._entries[entry_key] = entry
            self._entries.move_to_end(entry_key)
            self._writes += 1
            self._evict_if_needed()
            return dict(entry)

    def lookup(
        self,
        *,
        key: str,
        kind: str = "",
        verified_only: bool = True,
    ) -> dict[str, Any] | None:
        entry_key = str(key or "").strip()
        if not entry_key:
            return None
        requested_kind = str(kind or "").strip()
        with self._lock:
            entry = self._entries.get(entry_key)
            if entry is None:
                return None
            if requested_kind and entry.get("kind") != requested_kind:
                return None
            if verified_only and not entry.get("verified", False):
                return None
            entry["hits"] = int(entry.get("hits", 0) or 0) + 1
            entry["updated_at"] = time.time()
            self._hits += 1
            self._entries.move_to_end(entry_key)
            public = dict(entry)
            public.pop("compression_payload", None)
            return public

    def lookup_related(
        self,
        *,
        query_text: str,
        kind: str = "",
        verified_only: bool = True,
        min_score: float = 0.7,
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        query_text = " ".join(str(query_text or "").split()).strip()
        if not query_text:
            return []
        with self._lock:
            ranked = []
            for entry in self._entries.values():
                if kind and entry.get("kind") != kind:
                    continue
                if verified_only and not entry.get("verified", False):
                    continue
                score = related_text_score(
                    query_text,
                    entry.get("compression_payload"),
                    codec_name=self._codec_name,
                    bits=self._bits,
                )
                if score < min_score:
                    continue
                ranked.append((score, entry))
            ranked.sort(
                key=lambda item: (
                    item[0],
                    int(item[1].get("hits", 0) or 0),
                    float(item[1].get("updated_at", 0) or 0),
                ),
                reverse=True,
            )
            results = []
            for score, entry in ranked[: max(1, int(top_k or 1))]:
                entry["hits"] = int(entry.get("hits", 0) or 0) + 1
                entry["updated_at"] = time.time()
                self._hits += 1
                public = dict(entry)
                public["related_score"] = round(float(score), 4)
                public.pop("compression_payload", None)
                results.append(public)
            return results

    def stats(self) -> dict[str, Any]:
        with self._lock:
            verified = sum(1 for entry in self._entries.values() if entry.get("verified"))
            compression_entries = [
                dict(entry.get("compression", {}) or {})
                for entry in self._entries.values()
                if entry.get("compression")
            ]
            return {
                "total_entries": len(self._entries),
                "verified_entries": verified,
                "unverified_entries": len(self._entries) - verified,
                "writes": self._writes,
                "hits": self._hits,
                "max_entries": self._max_entries,
                "codec": self._codec_name,
                "bits": self._bits,
                "compressed_entries": len(compression_entries),
                "avg_compression_ratio": round(
                    sum(
                        float(item.get("compression_ratio", 0.0) or 0.0)
                        for item in compression_entries
                    )
                    / len(compression_entries),
                    6,
                )
                if compression_entries
                else 0.0,
            }

    def snapshot(self, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            entries = []
            for entry in self._entries.values():
                public = dict(entry)
                public.pop("compression_payload", None)
                entries.append(public)
            if limit is not None:
                entries = entries[-int(limit or 0) :]
        return {
            "entries": entries[::-1],
            "stats": self.stats(),
        }

    def merge(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        payload = payload or {}
        entries = payload.get("entries", []) or []
        imported = 0
        skipped = 0
        with self._lock:
            for entry in entries:
                key = str(entry.get("key", "") or "").strip()
                answer = " ".join(str(entry.get("answer", "") or "").split()).strip()
                if not key or not answer:
                    skipped += 1
                    continue
                existing = self._entries.get(key)
                incoming_updated = float(entry.get("updated_at", 0) or 0)
                if (
                    existing is not None
                    and float(existing.get("updated_at", 0) or 0) > incoming_updated
                ):
                    skipped += 1
                    continue
                normalized = {
                    "key": key,
                    "kind": str(entry.get("kind", "") or "").strip() or "generic",
                    "answer": answer,
                    "verified": bool(entry.get("verified", False)),
                    "compression": dict(entry.get("compression", {}) or {}),
                    "metadata": dict(entry.get("metadata", {}) or {}),
                    "source": str(entry.get("source", "import") or "import"),
                    "created_at": float(entry.get("created_at", time.time()) or time.time()),
                    "updated_at": incoming_updated
                    or float(entry.get("created_at", time.time()) or time.time()),
                    "hits": int(entry.get("hits", 0) or 0),
                }
                self._entries[key] = normalized
                self._entries.move_to_end(key)
                imported += 1
            self._writes += imported
            self._evict_if_needed()
        return {
            "imported": imported,
            "skipped": skipped,
            "total_entries": len(self._entries),
        }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._writes = 0
            self._hits = 0

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


def _merge_reasoning_metadata(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing or {})
    merged.update(dict(incoming or {}))
    signatures: list[str] = []
    for value in (
        existing.get("prompt_signatures"),
        incoming.get("prompt_signatures"),
    ):
        if isinstance(value, list):
            signatures.extend(str(item).strip() for item in value if str(item).strip())
    for value in (existing.get("prompt_signature"), incoming.get("prompt_signature")):
        token = str(value or "").strip()
        if token:
            signatures.append(token)
    deduped_signatures: list[str] = []
    seen: set[str] = set()
    for signature in signatures:
        if signature in seen:
            continue
        seen.add(signature)
        deduped_signatures.append(signature)
    merged["prompt_signatures"] = deduped_signatures
    merged["prompt_diversity_count"] = len(deduped_signatures)
    merged["observation_count"] = max(
        int(existing.get("observation_count", 0) or 0) + 1,
        int(incoming.get("observation_count", 0) or 0),
        len(deduped_signatures),
    )
    if merged.get("promotion_required"):
        merged["promotion_state"] = (
            "dynamic_verified" if len(deduped_signatures) >= 2 else "near_threshold_shadow"
        )
    else:
        merged["promotion_state"] = str(merged.get("promotion_state", "") or "dynamic_verified")
    return merged


def _candidate_promoted(candidate: dict[str, Any], entry: dict[str, Any]) -> bool:
    if not bool(candidate.get("promotion_required", False)):
        return True
    metadata = dict(entry.get("metadata", {}) or {})
    return str(metadata.get("promotion_state", "") or "") == "dynamic_verified"


def _prompt_signature(request_text: str) -> str:
    normalized = normalize_text(request_text)
    if not normalized:
        return ""
    return stable_digest({"text": normalized})


def _prompt_module_signatures(request_kwargs: dict[str, Any] | None) -> list[str]:
    request_kwargs = request_kwargs or {}
    distilled = [
        str(item).strip()
        for item in (request_kwargs.get("byte_distilled_prompt_modules") or [])
        if str(item).strip()
    ]
    if distilled:
        deduped_modules: list[str] = []
        seen_modules: set[str] = set()
        for signature in distilled:
            if signature in seen_modules:
                continue
            seen_modules.add(signature)
            deduped_modules.append(signature)
        return deduped_modules
    messages = list(request_kwargs.get("messages") or [])
    signatures: list[str] = []
    for message in messages:
        role = str(message.get("role", "") or "").strip().lower()
        content = " ".join(str(message.get("content", "") or "").split()).strip()
        if not role or not content:
            continue
        signatures.append(stable_digest({"role": role, "content": normalize_text(content)}))
    deduped: list[str] = []
    seen: set[str] = set()
    for signature in signatures:
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(signature)
    return deduped


def _related_lookup_min_score(memory_candidate: dict[str, Any], *, config: Any | None) -> float:
    score = float(memory_candidate.get("min_related_score", 0.7) or 0.7)
    kind = str(memory_candidate.get("kind", "") or "").strip().lower()
    if kind == "capital_city":
        score = max(score, 0.94)
    elif kind in {"grounded_value", "policy_label"}:
        score = max(score, 0.86)

    config_enabled = bool(
        getattr(config, "context_compiler_related_memory", True) if config is not None else True
    )
    if config_enabled and config is not None:
        configured = getattr(config, "context_compiler_related_min_score", None)
        if configured is not None:
            score = max(score, float(configured or 0.0))
    return max(0.0, min(score, 0.995))


__all__ = [
    "ReasoningMemoryStore",
    "ReasoningShortcut",
    "_candidate_promoted",
    "_prompt_module_signatures",
    "_prompt_signature",
    "_related_lookup_min_score",
]
