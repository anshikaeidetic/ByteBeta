"""Artifact optimization-memory store."""


from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from byte.processor._optimization_summary import (
    _json_safe,
    _lexical_overlap_score,
    _lexical_tokens,
    _public_artifact_entry,
    compact_text,
    compression_text_entry,
    encode_text_payload,
    estimate_tokens,
    related_text_score,
    stable_digest,
    summarize_artifact_payload,
    summarize_artifact_sketch,
)


class ArtifactMemoryStore:
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
        artifact_type: str,
        value: Any,
        *,
        fingerprint: str = "",
        summary: str = "",
        sketch: str = "",
        scope: str = "",
        metadata: dict[str, Any] | None = None,
        source: str = "request",
    ) -> dict[str, Any]:
        artifact_fingerprint = fingerprint or stable_digest(value)
        key = stable_digest(
            {"scope": scope or "", "type": artifact_type or "", "fingerprint": artifact_fingerprint}
        )
        now = time.time()
        compression, _ = compression_text_entry(
            " ".join(
                (
                    summary or "",
                    sketch or "",
                    compact_text(value, max_chars=512),
                )
            ),
            codec_name=self._codec_name,
            bits=self._bits,
        )
        entry = {
            "key": key,
            "scope": scope or "",
            "artifact_type": str(artifact_type or ""),
            "fingerprint": artifact_fingerprint,
            "summary": summary or summarize_artifact_payload(artifact_type, value, max_chars=320),
            "sketch": sketch or summarize_artifact_sketch(artifact_type, value, max_chars=180),
            "preview": compact_text(value, max_chars=200),
            "chars": len(compact_text(value, max_chars=20000)),
            "tokens_estimate": estimate_tokens(value),
            "compression": compression,
            "compression_payload": encode_text_payload(
                " ".join(
                    (
                        summary or summarize_artifact_payload(artifact_type, value, max_chars=320),
                        sketch or summarize_artifact_sketch(artifact_type, value, max_chars=180),
                        compact_text(value, max_chars=512),
                    )
                ),
                codec_name=self._codec_name,
                bits=self._bits,
            ),
            "metadata": _json_safe(dict(metadata or {})),
            "source": source or "request",
            "created_at": now,
            "updated_at": now,
            "hits": 0,
        }
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                entry["created_at"] = existing.get("created_at", now)
                entry["hits"] = int(existing.get("hits", 0) or 0) + 1
                self._hits += 1
                self._entries.pop(key, None)
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._writes += 1
            self._evict_if_needed()
            return _public_artifact_entry(entry)

    def get(
        self,
        artifact_type: str,
        *,
        fingerprint: str,
        scope: str = "",
    ) -> dict[str, Any] | None:
        key = stable_digest(
            {"scope": scope or "", "type": artifact_type or "", "fingerprint": fingerprint or ""}
        )
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry["hits"] = int(entry.get("hits", 0) or 0) + 1
            entry["updated_at"] = time.time()
            self._hits += 1
            return _public_artifact_entry(entry)

    def find_related(
        self,
        artifact_type: str,
        *,
        query_text: str,
        scope: str = "",
        top_k: int = 1,
        min_score: float = 0.18,
    ) -> list[dict[str, Any]]:
        query_tokens = _lexical_tokens(query_text)
        if not query_tokens:
            return []
        with self._lock:
            candidates = []
            for entry in self._entries.values():
                if artifact_type and entry.get("artifact_type") != artifact_type:
                    continue
                if scope and entry.get("scope") not in ("", scope):
                    continue
                score = _lexical_overlap_score(
                    query_tokens,
                    _lexical_tokens(
                        " ".join(
                            (
                                str(entry.get("summary", "") or ""),
                                str(entry.get("sketch", "") or ""),
                                str(entry.get("preview", "") or ""),
                            )
                        )
                    ),
                )
                compressed_score = related_text_score(
                    query_text,
                    entry.get("compression_payload"),
                    codec_name=self._codec_name,
                    bits=self._bits,
                )
                score = max(score, compressed_score)
                if score < min_score:
                    continue
                candidates.append((score, entry))

            ranked = sorted(
                candidates,
                key=lambda item: (
                    item[0],
                    int(item[1].get("hits", 0) or 0),
                    float(item[1].get("updated_at", 0) or 0),
                ),
                reverse=True,
            )[: max(1, int(top_k or 1))]
            results = []
            for score, entry in ranked:
                entry["hits"] = int(entry.get("hits", 0) or 0) + 1
                entry["updated_at"] = time.time()
                self._hits += 1
                public = _public_artifact_entry(entry)
                public["related_score"] = round(float(score), 4)
                results.append(public)
            return results

    def snapshot(self, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            entries = [_public_artifact_entry(entry) for entry in self._entries.values()]
            if limit is not None:
                entries = entries[-int(limit or 0) :]
        return {"entries": entries[::-1], "stats": self.stats()}

    def merge(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        payload = payload or {}
        imported = 0
        skipped = 0
        with self._lock:
            for entry in payload.get("entries", []) or []:
                key = entry.get("key") or stable_digest(entry)
                existing = self._entries.get(key)
                existing_updated = (existing or {}).get("updated_at", 0) or 0
                incoming_updated = entry.get("updated_at", 0) or 0
                if existing is not None and existing_updated > incoming_updated:
                    skipped += 1
                    continue
                normalized = {
                    "key": key,
                    "scope": entry.get("scope", "") or "",
                    "artifact_type": entry.get("artifact_type", "") or "",
                    "fingerprint": entry.get("fingerprint", "") or "",
                    "summary": entry.get("summary", "") or "",
                    "sketch": entry.get("sketch", "") or "",
                    "preview": entry.get("preview", "") or "",
                    "chars": int(entry.get("chars", 0) or 0),
                    "tokens_estimate": int(entry.get("tokens_estimate", 0) or 0),
                    "compression": _json_safe(dict(entry.get("compression", {}) or {})),
                    "metadata": _json_safe(dict(entry.get("metadata", {}) or {})),
                    "source": entry.get("source", "import") or "import",
                    "created_at": entry.get("created_at", time.time()),
                    "updated_at": incoming_updated or entry.get("created_at", time.time()),
                    "hits": int(entry.get("hits", 0) or 0),
                }
                self._entries[key] = normalized
                self._entries.move_to_end(key)
                imported += 1
            self._writes += imported
            self._evict_if_needed()
        return {"imported": imported, "skipped": skipped, "total_entries": len(self._entries)}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._writes = 0
            self._hits = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            compression_entries = [
                dict(entry.get("compression", {}) or {})
                for entry in self._entries.values()
                if entry.get("compression")
            ]
            return {
                "total_entries": len(self._entries),
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

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

__all__ = ["ArtifactMemoryStore"]
