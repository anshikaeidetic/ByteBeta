"""Prompt-piece optimization-memory store."""


from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Sequence
from threading import Lock
from typing import Any

from byte.processor._optimization_summary import (
    _json_safe,
    _public_piece_entry,
    compact_text,
    compression_text_entry,
    estimate_tokens,
    stable_digest,
    summarize_artifact_payload,
)


class PromptPieceStore:
    def __init__(self, *, max_entries: int = 4000, codec_name: str = "disabled", bits: int = 8) -> None:
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max(1, int(max_entries or 1))
        self._codec_name = str(codec_name or "disabled").strip().lower()
        self._bits = max(1, int(bits or 1))
        self._writes = 0
        self._hits = 0
        self._lock = Lock()

    def remember(
        self,
        piece_type: str,
        content: Any,
        *,
        scope: str = "",
        source: str = "request",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        digest = stable_digest(content)
        key = stable_digest({"scope": scope or "", "type": piece_type or "", "digest": digest})
        now = time.time()
        compression, _ = compression_text_entry(
            content,
            codec_name=self._codec_name,
            bits=self._bits,
        )
        entry = {
            "key": key,
            "scope": scope or "",
            "piece_type": str(piece_type or ""),
            "digest": digest,
            "preview": compact_text(content, max_chars=160),
            "summary": summarize_artifact_payload(piece_type, content, max_chars=240),
            "chars": len(compact_text(content, max_chars=20000)),
            "tokens_estimate": estimate_tokens(content),
            "compression": compression,
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
            return _public_piece_entry(entry)

    def remember_many(
        self,
        pieces: Sequence[dict[str, Any]],
        *,
        scope: str = "",
        source: str = "request",
    ) -> list[dict[str, Any]]:
        remembered = []
        for piece in pieces or []:
            remembered.append(
                self.remember(
                    str(piece.get("type") or piece.get("piece_type") or "piece"),
                    piece.get("content"),
                    scope=scope,
                    source=source,
                    metadata={
                        k: v for k, v in piece.items() if k not in {"type", "piece_type", "content"}
                    },
                )
            )
        return remembered

    def snapshot(self, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            entries = [_public_piece_entry(entry) for entry in self._entries.values()]
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
                    "piece_type": entry.get("piece_type", "") or "",
                    "digest": entry.get("digest", "") or "",
                    "preview": entry.get("preview", "") or "",
                    "summary": entry.get("summary", "") or "",
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
            total_tokens = sum(
                int(entry.get("tokens_estimate", 0) or 0) for entry in self._entries.values()
            )
            compression_entries = [
                dict(entry.get("compression", {}) or {})
                for entry in self._entries.values()
                if entry.get("compression")
            ]
            return {
                "total_entries": len(self._entries),
                "writes": self._writes,
                "hits": self._hits,
                "estimated_tokens": total_tokens,
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

__all__ = ["PromptPieceStore"]
