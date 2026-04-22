"""Session-delta optimization-memory store."""


from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from byte.processor._optimization_summary import (
    _json_safe,
    _public_session_entry,
    stable_digest,
    summarize_artifact_payload,
)


class SessionDeltaStore:
    def __init__(self, *, max_entries: int = 4000) -> None:
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max(1, int(max_entries or 1))
        self._writes = 0
        self._hits = 0
        self._lock = Lock()

    def note(
        self,
        session_key: str,
        artifact_type: str,
        value: Any,
        *,
        scope: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not session_key:
            return {
                "session_key": "",
                "artifact_type": artifact_type,
                "changed": True,
                "previous_digest": "",
                "current_digest": stable_digest(value),
                "summary": summarize_artifact_payload(artifact_type, value, max_chars=240),
            }
        key = stable_digest(
            {"scope": scope or "", "session_key": session_key, "artifact_type": artifact_type or ""}
        )
        now = time.time()
        current_digest = stable_digest(value)
        summary = summarize_artifact_payload(artifact_type, value, max_chars=240)
        with self._lock:
            existing = self._entries.get(key)
            previous_digest = str((existing or {}).get("current_digest", "") or "")
            changed = previous_digest != current_digest
            entry = {
                "key": key,
                "scope": scope or "",
                "session_key": session_key,
                "artifact_type": str(artifact_type or ""),
                "previous_digest": previous_digest,
                "current_digest": current_digest,
                "changed": changed,
                "summary": summary,
                "metadata": _json_safe(dict(metadata or {})),
                "created_at": (existing or {}).get("created_at", now),
                "updated_at": now,
                "hits": int((existing or {}).get("hits", 0) or 0) + (0 if changed else 1),
            }
            if not changed:
                self._hits += 1
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._writes += 1
            self._evict_if_needed()
            return _public_session_entry(entry)

    def snapshot(self, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            entries = [_public_session_entry(entry) for entry in self._entries.values()]
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
                    "session_key": entry.get("session_key", "") or "",
                    "artifact_type": entry.get("artifact_type", "") or "",
                    "previous_digest": entry.get("previous_digest", "") or "",
                    "current_digest": entry.get("current_digest", "") or "",
                    "changed": bool(entry.get("changed", False)),
                    "summary": entry.get("summary", "") or "",
                    "metadata": _json_safe(dict(entry.get("metadata", {}) or {})),
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
            unchanged = sum(1 for entry in self._entries.values() if not entry.get("changed", True))
            return {
                "total_entries": len(self._entries),
                "writes": self._writes,
                "hits": self._hits,
                "unchanged_entries": unchanged,
                "max_entries": self._max_entries,
            }

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

__all__ = ["SessionDeltaStore"]
