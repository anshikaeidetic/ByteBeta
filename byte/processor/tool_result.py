import hashlib
import json
import time
from threading import Lock
from typing import Any


def tool_result_key(tool_name: str, tool_args: Any, *, scope: str = "") -> str:
    payload = {
        "scope": scope or "",
        "tool": tool_name or "",
        "args": _normalize_tool_args(tool_args),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ToolResultStore:
    """Deterministic tool-result memory for API, DB, and function outputs."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._expired = 0
        self._lock = Lock()

    def put(
        self,
        tool_name: str,
        tool_args: Any,
        result: Any,
        *,
        ttl: float | None = None,
        scope: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        expires_at = now + ttl if ttl else None
        key = tool_result_key(tool_name, tool_args, scope=scope)
        entry = {
            "key": key,
            "tool_name": tool_name,
            "tool_args": _normalize_tool_args(tool_args),
            "result": result,
            "metadata": dict(metadata or {}),
            "scope": scope or "",
            "created_at": now,
            "expires_at": expires_at,
            "hits": 0,
        }
        with self._lock:
            self._entries[key] = entry
            self._writes += 1
        return _public_entry(entry, byte=False)

    def get(self, tool_name: str, tool_args: Any, *, scope: str = "") -> dict[str, Any] | None:
        key = tool_result_key(tool_name, tool_args, scope=scope)
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at = entry.get("expires_at")
            if expires_at and expires_at < now:
                del self._entries[key]
                self._expired += 1
                self._misses += 1
                return None
            entry["hits"] += 1
            self._hits += 1
            return _public_entry(entry, byte=True)

    def snapshot(self, limit: int | None = None) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            entries: list[dict[str, Any]] = []
            for key, entry in self._entries.items():
                expires_at = entry.get("expires_at")
                if expires_at and expires_at < now:
                    continue
                exported = _public_entry(entry, byte=False)
                exported["key"] = key
                entries.append(exported)
            entries.sort(
                key=lambda item: (
                    item.get("scope", ""),
                    item.get("tool_name", ""),
                    item.get("created_at", 0),
                )
            )
            if limit is not None:
                entries = entries[:limit]
            total_lookups = self._hits + self._misses
            stats = {
                "total_entries": len(self._entries),
                "writes": self._writes,
                "hits": self._hits,
                "misses": self._misses,
                "expired": self._expired,
                "total_lookups": total_lookups,
                "hit_ratio": round(self._hits / total_lookups, 4) if total_lookups else 0.0,
            }
            return {
                "entries": entries,
                "stats": stats,
            }

    def merge(self, payload: dict[str, Any]) -> dict[str, Any]:
        entries = payload.get("entries", []) if payload else []
        now = time.time()
        imported = 0
        skipped = 0
        with self._lock:
            for entry in entries:
                expires_at = entry.get("expires_at")
                if expires_at and expires_at < now:
                    skipped += 1
                    continue
                key = entry.get("key") or tool_result_key(
                    entry.get("tool_name", ""),
                    entry.get("tool_args"),
                    scope=entry.get("scope", ""),
                )
                existing = self._entries.get(key)
                if existing and (existing.get("created_at", 0) or 0) > (
                    entry.get("created_at", 0) or 0
                ):
                    skipped += 1
                    continue
                self._entries[key] = {
                    "key": key,
                    "tool_name": entry.get("tool_name", ""),
                    "tool_args": _normalize_tool_args(entry.get("tool_args")),
                    "result": entry.get("result"),
                    "metadata": dict(entry.get("metadata", {}) or {}),
                    "scope": entry.get("scope", "") or "",
                    "created_at": entry.get("created_at", now) or now,
                    "expires_at": expires_at,
                    "hits": int(entry.get("hits", 0) or 0),
                }
                imported += 1
            self._writes += imported
        return {
            "imported": imported,
            "skipped": skipped,
            "total_entries": len(self._entries),
        }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            self._writes = 0
            self._expired = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_lookups = self._hits + self._misses
            return {
                "total_entries": len(self._entries),
                "writes": self._writes,
                "hits": self._hits,
                "misses": self._misses,
                "expired": self._expired,
                "total_lookups": total_lookups,
                "hit_ratio": round(self._hits / total_lookups, 4) if total_lookups else 0.0,
            }


def _normalize_tool_args(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_tool_args(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_tool_args(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_tool_args(item) for item in value)
    return value


def _public_entry(entry: dict[str, Any], *, byte: bool) -> dict[str, Any]:
    return {
        "byte": byte,
        "tool_name": entry["tool_name"],
        "tool_args": entry["tool_args"],
        "result": entry["result"],
        "metadata": dict(entry.get("metadata", {})),
        "scope": entry.get("scope", ""),
        "created_at": entry.get("created_at"),
        "expires_at": entry.get("expires_at"),
        "hits": entry.get("hits", 0),
    }
