"""Failure memory store for Byte execution workflows."""

import time
from collections import Counter
from threading import Lock
from typing import Any

from byte.processor.intent import extract_request_intent

from ._execution_keys import _json_safe, _stable_hash


class FailureMemoryStore:
    """Remember recurrent failure patterns to steer future requests safely."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._writes = 0
        self._lock = Lock()

    def record(
        self,
        request_kwargs: dict[str, Any] | None,
        *,
        reason: str,
        provider: str = "",
        model: str = "",
        scope: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_kwargs = request_kwargs or {}
        intent = extract_request_intent(request_kwargs)
        key = _stable_hash(
            {
                "scope": scope or "",
                "provider": provider or "",
                "model": model or "",
                "route_key": intent.route_key,
            }
        )
        with self._lock:
            entry = self._entries.setdefault(
                key,
                {
                    "key": key,
                    "scope": scope or "",
                    "provider": provider or "",
                    "model": model or "",
                    "category": intent.category,
                    "route_key": intent.route_key,
                    "reasons": Counter(),
                    "negative_context": {},
                    "total_failures": 0,
                    "metadata": {},
                    "created_at": time.time(),
                    "updated_at": time.time(),
                },
            )
            entry["reasons"][reason or "unknown"] += 1
            entry["total_failures"] += 1
            entry["updated_at"] = time.time()
            if metadata:
                metadata = _json_safe(dict(metadata))
                entry["metadata"].update(metadata)
                negative_context = metadata.get("negative_context_digests", {}) or {}
                negative_summaries = metadata.get("negative_context_summaries", {}) or {}
                for artifact_type, digests in dict(negative_context).items():
                    bucket = entry["negative_context"].setdefault(str(artifact_type), Counter())
                    for digest in digests or []:
                        bucket[str(digest)] += 1
                if negative_summaries:
                    existing_summaries = dict(
                        entry["metadata"].get("negative_context_summaries", {}) or {}
                    )
                    for digest, summary in dict(negative_summaries).items():
                        if digest not in existing_summaries and summary not in (None, "", [], {}):
                            existing_summaries[digest] = summary
                    entry["metadata"]["negative_context_summaries"] = existing_summaries
            self._writes += 1
            return _public_failure_entry(entry)

    def hint(
        self,
        request_kwargs: dict[str, Any] | None,
        *,
        provider: str = "",
        model: str = "",
        scope: str = "",
    ) -> dict[str, Any]:
        request_kwargs = request_kwargs or {}
        intent = extract_request_intent(request_kwargs)
        combined = Counter()
        total = 0
        negative_context = {}
        with self._lock:
            for entry in self._entries.values():
                if entry.get("route_key") != intent.route_key:
                    continue
                if scope and entry.get("scope") not in ("", scope):
                    continue
                if provider and entry.get("provider") not in ("", provider):
                    continue
                if model and entry.get("model") not in ("", model):
                    continue
                combined.update(entry.get("reasons", {}))
                total += int(entry.get("total_failures", 0) or 0)
                for artifact_type, digests in dict(entry.get("negative_context", {}) or {}).items():
                    bucket = negative_context.setdefault(str(artifact_type), Counter())
                    bucket.update(Counter(dict(digests or {})))
        return {
            "route_key": intent.route_key,
            "total_failures": total,
            "reasons": dict(combined),
            "clarify_first": combined.get("ambiguous_request", 0) >= 1,
            "prefer_expensive": (
                combined.get("cheap_response_rejected", 0) >= 1
                or combined.get("verification_failed", 0) >= 1
            ),
            "prefer_tool_context": combined.get("missing_tool_context", 0) >= 1,
            "avoid_cache_reuse": combined.get("schema_invalid", 0) >= 1,
            "negative_context_digests": {
                artifact_type: [digest for digest, count in counter.most_common() if count >= 1]
                for artifact_type, counter in negative_context.items()
                if counter
            },
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_failures = sum(
                int(entry.get("total_failures", 0) or 0) for entry in self._entries.values()
            )
            top_routes = sorted(
                (
                    {
                        "route_key": entry.get("route_key", ""),
                        "total_failures": int(entry.get("total_failures", 0) or 0),
                        "reasons": dict(entry.get("reasons", {})),
                        "negative_context": {
                            artifact_type: dict(counter)
                            for artifact_type, counter in dict(
                                entry.get("negative_context", {}) or {}
                            ).items()
                        },
                    }
                    for entry in self._entries.values()
                ),
                key=lambda item: item["total_failures"],
                reverse=True,
            )[:10]
        return {
            "total_routes": len(self._entries),
            "total_failures": total_failures,
            "writes": self._writes,
            "top_routes": top_routes,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            entries = [_public_failure_entry(entry) for entry in self._entries.values()]
        return {
            "entries": entries,
            "stats": self.stats(),
        }

    def merge(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        payload = payload or {}
        imported = 0
        skipped = 0
        with self._lock:
            for entry in payload.get("entries", []) or []:
                key = entry.get("key") or _stable_hash(entry)
                existing = self._entries.get(key)
                if existing is not None and (existing.get("updated_at", 0) or 0) > (
                    entry.get("updated_at", 0) or 0
                ):
                    skipped += 1
                    continue
                self._entries[key] = {
                    "key": key,
                    "scope": entry.get("scope", "") or "",
                    "provider": entry.get("provider", "") or "",
                    "model": entry.get("model", "") or "",
                    "category": entry.get("category", "") or "",
                    "route_key": entry.get("route_key", "") or "",
                    "reasons": Counter(dict(entry.get("reasons", {}) or {})),
                    "negative_context": {
                        str(artifact_type): Counter(dict(counter or {}))
                        for artifact_type, counter in dict(
                            entry.get("negative_context", {}) or {}
                        ).items()
                    },
                    "total_failures": int(entry.get("total_failures", 0) or 0),
                    "metadata": _json_safe(dict(entry.get("metadata", {}) or {})),
                    "created_at": entry.get("created_at", time.time()),
                    "updated_at": entry.get("updated_at", time.time()),
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
            self._writes = 0

def _public_failure_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": entry["key"],
        "scope": entry.get("scope", ""),
        "provider": entry.get("provider", ""),
        "model": entry.get("model", ""),
        "category": entry.get("category", ""),
        "route_key": entry.get("route_key", ""),
        "reasons": dict(entry.get("reasons", {})),
        "negative_context": {
            artifact_type: dict(counter)
            for artifact_type, counter in dict(entry.get("negative_context", {}) or {}).items()
        },
        "total_failures": int(entry.get("total_failures", 0) or 0),
        "metadata": dict(entry.get("metadata", {})),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
    }
