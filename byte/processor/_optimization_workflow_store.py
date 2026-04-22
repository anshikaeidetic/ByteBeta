"""Workflow-plan optimization-memory store."""


from __future__ import annotations

import time
from collections import Counter, OrderedDict
from threading import Lock
from typing import Any

from byte.processor._optimization_summary import (
    _extract_request_intent,
    _json_safe,
    _public_workflow_entry,
    _success_rate,
    stable_digest,
)


class WorkflowPlanStore:
    def __init__(self, *, max_entries: int = 2000) -> None:
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max(1, int(max_entries or 1))
        self._writes = 0
        self._hits = 0
        self._lock = Lock()

    def remember(
        self,
        request_kwargs: dict[str, Any] | None,
        *,
        action: str,
        route_preference: str = "",
        counterfactual_action: str = "",
        counterfactual_reason: str = "",
        repo_fingerprint: str = "",
        artifact_fingerprint: str = "",
        scope: str = "",
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_kwargs = request_kwargs or {}
        intent = _extract_request_intent(request_kwargs)
        key = stable_digest(
            {
                "scope": scope or "",
                "route_key": intent.route_key,
                "repo": repo_fingerprint or "",
                "artifact": artifact_fingerprint or "",
                "tool_signature": intent.tool_signature,
            }
        )
        now = time.time()
        with self._lock:
            existing = self._entries.get(key)
            if existing is None:
                entry = {
                    "key": key,
                    "scope": scope or "",
                    "category": intent.category,
                    "route_key": intent.route_key,
                    "canonical_key": intent.canonical_key,
                    "tool_signature": intent.tool_signature,
                    "repo_fingerprint": repo_fingerprint or "",
                    "artifact_fingerprint": artifact_fingerprint or "",
                    "preferred_action": action or "",
                    "route_preference": route_preference or "",
                    "successes": 1 if success else 0,
                    "failures": 0 if success else 1,
                    "failed_actions": Counter(),
                    "counterfactual_actions": Counter(),
                    "metadata": _json_safe(dict(metadata or {})),
                    "created_at": now,
                    "updated_at": now,
                    "hits": 0,
                }
                if not success and action:
                    entry["failed_actions"][action] += 1
                if counterfactual_action:
                    entry["counterfactual_actions"][counterfactual_action] += 1
                    entry["metadata"]["counterfactual_reason"] = counterfactual_reason or ""
            else:
                entry = dict(existing)
                entry["preferred_action"] = action or entry.get("preferred_action", "")
                entry["route_preference"] = route_preference or entry.get("route_preference", "")
                entry["successes"] = int(entry.get("successes", 0) or 0) + (1 if success else 0)
                entry["failures"] = int(entry.get("failures", 0) or 0) + (0 if success else 1)
                entry["failed_actions"] = Counter(dict(entry.get("failed_actions", {}) or {}))
                entry["counterfactual_actions"] = Counter(
                    dict(entry.get("counterfactual_actions", {}) or {})
                )
                if not success and action:
                    entry["failed_actions"][action] += 1
                if counterfactual_action:
                    entry["counterfactual_actions"][counterfactual_action] += 1
                merged = dict(entry.get("metadata", {}))
                merged.update(_json_safe(dict(metadata or {})))
                if counterfactual_reason:
                    merged["counterfactual_reason"] = counterfactual_reason
                entry["metadata"] = merged
                entry["updated_at"] = now
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._writes += 1
            self._evict_if_needed()
            return _public_workflow_entry(entry)

    def hint(
        self,
        request_kwargs: dict[str, Any] | None,
        *,
        repo_fingerprint: str = "",
        artifact_fingerprint: str = "",
        scope: str = "",
    ) -> dict[str, Any]:
        request_kwargs = request_kwargs or {}
        intent = _extract_request_intent(request_kwargs)
        candidates = []
        with self._lock:
            for entry in self._entries.values():
                if scope and entry.get("scope") not in ("", scope):
                    continue
                if entry.get("route_key") != intent.route_key:
                    continue
                entry_repo = str(entry.get("repo_fingerprint", "") or "")
                if repo_fingerprint and entry_repo and entry_repo != repo_fingerprint:
                    continue
                entry_artifact = str(entry.get("artifact_fingerprint", "") or "")
                if (
                    artifact_fingerprint
                    and entry_artifact
                    and entry_artifact != artifact_fingerprint
                ):
                    continue
                candidates.append(entry)

        if not candidates:
            return {
                "workflow_available": False,
                "preferred_action": "",
                "route_preference": "",
                "success_rate": 0.0,
                "prefer_tool_context": False,
                "prefer_expensive": False,
                "prefer_verified_patch": False,
                "avoid_action": "",
                "counterfactual_action": "",
            }

        ranked = sorted(
            candidates,
            key=lambda item: (
                _success_rate(item),
                int(item.get("successes", 0) or 0),
                int(item.get("hits", 0) or 0),
                float(item.get("updated_at", 0) or 0),
            ),
            reverse=True,
        )
        best = ranked[0]
        with self._lock:
            current = self._entries.get(best["key"])
            if current is not None:
                current["hits"] = int(current.get("hits", 0) or 0) + 1
                current["updated_at"] = time.time()
                self._hits += 1
                best = current
        action = str(best.get("preferred_action", "") or "")
        route_preference = str(best.get("route_preference", "") or "")
        failed_actions = Counter(dict(best.get("failed_actions", {}) or {}))
        counterfactual_actions = Counter(dict(best.get("counterfactual_actions", {}) or {}))
        return {
            "workflow_available": True,
            "preferred_action": action,
            "route_preference": route_preference,
            "success_rate": round(_success_rate(best), 4),
            "prefer_tool_context": action == "tool_first" or route_preference == "tool",
            "prefer_expensive": action == "direct_expensive" or route_preference == "expensive",
            "prefer_verified_patch": action == "reuse_verified_patch",
            "avoid_action": failed_actions.most_common(1)[0][0] if failed_actions else "",
            "counterfactual_action": counterfactual_actions.most_common(1)[0][0]
            if counterfactual_actions
            else "",
            "metadata": _json_safe(dict(best.get("metadata", {}) or {})),
        }

    def snapshot(self, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            entries = [_public_workflow_entry(entry) for entry in self._entries.values()]
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
                    "category": entry.get("category", "") or "",
                    "route_key": entry.get("route_key", "") or "",
                    "canonical_key": entry.get("canonical_key", "") or "",
                    "tool_signature": entry.get("tool_signature", "") or "",
                    "repo_fingerprint": entry.get("repo_fingerprint", "") or "",
                    "artifact_fingerprint": entry.get("artifact_fingerprint", "") or "",
                    "preferred_action": entry.get("preferred_action", "") or "",
                    "route_preference": entry.get("route_preference", "") or "",
                    "successes": int(entry.get("successes", 0) or 0),
                    "failures": int(entry.get("failures", 0) or 0),
                    "failed_actions": Counter(dict(entry.get("failed_actions", {}) or {})),
                    "counterfactual_actions": Counter(
                        dict(entry.get("counterfactual_actions", {}) or {})
                    ),
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
            success_total = sum(
                int(entry.get("successes", 0) or 0) for entry in self._entries.values()
            )
            failure_total = sum(
                int(entry.get("failures", 0) or 0) for entry in self._entries.values()
            )
            return {
                "total_entries": len(self._entries),
                "writes": self._writes,
                "hits": self._hits,
                "total_successes": success_total,
                "total_failures": failure_total,
                "max_entries": self._max_entries,
            }

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

__all__ = ["WorkflowPlanStore"]
