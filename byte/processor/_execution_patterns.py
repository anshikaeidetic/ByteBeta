"""Patch-pattern capture and replay helpers for Byte workflows."""

import difflib
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from byte.processor.intent import extract_request_intent

from ._execution_keys import _json_safe, _stable_hash


class PatchPatternStore:
    """Reuse verified code diffs as deterministic patch suggestions."""

    def __init__(self, *, max_entries: int = 512) -> None:
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max(1, int(max_entries or 1))
        self._writes = 0
        self._suggestions = 0
        self._lock = Lock()

    def remember(
        self,
        request_kwargs: dict[str, Any] | None,
        *,
        patch: Any,
        repo_fingerprint: str = "",
        verified: bool = False,
        model: str = "",
        provider: str = "",
        scope: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        patch_text = "" if patch is None else str(patch)
        replacements = _extract_patch_replacements(patch_text)
        if not replacements:
            return None
        request_kwargs = request_kwargs or {}
        intent = extract_request_intent(request_kwargs)
        entry_key = _stable_hash(
            {
                "scope": scope or "",
                "route_key": intent.route_key,
                "repo_fingerprint": repo_fingerprint or "",
                "replacements": replacements,
            }
        )
        now = time.time()
        entry = {
            "key": entry_key,
            "scope": scope or "",
            "provider": provider or "",
            "model": model or str(request_kwargs.get("model", "") or ""),
            "repo_fingerprint": repo_fingerprint or "",
            "category": intent.category,
            "route_key": intent.route_key,
            "canonical_key": intent.canonical_key,
            "slots": _json_safe(dict(intent.slots or {})),
            "verified": bool(verified),
            "replacements": replacements,
            "patch": patch_text,
            "metadata": _json_safe(dict(metadata or {})),
            "created_at": now,
            "updated_at": now,
            "hits": 0,
        }
        with self._lock:
            existing = self._entries.get(entry_key)
            if existing is not None:
                entry["created_at"] = existing.get("created_at", now)
                entry["hits"] = int(existing.get("hits", 0) or 0)
                self._entries.pop(entry_key, None)
            self._entries[entry_key] = entry
            self._entries.move_to_end(entry_key)
            self._writes += 1
            self._evict_if_needed()
            return _public_patch_entry(entry)

    def suggest(
        self,
        request_kwargs: dict[str, Any] | None,
        *,
        repo_fingerprint: str = "",
        scope: str = "",
        verified_only: bool = True,
    ) -> dict[str, Any] | None:
        request_kwargs = request_kwargs or {}
        intent = extract_request_intent(request_kwargs)
        code_context = _extract_code_context(request_kwargs)
        if not code_context:
            return None
        with self._lock:
            candidates = list(self._entries.values())[::-1]
        for entry in candidates:
            if scope and entry.get("scope") not in ("", scope):
                continue
            if verified_only and not entry.get("verified", False):
                continue
            if (
                entry.get("route_key") != intent.route_key
                and entry.get("category") != intent.category
            ):
                continue
            entry_repo = entry.get("repo_fingerprint", "")
            if repo_fingerprint and entry_repo and entry_repo != repo_fingerprint:
                continue
            patched_code, replacements_applied = _apply_replacements(
                code_context,
                entry.get("replacements", []),
            )
            if not replacements_applied:
                continue
            diff_lines = list(
                difflib.unified_diff(
                    code_context.splitlines(),
                    patched_code.splitlines(),
                    fromfile="original",
                    tofile="byte_suggested",
                    lineterm="",
                )
            )
            with self._lock:
                current = self._entries.get(entry["key"])
                if current is not None:
                    current["hits"] = int(current.get("hits", 0) or 0) + 1
                    current["updated_at"] = time.time()
                self._suggestions += 1
            return {
                "pattern_id": entry["key"],
                "route_key": entry.get("route_key", ""),
                "verified": bool(entry.get("verified", False)),
                "patched_code": patched_code,
                "patch_text": "\n".join(diff_lines),
                "replacements_applied": replacements_applied,
                "repo_fingerprint": entry_repo,
            }
        return None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            verified = sum(1 for entry in self._entries.values() if entry.get("verified"))
            return {
                "total_entries": len(self._entries),
                "verified_entries": verified,
                "writes": self._writes,
                "suggestions": self._suggestions,
                "max_entries": self._max_entries,
            }

    def snapshot(self, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            entries = [_public_patch_entry(entry) for entry in self._entries.values()]
            if limit is not None:
                entries = entries[-int(limit or 0) :]
        return {
            "entries": entries[::-1],
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
                    "repo_fingerprint": entry.get("repo_fingerprint", "") or "",
                    "category": entry.get("category", "") or "",
                    "route_key": entry.get("route_key", "") or "",
                    "canonical_key": entry.get("canonical_key", "") or "",
                    "slots": _json_safe(dict(entry.get("slots", {}) or {})),
                    "verified": bool(entry.get("verified", False)),
                    "replacements": list(entry.get("replacements", []) or []),
                    "patch": entry.get("patch", "") or "",
                    "metadata": _json_safe(dict(entry.get("metadata", {}) or {})),
                    "created_at": entry.get("created_at", time.time()),
                    "updated_at": entry.get("updated_at", time.time()),
                    "hits": int(entry.get("hits", 0) or 0),
                }
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
            self._suggestions = 0

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

def _extract_patch_replacements(patch_text: str) -> list[dict[str, str]]:
    replacements = []
    removed: list[str] = []
    added: list[str] = []
    for line in (patch_text or "").splitlines():
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
            continue
        if line.startswith("+"):
            added.append(line[1:])
            continue
    for old, new in zip(removed, added):
        if old.strip() == new.strip():
            continue
        replacements.append({"old": old, "new": new})
    return replacements


def _extract_code_context(request_kwargs: dict[str, Any]) -> str:
    for key in ("prompt", "input"):
        value = request_kwargs.get(key)
        if value:
            return _extract_code_block(str(value))
    messages = request_kwargs.get("messages") or []
    if messages:
        return _extract_code_block(str(messages[-1].get("content", "") or ""))
    return ""


def _extract_code_block(text: str) -> str:
    marker = "```"
    if marker not in text:
        return ""
    parts = text.split(marker)
    for block in parts[1:]:
        block = block.lstrip()
        lines = block.splitlines()
        if not lines:
            continue
        if lines[0].strip().isidentifier():
            lines = lines[1:]
        code = "\n".join(lines).strip()
        if code:
            return code
    return ""


def _apply_replacements(code: str, replacements: list[dict[str, str]]) -> tuple:
    updated = code
    applied = []
    for replacement in replacements:
        old = replacement.get("old", "")
        new = replacement.get("new", "")
        if old and old in updated:
            updated = updated.replace(old, new)
            applied.append(replacement)
    return updated, applied


def _public_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": entry["key"],
        "request_key": entry["request_key"],
        "scope": entry.get("scope", ""),
        "provider": entry.get("provider", ""),
        "model": entry.get("model", ""),
        "repo_fingerprint": entry.get("repo_fingerprint", ""),
        "category": entry.get("category", ""),
        "route_key": entry.get("route_key", ""),
        "canonical_key": entry.get("canonical_key", ""),
        "payload_digest": entry.get("payload_digest", ""),
        "slots": _json_safe(entry.get("slots") or {}),
        "answer_digest": entry.get("answer_digest", ""),
        "answer_preview": entry.get("answer_preview", ""),
        "verified": bool(entry.get("verified", False)),
        "verification_status": entry.get("verification_status", "unverified"),
        "checks": dict(entry.get("checks", {})),
        "patch": entry.get("patch"),
        "test_command": entry.get("test_command"),
        "test_result": entry.get("test_result"),
        "lint_result": entry.get("lint_result"),
        "schema_validation": entry.get("schema_validation"),
        "tool_checks": entry.get("tool_checks"),
        "metadata": dict(entry.get("metadata", {})),
        "last_source": entry.get("last_source", "llm"),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "hits": int(entry.get("hits", 0) or 0),
    }


def _public_patch_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": entry["key"],
        "scope": entry.get("scope", ""),
        "provider": entry.get("provider", ""),
        "model": entry.get("model", ""),
        "repo_fingerprint": entry.get("repo_fingerprint", ""),
        "category": entry.get("category", ""),
        "route_key": entry.get("route_key", ""),
        "canonical_key": entry.get("canonical_key", ""),
        "slots": _json_safe(entry.get("slots") or {}),
        "verified": bool(entry.get("verified", False)),
        "replacements": list(entry.get("replacements", [])),
        "patch": entry.get("patch", ""),
        "metadata": dict(entry.get("metadata", {})),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "hits": int(entry.get("hits", 0) or 0),
    }
