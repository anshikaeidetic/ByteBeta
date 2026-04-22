"""Verified execution memory store for Byte workflows."""

import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from byte.processor.intent import extract_request_intent

from ._execution_keys import _json_safe, _short_digest, _stable_hash, execution_request_key


class ExecutionMemoryStore:
    """Track verified outputs so risky cache reuse can be gated."""

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
        answer: Any,
        verification: Any | None = None,
        patch: Any | None = None,
        test_command: Any | None = None,
        test_result: Any | None = None,
        lint_result: Any | None = None,
        schema_validation: Any | None = None,
        tool_checks: Any | None = None,
        repo_fingerprint: str = "",
        model: str = "",
        provider: str = "",
        scope: str = "",
        metadata: dict[str, Any] | None = None,
        source: str = "llm",
    ) -> dict[str, Any]:
        request_kwargs = request_kwargs or {}
        intent = extract_request_intent(request_kwargs)
        request_key = execution_request_key(
            request_kwargs,
            model=model,
            scope=scope,
            repo_fingerprint=repo_fingerprint,
        )
        verification_payload = _normalize_verification(
            verification=verification,
            test_result=test_result,
            lint_result=lint_result,
            schema_validation=schema_validation,
            tool_checks=tool_checks,
        )
        now = time.time()
        answer_text = "" if answer is None else str(answer)
        entry_key = _stable_hash(
            {
                "request_key": request_key,
                "answer_digest": _short_digest(answer_text),
                "source": source or "llm",
            }
        )
        entry = {
            "key": entry_key,
            "request_key": request_key,
            "scope": scope or "",
            "provider": provider or "",
            "model": model or str(request_kwargs.get("model", "") or ""),
            "repo_fingerprint": repo_fingerprint or "",
            "category": intent.category,
            "route_key": intent.route_key,
            "canonical_key": intent.canonical_key,
            "payload_digest": intent.payload_digest,
            "slots": _json_safe(dict(intent.slots or {})),
            "answer_digest": _short_digest(answer_text),
            "answer_preview": answer_text[:240],
            "verified": verification_payload["verified"],
            "verification_status": verification_payload["status"],
            "checks": verification_payload["checks"],
            "patch": _json_safe(patch),
            "test_command": _json_safe(test_command),
            "test_result": _json_safe(test_result),
            "lint_result": _json_safe(lint_result),
            "schema_validation": _json_safe(schema_validation),
            "tool_checks": _json_safe(tool_checks),
            "metadata": _json_safe(dict(metadata or {})),
            "last_source": source or "llm",
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
            return _public_entry(entry)

    def lookup(
        self,
        request_kwargs: dict[str, Any] | None,
        *,
        answer: Any | None = None,
        repo_fingerprint: str = "",
        model: str = "",
        scope: str = "",
        verified_only: bool = False,
    ) -> dict[str, Any] | None:
        request_key = execution_request_key(
            request_kwargs or {},
            model=model,
            scope=scope,
            repo_fingerprint=repo_fingerprint,
        )
        answer_digest = (
            _short_digest("" if answer is None else str(answer)) if answer is not None else ""
        )
        with self._lock:
            for entry in reversed(list(self._entries.values())):
                if entry.get("request_key") != request_key:
                    continue
                if answer_digest and entry.get("answer_digest") != answer_digest:
                    continue
                if verified_only and not entry.get("verified", False):
                    continue
                entry["hits"] = int(entry.get("hits", 0) or 0) + 1
                entry["updated_at"] = time.time()
                self._hits += 1
                return _public_entry(entry)
        return None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            verified = sum(1 for entry in self._entries.values() if entry.get("verified"))
            return {
                "total_entries": len(self._entries),
                "verified_entries": verified,
                "unverified_entries": len(self._entries) - verified,
                "writes": self._writes,
                "hits": self._hits,
                "max_entries": self._max_entries,
            }

    def snapshot(self, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            entries = [_public_entry(entry) for entry in self._entries.values()]
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
                key = entry.get("key") or _stable_hash(entry)
                existing = self._entries.get(key)
                incoming_updated = entry.get("updated_at", 0) or 0
                if existing is not None and (existing.get("updated_at", 0) or 0) > incoming_updated:
                    skipped += 1
                    continue
                normalized = {
                    "key": key,
                    "request_key": entry.get("request_key", "") or "",
                    "scope": entry.get("scope", "") or "",
                    "provider": entry.get("provider", "") or "",
                    "model": entry.get("model", "") or "",
                    "repo_fingerprint": entry.get("repo_fingerprint", "") or "",
                    "category": entry.get("category", "") or "",
                    "route_key": entry.get("route_key", "") or "",
                    "canonical_key": entry.get("canonical_key", "") or "",
                    "payload_digest": entry.get("payload_digest", "") or "",
                    "slots": _json_safe(dict(entry.get("slots", {}) or {})),
                    "answer_digest": entry.get("answer_digest", "") or "",
                    "answer_preview": entry.get("answer_preview", "") or "",
                    "verified": bool(entry.get("verified", False)),
                    "verification_status": entry.get("verification_status", "unverified")
                    or "unverified",
                    "checks": dict(entry.get("checks", {}) or {}),
                    "patch": _json_safe(entry.get("patch")),
                    "test_command": _json_safe(entry.get("test_command")),
                    "test_result": _json_safe(entry.get("test_result")),
                    "lint_result": _json_safe(entry.get("lint_result")),
                    "schema_validation": _json_safe(entry.get("schema_validation")),
                    "tool_checks": _json_safe(entry.get("tool_checks")),
                    "metadata": _json_safe(dict(entry.get("metadata", {}) or {})),
                    "last_source": entry.get("last_source", "import") or "import",
                    "created_at": entry.get("created_at", time.time()),
                    "updated_at": incoming_updated or entry.get("created_at", time.time()),
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

def _normalize_verification(
    *,
    verification: Any | None = None,
    test_result: Any | None = None,
    lint_result: Any | None = None,
    schema_validation: Any | None = None,
    tool_checks: Any | None = None,
) -> dict[str, Any]:
    checks = {}
    explicit_verified = None
    explicit_status = ""
    if isinstance(verification, dict):
        explicit_verified = verification.get("verified")
        explicit_status = str(verification.get("status") or "").strip().lower()
        if test_result is None:
            test_result = verification.get("test_result")
        if lint_result is None:
            lint_result = verification.get("lint_result")
        if schema_validation is None:
            schema_validation = verification.get("schema_validation")
        if tool_checks is None:
            tool_checks = verification.get("tool_checks")
    elif isinstance(verification, bool):
        explicit_verified = verification

    checks["tests"] = _result_to_flag(test_result)
    checks["lint"] = _result_to_flag(lint_result)
    checks["schema"] = _result_to_flag(schema_validation)
    checks["tools"] = _tool_checks_to_flag(tool_checks)

    populated = [flag for flag in checks.values() if flag is not None]
    if explicit_verified is not None:
        verified = bool(explicit_verified)
    elif populated:
        verified = all(populated)
    else:
        verified = False

    status = explicit_status or (
        "verified"
        if verified
        else "failed"
        if any(flag is False for flag in populated)
        else "unverified"
    )
    return {
        "verified": verified,
        "status": status,
        "checks": checks,
    }


def _tool_checks_to_flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("passed", "ok", "success", "verified"):
            if key in value:
                return bool(value.get(key))
        return all(_tool_checks_to_flag(item) is not False for item in value.values())
    if isinstance(value, list):
        flags = [_tool_checks_to_flag(item) for item in value]
        populated = [flag for flag in flags if flag is not None]
        return all(populated) if populated else None
    return bool(value)


def _result_to_flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("passed", "ok", "success", "verified"):
            if key in value:
                return bool(value.get(key))
        if "failures" in value:
            failures = value.get("failures")
            if isinstance(failures, int):
                return failures == 0
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"pass", "passed", "ok", "success", "verified"}:
            return True
        if lowered in {"fail", "failed", "error"}:
            return False
    return bool(value)

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
