import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from byte.processor.intent import extract_request_intent
from byte.quantization.vector import build_vector_codec


def ai_memory_key(
    request_kwargs: dict[str, Any] | None = None,
    *,
    canonical_key: str = "",
    model: str = "",
    scope: str = "",
) -> str:
    intent = extract_request_intent(request_kwargs or {})
    payload = {
        "scope": scope or "",
        "model": model or str((request_kwargs or {}).get("model", "") or ""),
        "canonical_key": canonical_key or intent.canonical_key,
        "route_key": intent.route_key,
        "payload_digest": intent.payload_digest,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AIMemoryStore:
    """Provider-agnostic interaction memory for answers, reasoning, tools, and embeddings."""

    def __init__(
        self,
        *,
        max_entries: int = 2000,
        embedding_preview_dims: int = 32,
        embedding_codec: str = "disabled",
        embedding_bits: int = 8,
    ) -> None:
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max(1, int(max_entries or 1))
        self._embedding_preview_dims = max(0, int(embedding_preview_dims or 0))
        self._embedding_codec = str(embedding_codec or "disabled").strip().lower()
        self._embedding_bits = max(1, int(embedding_bits or 1))
        self._writes = 0
        self._touches = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._evictions = 0
        self._lock = Lock()

    def remember(
        self,
        request_kwargs: dict[str, Any] | None,
        *,
        answer: Any,
        reasoning: Any | None = None,
        tool_outputs: Any | None = None,
        embedding_data: Any | None = None,
        model: str = "",
        provider: str = "",
        scope: str = "",
        metadata: dict[str, Any] | None = None,
        source: str = "llm",
    ) -> dict[str, Any]:
        request_kwargs = request_kwargs or {}
        intent = extract_request_intent(request_kwargs)
        now = time.time()
        key = ai_memory_key(
            request_kwargs,
            canonical_key=intent.canonical_key,
            model=model,
            scope=scope,
        )
        question_text = _extract_question_text(request_kwargs)
        preview, compression = _embedding_preview(
            embedding_data,
            self._embedding_preview_dims,
            codec_name=self._embedding_codec,
            bits=self._embedding_bits,
        )
        entry = {
            "key": key,
            "scope": scope or "",
            "provider": provider or "",
            "model": model or str(request_kwargs.get("model", "") or ""),
            "category": intent.category,
            "route_key": intent.route_key,
            "canonical_key": intent.canonical_key,
            "payload_digest": intent.payload_digest,
            "tool_signature": intent.tool_signature,
            "slots": _json_safe(dict(intent.slots or {})),
            "question": question_text,
            "question_digest": _short_digest(question_text),
            "answer": _json_safe(answer),
            "reasoning": _json_safe(reasoning),
            "tool_outputs": _json_safe(tool_outputs),
            "embedding": preview,
            "embedding_compression": compression,
            "metadata": _json_safe(dict(metadata or {})),
            "last_source": source or "llm",
            "created_at": now,
            "updated_at": now,
            "hits": 0,
        }

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                entry["created_at"] = existing.get("created_at", now)
                entry["hits"] = int(existing.get("hits", 0) or 0)
                self._entries.pop(key, None)
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._writes += 1
            self._evict_if_needed()
            return _public_entry(entry)

    def touch(
        self,
        request_kwargs: dict[str, Any] | None,
        *,
        answer: Any | None = None,
        reasoning: Any | None = None,
        tool_outputs: Any | None = None,
        embedding_data: Any | None = None,
        model: str = "",
        provider: str = "",
        scope: str = "",
        metadata: dict[str, Any] | None = None,
        source: str = "cache",
    ) -> dict[str, Any]:
        request_kwargs = request_kwargs or {}
        intent = extract_request_intent(request_kwargs)
        key = ai_memory_key(
            request_kwargs,
            canonical_key=intent.canonical_key,
            model=model,
            scope=scope,
        )
        now = time.time()
        with self._lock:
            existing = self._entries.get(key)
            if existing is None:
                self._cache_misses += 1
            else:
                existing["hits"] = int(existing.get("hits", 0) or 0) + 1
                existing["updated_at"] = now
                existing["last_source"] = source or "cache"
                if metadata:
                    merged = dict(existing.get("metadata", {}))
                    merged.update(_json_safe(dict(metadata)))
                    existing["metadata"] = merged
                self._entries.move_to_end(key)
                self._touches += 1
                self._cache_hits += 1
                return _public_entry(existing)

        remembered = self.remember(
            request_kwargs,
            answer=answer,
            reasoning=reasoning,
            tool_outputs=tool_outputs,
            embedding_data=embedding_data,
            model=model,
            provider=provider,
            scope=scope,
            metadata=metadata,
            source=source,
        )
        with self._lock:
            entry = self._entries.get(remembered["key"])
            if entry is not None:
                entry["hits"] = int(entry.get("hits", 0) or 0) + 1
                self._touches += 1
                self._cache_hits += 1
                return _public_entry(entry)
        return remembered

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        limit = max(1, int(limit or 1))
        with self._lock:
            return [_public_entry(entry) for entry in list(self._entries.values())[-limit:]][::-1]

    def snapshot(self, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            entries = [_public_entry(entry) for entry in self._entries.values()]
            if limit is not None:
                entries = entries[-int(limit or 0) :]
            stats = _stats_payload(
                self._entries,
                writes=self._writes,
                touches=self._touches,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                evictions=self._evictions,
                max_entries=self._max_entries,
                embedding_preview_dims=self._embedding_preview_dims,
                embedding_codec=self._embedding_codec,
                embedding_bits=self._embedding_bits,
            )
            return {
                "entries": entries[::-1],
                "stats": stats,
            }

    def merge(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        payload = payload or {}
        entries = payload.get("entries", []) or []
        imported = 0
        skipped = 0
        with self._lock:
            for entry in entries:
                key = entry.get("key") or ai_memory_key(
                    {"model": entry.get("model", "")},
                    canonical_key=entry.get("canonical_key", ""),
                    model=entry.get("model", ""),
                    scope=entry.get("scope", ""),
                )
                existing = self._entries.get(key)
                existing_updated = (existing or {}).get("updated_at", 0) or 0
                incoming_updated = entry.get("updated_at", 0) or 0
                if existing is not None and existing_updated > incoming_updated:
                    skipped += 1
                    continue
                normalized = {
                    "key": key,
                    "scope": entry.get("scope", "") or "",
                    "provider": entry.get("provider", "") or "",
                    "model": entry.get("model", "") or "",
                    "category": entry.get("category", "") or "",
                    "route_key": entry.get("route_key", "") or "",
                    "canonical_key": entry.get("canonical_key", "") or "",
                    "payload_digest": entry.get("payload_digest", "") or "",
                    "tool_signature": entry.get("tool_signature", "") or "",
                    "slots": _json_safe(dict(entry.get("slots", {}) or {})),
                    "question": entry.get("question", "") or "",
                    "question_digest": entry.get("question_digest", "")
                    or _short_digest(entry.get("question", "")),
                    "answer": _json_safe(entry.get("answer")),
                    "reasoning": _json_safe(entry.get("reasoning")),
                    "tool_outputs": _json_safe(entry.get("tool_outputs")),
                    "embedding": _json_safe(entry.get("embedding")),
                    "embedding_compression": _json_safe(entry.get("embedding_compression") or {}),
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
            self._touches = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._evictions = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return _stats_payload(
                self._entries,
                writes=self._writes,
                touches=self._touches,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                evictions=self._evictions,
                max_entries=self._max_entries,
                embedding_preview_dims=self._embedding_preview_dims,
                embedding_codec=self._embedding_codec,
                embedding_bits=self._embedding_bits,
            )

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1


def _extract_question_text(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    if messages:
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return " ".join(part for part in parts if part).strip()
        return str(content or "")
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input") or "")
    return ""


def _embedding_preview(
    value: Any,
    preview_dims: int,
    *,
    codec_name: str,
    bits: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if value is None:
        return {}, {}
    try:
        if hasattr(value, "tolist"):
            raw = value.tolist()
        elif isinstance(value, (list, tuple)):
            raw = list(value)
        elif isinstance(value, str):
            return (
                {
                    "kind": "string",
                    "digest": _short_digest(value),
                    "dimensions": 1,
                    "preview": [value[: min(len(value), max(preview_dims, 1))]],
                },
                {},
            )
        else:
            return (
                {
                    "kind": type(value).__name__,
                    "digest": _short_digest(str(value)),
                    "dimensions": 1,
                    "preview": [_json_safe(value)],
                },
                {},
            )

        if raw and isinstance(raw[0], list):
            flat = raw[0]
        else:
            flat = raw
        dims = len(flat)
        preview = flat[:preview_dims] if preview_dims > 0 else []
        summary = {
            "kind": "vector",
            "digest": _short_digest(json.dumps(flat, separators=(",", ":"), default=str)),
            "dimensions": dims,
            "preview": _json_safe(preview),
        }
        compression = _embedding_compression(flat, codec_name=codec_name, bits=bits)
        return summary, compression
    except Exception:
        return (
            {
                "kind": type(value).__name__,
                "digest": _short_digest(str(value)),
                "dimensions": 1,
                "preview": [_json_safe(value)],
            },
            {},
        )


def _embedding_compression(value: Any, *, codec_name: str, bits: int) -> dict[str, Any]:
    codec = build_vector_codec(codec_name, bits=bits)
    if codec is None:
        return {}
    try:
        payload = codec.encode(value)
    except Exception:
        return {}
    summary = payload.summary()
    raw_bytes = int(getattr(payload, "raw_nbytes", 0) or 0)
    compressed_bytes = int(getattr(payload, "compressed_nbytes", 0) or 0)
    summary["compression_ratio"] = (
        round(float(compressed_bytes) / float(raw_bytes), 6) if raw_bytes > 0 else 0.0
    )
    return summary


def _short_digest(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _public_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": entry["key"],
        "scope": entry.get("scope", ""),
        "provider": entry.get("provider", ""),
        "model": entry.get("model", ""),
        "category": entry.get("category", ""),
        "route_key": entry.get("route_key", ""),
        "canonical_key": entry.get("canonical_key", ""),
        "payload_digest": entry.get("payload_digest", ""),
        "tool_signature": entry.get("tool_signature", ""),
        "slots": _json_safe(entry.get("slots") or {}),
        "question": entry.get("question", ""),
        "question_digest": entry.get("question_digest", ""),
        "answer": _json_safe(entry.get("answer")),
        "reasoning": _json_safe(entry.get("reasoning")),
        "tool_outputs": _json_safe(entry.get("tool_outputs")),
        "embedding": _json_safe(entry.get("embedding")),
        "embedding_compression": _json_safe(entry.get("embedding_compression") or {}),
        "metadata": _json_safe(entry.get("metadata", {})),
        "last_source": entry.get("last_source", ""),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "hits": int(entry.get("hits", 0) or 0),
    }


def _stats_payload(
    entries: "OrderedDict[str, dict[str, Any]]",
    *,
    writes: int,
    touches: int,
    cache_hits: int,
    cache_misses: int,
    evictions: int,
    max_entries: int,
    embedding_preview_dims: int,
    embedding_codec: str,
    embedding_bits: int,
) -> dict[str, Any]:
    total_lookups = cache_hits + cache_misses
    categories: dict[str, int] = {}
    compressed_entries = 0
    compression_ratio_sum = 0.0
    for entry in entries.values():
        category = entry.get("category", "") or "unknown"
        categories[category] = categories.get(category, 0) + 1
        compression = dict(entry.get("embedding_compression", {}) or {})
        if compression:
            compressed_entries += 1
            compression_ratio_sum += float(compression.get("compression_ratio", 0.0) or 0.0)
    return {
        "total_entries": len(entries),
        "writes": writes,
        "touches": touches,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "lookup_hit_ratio": round(cache_hits / total_lookups, 4) if total_lookups else 0.0,
        "evictions": evictions,
        "max_entries": max_entries,
        "embedding_preview_dims": embedding_preview_dims,
        "embedding_codec": embedding_codec,
        "embedding_bits": embedding_bits,
        "compressed_entries": compressed_entries,
        "avg_embedding_compression_ratio": round(
            compression_ratio_sum / compressed_entries, 6
        )
        if compressed_entries
        else 0.0,
        "categories": categories,
    }
