"""Auxiliary-context compilation helpers for request context compilation."""

from __future__ import annotations

import hashlib
from typing import Any

from byte.processor._optimization_summary import (
    compact_text,
    distill_artifact_for_focus,
    stable_digest,
    summarize_artifact_payload,
    summarize_artifact_sketch,
)
from byte.processor._pre_relevance import (
    _compact_json,
    _lexical_overlap,
    _lexical_tokens,
    _negative_focus_penalty,
)
from byte.processor._pre_selection import _select_relevant_context_items
from byte.utils.multimodal import content_signature


def _aux_context_priority(artifact_type: str, focus_score: float, stats: dict[str, Any]) -> float:
    base = {
        "changed_hunks": 1.0,
        "tool_result_context": 0.98,
        "retrieval_context": 0.96,
        "document_context": 0.94,
        "support_articles": 0.92,
        "changed_files": 0.88,
        "repo_summary": 0.82,
        "repo_snapshot": 0.8,
        "prompt_pieces": 0.76,
    }.get(str(artifact_type or ""), 0.72)
    priority = base + min(max(float(focus_score or 0.0), 0.0), 1.0) * 0.35
    if stats.get("session_delta_hits"):
        priority += 0.08
    if stats.get("related_artifact_hits"):
        priority += 0.04
    if stats.get("artifact_reuse_hits"):
        priority += 0.02
    return round(priority, 4)

def _request_file_name(data: dict[str, Any]) -> str:
    file_obj = data.get("file")
    if isinstance(file_obj, dict):
        return str(file_obj.get("name") or "")
    return str(getattr(file_obj, "name", "") or "")


def _request_file_bytes(data: dict[str, Any]) -> bytes:
    file_obj = data.get("file")
    if isinstance(file_obj, dict):
        payload = file_obj.get("bytes", b"")
        return payload if isinstance(payload, bytes) else bytes(payload)
    if hasattr(file_obj, "read"):
        position = file_obj.tell() if hasattr(file_obj, "tell") else None
        payload = file_obj.read()
        if position is not None and hasattr(file_obj, "seek"):
            file_obj.seek(position)
        return payload if isinstance(payload, bytes) else bytes(payload)
    if hasattr(file_obj, "peek"):
        payload = file_obj.peek()
        return payload if isinstance(payload, bytes) else bytes(payload)
    return b""


def _fallback_request_signature(data: dict[str, Any]) -> str:
    parts = []

    if data.get("prompt") is not None:
        parts.append(str(data.get("prompt")))

    if data.get("input") is not None:
        input_data = data.get("input")
        if isinstance(input_data, list):
            parts.append(content_signature(input_data))
        else:
            parts.append(str(input_data))

    if data.get("file") is not None:
        file_name = _request_file_name(data)
        file_bytes = _request_file_bytes(data)
        parts.append(f"file::{file_name}::sha256:{hashlib.sha256(file_bytes).hexdigest()[:16]}")

    return "\n".join(part for part in parts if part)


def _compile_aux_context(
    ctx_key: str,
    value: Any,
    *,
    max_chars: int,
    artifact_memory_store: Any | None = None,
    session_delta_store: Any | None = None,
    session_key: str = "",
    memory_scope: str = "",
    seen_digests: set | None = None,
    request_focus: str = "",
    relevance_top_k: int = 4,
    related_memory: bool = True,
    related_min_score: float = 0.18,
    negative_context_digests: dict[str, list[str]] | None = None,
    context_sketches: bool = True,
    focus_distillation: bool = True,
    stable_prefix: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_type = ctx_key.replace("byte_", "", 1)
    stats = {
        "artifact_entries": 0,
        "artifact_reuse_hits": 0,
        "related_artifact_hits": 0,
        "session_delta_hits": 0,
        "saved_chars": 0,
        "relevance_pruned_items": 0,
        "negative_pruned_items": 0,
        "sketches_used": 0,
        "focused_distillation_hits": 0,
    }
    selected_value, pruned_items, negative_pruned = _select_relevant_context_items(
        artifact_type,
        value,
        request_focus=request_focus,
        top_k=relevance_top_k,
        negative_digests=(negative_context_digests or {}).get(artifact_type, []),
    )
    value = selected_value
    stats["relevance_pruned_items"] = pruned_items
    stats["negative_pruned_items"] = negative_pruned
    artifact_digest = stable_digest({"artifact_type": artifact_type, "value": value})
    if seen_digests is not None:
        if artifact_digest in seen_digests:
            return {}, stats
        seen_digests.add(artifact_digest)

    summary = summarize_artifact_payload(artifact_type, value, max_chars=min(max_chars // 2, 640))
    sketch = (
        summarize_artifact_sketch(artifact_type, value, max_chars=min(max_chars // 3, 220))
        if context_sketches
        else ""
    )
    original_chars = len(_compact_json(value))
    note_body = summary
    if focus_distillation and request_focus:
        distilled = distill_artifact_for_focus(
            artifact_type,
            value,
            query_text=request_focus,
            max_chars=min(max_chars // 3, 320),
            top_k=2
            if artifact_type
            in {
                "retrieval_context",
                "document_context",
                "support_articles",
                "tool_result_context",
                "changed_hunks",
            }
            else 1,
        )
        if distilled:
            note_body = distilled
            if distilled != summary:
                stats["focused_distillation_hits"] += 1
    if sketch:
        note = f"{sketch}\n{note_body}".strip()
        stats["sketches_used"] += 1
    else:
        note = note_body

    if artifact_memory_store is not None:
        existing_artifact = artifact_memory_store.get(
            artifact_type,
            fingerprint=artifact_digest,
            scope=memory_scope or "",
        )
        if existing_artifact:
            stats["artifact_reuse_hits"] += 1
            summary = str(existing_artifact.get("summary") or summary or "")
            sketch = str(existing_artifact.get("sketch") or sketch or "")
            if not stats["focused_distillation_hits"]:
                note_body = summary
            note = f"{sketch}\n{note_body}".strip() if sketch else note_body
        else:
            if related_memory and request_focus:
                related = artifact_memory_store.find_related(
                    artifact_type,
                    query_text=request_focus,
                    scope=memory_scope or "",
                    top_k=1,
                    min_score=related_min_score,
                )
                if related:
                    related_entry = related[0]
                    related_summary = str(related_entry.get("summary") or "").strip()
                    related_sketch = str(related_entry.get("sketch") or "").strip()
                    if (
                        related_summary
                        and not stats["focused_distillation_hits"]
                        and len(related_summary) <= len(summary)
                    ):
                        summary = related_summary
                        note_body = related_summary
                        note = (
                            f"{related_sketch}\n{related_summary}".strip()
                            if related_sketch
                            else related_summary
                        )
                    stats["related_artifact_hits"] += 1
            artifact_memory_store.remember(
                artifact_type,
                value,
                fingerprint=artifact_digest,
                summary=summary,
                sketch=sketch,
                scope=memory_scope or "",
                source="context_compiler",
                metadata={"context_key": ctx_key},
            )
            stats["artifact_entries"] += 1

    focus_score = 0.0
    if request_focus:
        focus_score = max(
            0.0,
            _lexical_overlap(_lexical_tokens(request_focus), _lexical_tokens(note_body))
            - _negative_focus_penalty(note_body),
        )

    if session_delta_store is not None and session_key:
        delta = session_delta_store.note(
            session_key,
            artifact_type,
            value,
            scope=memory_scope or "",
            metadata={"context_key": ctx_key},
        )
        if stable_prefix:
            if not delta.get("changed", True):
                stats["session_delta_hits"] += 1
        elif not delta.get("changed", True):
            stats["session_delta_hits"] += 1
            note = (
                f"Byte session delta: {artifact_type.replace('_', ' ')} unchanged "
                f"(digest {str(delta.get('current_digest', ''))[:8]}). "
                f"Reuse prior context; latest summary: {summary}"
            )
        elif delta.get("previous_digest"):
            note = (
                f"Byte session delta: {artifact_type.replace('_', ' ')} updated "
                f"from {str(delta.get('previous_digest', ''))[:8]} to {str(delta.get('current_digest', ''))[:8]}. "
                f"{summary}"
            )

    note_limit = min(max_chars // 2, 720)
    if stats["focused_distillation_hits"] and note_body and sketch:
        note = compact_text(f"{sketch}\n{note_body}".strip(), max_chars=note_limit)
    elif stats["focused_distillation_hits"] and note_body:
        note = compact_text(note_body, max_chars=note_limit)
    else:
        note = compact_text(note, max_chars=note_limit)
    stats["saved_chars"] = max(original_chars - len(note), 0)
    return {
        "artifact_type": artifact_type,
        "note": note,
        "digest": stable_digest({"artifact_type": artifact_type, "note": note}),
        "priority": _aux_context_priority(artifact_type, focus_score, stats),
        "focus_score": round(float(focus_score), 4),
    }, stats
