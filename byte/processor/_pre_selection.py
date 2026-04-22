from __future__ import annotations

"""Context-item selection helpers extracted from ``byte.processor.pre``."""

from typing import Any

from byte.processor._optimization_summary import stable_digest, summarize_artifact_payload
from byte.processor._pre_relevance import _lexical_overlap, _lexical_tokens, _negative_focus_penalty
from byte.utils.multimodal import content_signature


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def _request_focus_text(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    if messages:
        content = _message_content(messages[-1])
        return content_signature(content) if isinstance(content, list) else str(content or "")
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input") or "")
    return ""


def _select_relevant_context_items(
    artifact_type: str,
    value: Any,
    *,
    request_focus: str,
    top_k: int,
    negative_digests: list[str] | None = None,
) -> tuple[Any, int, int]:
    if artifact_type not in {
        "retrieval_context",
        "document_context",
        "support_articles",
        "tool_result_context",
        "changed_files",
        "changed_hunks",
    }:
        return value, 0, 0
    negative = {str(item) for item in (negative_digests or []) if str(item)}
    if not isinstance(value, list):
        return value, 0, 0
    filtered = []
    negative_pruned = 0
    for item in value:
        digest = stable_digest(item)
        if digest in negative:
            negative_pruned += 1
            continue
        filtered.append(item)
    value = filtered or value
    if len(value) <= max(1, int(top_k or 1)):
        return value, 0, negative_pruned

    query_tokens = _lexical_tokens(request_focus)
    if not query_tokens:
        kept = value[: max(1, int(top_k or 1))]
        return kept, max(0, len(value) - len(kept)), negative_pruned

    ranked = []
    for index, item in enumerate(value):
        preview = summarize_artifact_payload(artifact_type, item, max_chars=220)
        score = _lexical_overlap(query_tokens, _lexical_tokens(preview))
        score -= _negative_focus_penalty(preview)
        if artifact_type in {"changed_hunks", "changed_files"} and isinstance(item, dict):
            if any(
                token in preview.lower()
                for token in ("error", "diagnostic", "traceback", "test", "fail")
            ):
                score += 0.08
        ranked.append((score, index, item))

    ranked = sorted(ranked, key=lambda row: (row[0], -row[1]), reverse=True)
    keep = sorted(ranked[: max(1, int(top_k or 1))], key=lambda row: row[1])
    selected = [row[2] for row in keep]
    return selected, max(0, len(value) - len(selected)), negative_pruned


__all__ = ["_message_content", "_request_focus_text", "_select_relevant_context_items"]
