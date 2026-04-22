
"""Artifact summarization and focus extraction for optimization memory."""

from __future__ import annotations

from collections import Counter
from typing import Any

from byte.processor._optimization_focus import (
    _artifact_focus_segments,
    _artifact_item_label,
    _artifact_item_snippet,
    _path_like_label,
    _prefix_summary,
)
from byte.processor._optimization_text import (
    _coerce_list,
    _lexical_tokens,
    compact_text,
    stable_digest,
)


def summarize_artifact_payload(artifact_type: str, value: Any, *, max_chars: int = 480) -> str:
    artifact_type = str(artifact_type or "").strip().lower()
    if value in (None, "", [], {}):
        return ""

    if artifact_type in {"changed_files", "repo_files"}:
        items = _coerce_list(value)
        labels = [_path_like_label(item) for item in items[:8]]
        labels = [label for label in labels if label]
        summary = ", ".join(labels)
        if len(items) > len(labels):
            summary = f"{summary}, +{len(items) - len(labels)} more"
        return _prefix_summary("changed files", summary, max_chars=max_chars)

    if artifact_type in {"changed_hunks", "code_hunks"}:
        items = _coerce_list(value)
        previews = []
        for item in items[:4]:
            if isinstance(item, dict):
                path = str(item.get("path") or item.get("file") or item.get("name") or "unknown")
                hunk = str(item.get("hunk") or item.get("range") or item.get("summary") or "")
                previews.append(f"{path}: {compact_text(hunk, max_chars=120)}".strip())
            else:
                previews.append(compact_text(item, max_chars=120))
        summary = " | ".join(item for item in previews if item)
        return _prefix_summary("changed hunks", summary, max_chars=max_chars)

    if artifact_type in {"retrieval_context", "document_context", "support_articles"}:
        items = _coerce_list(value)
        previews = []
        seen = set()
        for item in items:
            label = _artifact_item_label(item)
            snippet = _artifact_item_snippet(item)
            digest = stable_digest({"label": label, "snippet": snippet})
            if digest in seen:
                continue
            seen.add(digest)
            preview = f"{label}: {snippet}".strip(": ")
            previews.append(compact_text(preview, max_chars=160))
            if len(previews) >= 4:
                break
        summary = " | ".join(previews)
        if len(items) > len(previews):
            summary = f"{summary} | +{len(items) - len(previews)} more".strip(" |")
        return _prefix_summary(artifact_type.replace("_", " "), summary, max_chars=max_chars)

    if artifact_type in {"repo_snapshot", "repo_summary", "workspace_summary"}:
        if isinstance(value, dict):
            pieces = []
            for field in ("repo", "workspace", "branch", "language", "framework"):
                field_value = value.get(field)
                if field_value not in (None, "", [], {}):
                    pieces.append(f"{field}={field_value}")
            files = value.get("files") or value.get("paths") or []
            if isinstance(files, list) and files:
                pieces.append(f"files={len(files)}")
            symbols = value.get("symbols") or value.get("exports") or []
            if isinstance(symbols, list) and symbols:
                pieces.append(f"symbols={len(symbols)}")
            summary = ", ".join(pieces) or compact_text(value, max_chars=max_chars)
            return _prefix_summary(artifact_type.replace("_", " "), summary, max_chars=max_chars)
        return _prefix_summary(
            artifact_type.replace("_", " "),
            compact_text(value, max_chars=max_chars),
            max_chars=max_chars,
        )

    if artifact_type in {"prompt_pieces", "prompt_piece"}:
        pieces = _coerce_list(value)
        previews = []
        for item in pieces[:6]:
            if isinstance(item, dict):
                part_type = str(item.get("type") or item.get("piece_type") or "piece")
                content = item.get("content") if "content" in item else item.get("text")
                previews.append(f"{part_type}: {compact_text(content, max_chars=96)}")
            else:
                previews.append(compact_text(item, max_chars=96))
        summary = " | ".join(previews)
        return _prefix_summary("prompt pieces", summary, max_chars=max_chars)

    if artifact_type in {"tools", "tool_schema", "tool_schemas"}:
        items = _coerce_list(value)
        names = []
        for item in items[:8]:
            if isinstance(item, dict):
                function = item.get("function", {}) if item.get("type") == "function" else item
                name = function.get("name") or item.get("name")
                if name:
                    names.append(str(name))
            else:
                names.append(compact_text(item, max_chars=48))
        summary = ", ".join(names)
        if len(items) > len(names):
            summary = f"{summary}, +{len(items) - len(names)} more"
        return _prefix_summary("tools", summary, max_chars=max_chars)

    return _prefix_summary(
        artifact_type.replace("_", " "),
        compact_text(value, max_chars=max_chars),
        max_chars=max_chars,
    )


def summarize_artifact_sketch(artifact_type: str, value: Any, *, max_chars: int = 220) -> str:
    artifact_type = str(artifact_type or "").strip().lower()
    if value in (None, "", [], {}):
        return ""
    items = _coerce_list(value)
    if artifact_type in {
        "retrieval_context",
        "document_context",
        "support_articles",
        "tool_result_context",
    }:
        tokens = Counter()
        labels = []
        for item in items[:12]:
            label = _artifact_item_label(item)
            if label:
                labels.append(label)
            snippet = _artifact_item_snippet(item)
            tokens.update(_lexical_tokens(f"{label} {snippet}"))
        top_terms = [token for token, _ in tokens.most_common(5)]
        label_preview = ", ".join(labels[:2])
        summary = f"items={len(items)}, top_terms={','.join(top_terms)}"
        if label_preview:
            summary = f"{summary}, labels={label_preview}"
        return _prefix_summary(
            f"{artifact_type.replace('_', ' ')} sketch", summary, max_chars=max_chars
        )
    if artifact_type in {"changed_files", "changed_hunks"}:
        labels = [_path_like_label(item) for item in items[:4] if item]
        summary = f"items={len(items)}, focus={', '.join(labels[:3])}"
        return _prefix_summary(
            f"{artifact_type.replace('_', ' ')} sketch", summary, max_chars=max_chars
        )
    if artifact_type in {"repo_snapshot", "repo_summary", "workspace_summary"} and isinstance(
        value, dict
    ):
        files = value.get("files") or value.get("paths") or []
        symbols = value.get("symbols") or value.get("exports") or []
        summary = (
            f"repo={value.get('repo') or value.get('workspace') or 'unknown'}, "
            f"files={len(files) if isinstance(files, list) else 0}, "
            f"symbols={len(symbols) if isinstance(symbols, list) else 0}"
        )
        return _prefix_summary(
            f"{artifact_type.replace('_', ' ')} sketch", summary, max_chars=max_chars
        )
    return _prefix_summary(
        f"{artifact_type.replace('_', ' ')} sketch",
        compact_text(value, max_chars=max_chars),
        max_chars=max_chars,
    )


def distill_artifact_for_focus(
    artifact_type: str,
    value: Any,
    *,
    query_text: str,
    max_chars: int = 240,
    top_k: int = 2,
) -> str:
    artifact_type = str(artifact_type or "").strip().lower()
    if value in (None, "", [], {}):
        return ""
    query_tokens = _lexical_tokens(query_text)
    if not query_tokens:
        return summarize_artifact_payload(artifact_type, value, max_chars=max_chars)

    if artifact_type in {
        "retrieval_context",
        "document_context",
        "support_articles",
        "tool_result_context",
        "changed_files",
        "changed_hunks",
        "code_hunks",
        "repo_snapshot",
        "repo_summary",
        "workspace_summary",
        "prompt_pieces",
        "prompt_piece",
        "tools",
        "tool_schema",
        "tool_schemas",
    }:
        segments = _artifact_focus_segments(
            artifact_type,
            value,
            query_tokens=query_tokens,
            max_chars=max_chars,
            top_k=top_k,
        )
        if segments:
            joined = " | ".join(segments)
            if any(":" in segment for segment in segments):
                return compact_text(joined, max_chars=max_chars)
            prefix = f"{artifact_type.replace('_', ' ')} focus"
            return _prefix_summary(prefix, joined, max_chars=max_chars)
    return summarize_artifact_payload(artifact_type, value, max_chars=max_chars)

__all__ = [
    "distill_artifact_for_focus",
    "summarize_artifact_payload",
    "summarize_artifact_sketch",
]
