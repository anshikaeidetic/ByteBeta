"""Message dedupe and summarization helpers for request context compilation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from byte.processor._pre_canonicalize import (
    _CODE_BLOCK_PATTERN,
    _code_digest,
    _normalize_code_text,
    normalize_text,
)
from byte.utils.multimodal import content_signature


def _dedupe_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    deduped = []
    seen = set()
    deduped_count = 0
    collapsed_code_blocks = 0
    seen_code = set()
    for message in messages:
        raw_content = message.get("content", "")
        if isinstance(raw_content, list):
            normalized = normalize_text(content_signature(raw_content))
        else:
            normalized = normalize_text(raw_content)
        key = (message.get("role", "user"), normalized)
        if key in seen:
            deduped_count += 1
            continue
        seen.add(key)
        if isinstance(raw_content, list):
            content = deepcopy(raw_content)
            collapsed = 0
        else:
            content, collapsed = _collapse_duplicate_code_blocks(str(raw_content), seen_code)
        if collapsed:
            collapsed_code_blocks += collapsed
        updated = deepcopy(message)
        updated["content"] = content
        deduped.append(updated)
    return deduped, deduped_count, collapsed_code_blocks


def _collapse_duplicate_code_blocks(text: str, seen_code: set) -> tuple[str, int]:
    collapsed = 0

    def _replace(match) -> Any:
        nonlocal collapsed
        code = _normalize_code_text(match.group("code"))
        digest = _code_digest(code)
        if not digest:
            return match.group(0)
        if digest in seen_code:
            collapsed += 1
            return f"```{match.group('language')}\n# duplicate code block omitted by Byte context compiler\n```"
        seen_code.add(digest)
        return match.group(0)

    return _CODE_BLOCK_PATTERN.sub(_replace, text or ""), collapsed


def _summarize_messages(messages: list[dict[str, Any]]) -> str:
    previews = []
    for message in messages[-4:]:
        role = str(message.get("role", "user"))
        raw_content = message.get("content", "") or ""
        content = (
            content_signature(raw_content) if isinstance(raw_content, list) else str(raw_content)
        )
        snippet = " ".join(content.split()[:20]).strip()
        if len(content.split()) > 20:
            snippet = f"{snippet} ..."
        previews.append(f"{role}: {snippet}")
    return "Previous conversation summary:\n" + "\n".join(previews)
