
"""Prompt-piece extraction helpers for optimization memory."""

from __future__ import annotations

from typing import Any


def extract_prompt_pieces(request_kwargs: dict[str, Any] | None) -> list[dict[str, Any]]:
    request_kwargs = request_kwargs or {}
    pieces: list[dict[str, Any]] = []
    messages = request_kwargs.get("messages") or []
    for index, message in enumerate(messages):
        role = str(message.get("role", "user") or "user")
        content = message.get("content", "")
        if role == "system":
            pieces.append({"type": "system", "content": content, "message_index": index})
        elif index < max(len(messages) - 1, 0):
            pieces.append({"type": f"history_{role}", "content": content, "message_index": index})

    for field, piece_type in (
        ("tools", "tools"),
        ("functions", "tool_schemas"),
        ("byte_prompt_pieces", "prompt_pieces"),
        ("byte_retrieval_context", "retrieval_context"),
        ("byte_document_context", "document_context"),
        ("byte_support_articles", "support_articles"),
        ("byte_repo_snapshot", "repo_snapshot"),
        ("byte_repo_summary", "repo_summary"),
        ("byte_changed_files", "changed_files"),
        ("byte_changed_hunks", "changed_hunks"),
        ("byte_tool_result_context", "tool_results"),
    ):
        value = request_kwargs.get(field)
        if value not in (None, "", [], {}):
            pieces.append({"type": piece_type, "content": value, "field": field})

    if request_kwargs.get("prompt") not in (None, ""):
        pieces.append({"type": "prompt", "content": request_kwargs.get("prompt")})
    if request_kwargs.get("input") not in (None, ""):
        pieces.append({"type": "input", "content": request_kwargs.get("input")})
    return pieces

def _extract_request_intent(request_kwargs: dict[str, Any]) -> Any:
    from byte.processor.intent import extract_request_intent  # pylint: disable=C0415

    return extract_request_intent(request_kwargs)

__all__ = ["_extract_request_intent", "extract_prompt_pieces"]
