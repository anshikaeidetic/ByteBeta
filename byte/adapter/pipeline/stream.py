import asyncio
from typing import Any

from .memory import (
    _record_ai_memory,
    _record_execution_memory,
    _record_reasoning_memory,
    _record_workflow_outcome,
)


def _stream_chunk_text(chunk) -> str:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            first_choice = choices[0] or {}
            delta = first_choice.get("delta", {}) or {}
            if delta.get("content") not in (None, ""):
                return str(delta.get("content") or "")
            if first_choice.get("text") not in (None, ""):
                return str(first_choice.get("text") or "")
            message = first_choice.get("message", {}) or {}
            if message.get("content") not in (None, ""):
                return str(message.get("content") or "")
        if chunk.get("text") not in (None, ""):
            return str(chunk.get("text") or "")
        if chunk.get("content") not in (None, ""):
            return str(chunk.get("content") or "")
    return ""


def _finalize_stream_memory(chat_cache, request_kwargs, context, *, answer_text, embedding_data) -> None:
    if answer_text in (None, ""):
        _record_workflow_outcome(
            chat_cache, request_kwargs, context, success=False, reason="empty_stream"
        )
        return
    synthetic_response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": answer_text},
            }
        ],
        "model": str(request_kwargs.get("model", "") or ""),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    _record_ai_memory(
        chat_cache,
        request_kwargs,
        context=context,
        answer=answer_text,
        embedding_data=embedding_data,
        llm_data=synthetic_response,
        source="llm",
    )
    _record_execution_memory(
        chat_cache,
        request_kwargs,
        context,
        answer=answer_text,
        llm_data=synthetic_response,
    )
    _record_reasoning_memory(
        chat_cache,
        request_kwargs,
        answer=answer_text,
        verified=False,
        source="stream",
    )
    _record_workflow_outcome(
        chat_cache, request_kwargs, context, success=True, reason="stream_completed"
    )


def _wrap_sync_stream_with_memory(chat_cache, request_kwargs, context, llm_data, *, embedding_data) -> Any:
    def iterator() -> Any:
        parts = []
        try:
            for item in llm_data:
                text = _stream_chunk_text(item)
                if text:
                    parts.append(text)
                yield item
        finally:
            try:
                _finalize_stream_memory(
                    chat_cache,
                    request_kwargs,
                    context,
                    answer_text="".join(parts),
                    embedding_data=embedding_data,
                )
            except Exception:  # pylint: disable=W0703
                pass

    return iterator()


def _wrap_async_stream_with_memory(
    chat_cache,
    request_kwargs,
    context,
    llm_data,
    *,
    embedding_data,
    pending_cache_tasks=None,
) -> Any:
    async def iterator() -> Any:
        parts = []
        try:
            async for item in llm_data:
                text = _stream_chunk_text(item)
                if text:
                    parts.append(text)
                yield item
        finally:
            try:
                _finalize_stream_memory(
                    chat_cache,
                    request_kwargs,
                    context,
                    answer_text="".join(parts),
                    embedding_data=embedding_data,
                )
            except Exception:  # pylint: disable=W0703
                pass
            if pending_cache_tasks:
                try:
                    await asyncio.gather(*pending_cache_tasks)
                finally:
                    pending_cache_tasks.clear()

    return iterator()


__all__ = [name for name in globals() if not name.startswith("__")]
