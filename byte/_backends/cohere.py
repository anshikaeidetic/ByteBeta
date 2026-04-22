import asyncio
import os
import time
from collections.abc import AsyncGenerator, Iterator
from typing import Any

from byte import cache
from byte._backends.http_transport import (
    async_wrap_sync_iterator,
    iter_json_sse_events,
    request_json,
)
from byte.adapter.adapter import aadapt, adapt
from byte.adapter.base import BaseCacheLLM
from byte.adapter.prompt_cache_bridge import (
    apply_native_prompt_cache,
    strip_native_prompt_cache_hints,
)
from byte.manager.scalar_data.base import Answer, DataType
from byte.processor.stream_cache import replay_as_stream

_API_URL = "https://api.cohere.com/v2/chat"


def _require_api_key(api_key=None) -> Any:
    resolved = api_key or os.getenv("COHERE_API_KEY")
    if not resolved:
        raise ValueError("Missing Cohere credentials. Provide api_key or set COHERE_API_KEY.")
    return resolved


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _normalize_messages(messages) -> Any:
    normalized = []
    for message in messages or []:
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text", "") or ""))
                else:
                    text_parts.append(str(part or ""))
            content = "".join(text_parts)
        normalized.append(
            {
                "role": str(message.get("role", "user") or "user"),
                "content": str(content or ""),
            }
        )
    return normalized


def _extract_text(payload: dict[str, Any]) -> str:
    message = payload.get("message", {}) or {}
    content = message.get("content", []) or []
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "") or item.get("content", "") or ""))
        joined = "".join(parts).strip()
        if joined:
            return joined
    if isinstance(message.get("text"), str):
        return message.get("text")
    if isinstance(payload.get("text"), str):
        return payload.get("text")
    return ""


def _usage_payload(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage", {}) or {}
    tokens = usage.get("tokens", {}) or {}
    billed = usage.get("billed_units", {}) or {}
    prompt_tokens = tokens.get("input_tokens", billed.get("input_tokens", 0))
    completion_tokens = tokens.get("output_tokens", billed.get("output_tokens", 0))
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int((prompt_tokens or 0) + (completion_tokens or 0)),
    }


def _response_to_openai(payload: dict[str, Any], requested_model: str) -> dict[str, Any]:
    text = _extract_text(payload)
    return {
        "byte_provider": "cohere",
        "id": payload.get("id", ""),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", requested_model) or requested_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": payload.get("finish_reason", "stop"),
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": _usage_payload(payload),
    }


def _stream_chunk_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices", []) or []
    if not choices:
        return ""
    delta = dict(choices[0].get("delta", {}) or {})
    return str(delta.get("content", "") or "")


def _cohere_stream_chunk(event_name: str, payload: dict[str, Any], requested_model: str) -> dict[str, Any] | None:
    normalized_event = str(
        event_name or payload.get("type") or payload.get("event_type") or ""
    ).strip().lower()
    finish_reason = payload.get("finish_reason") or payload.get("stop_reason")
    text = ""

    if normalized_event in {"message-start", "message_start"}:
        return {
            "byte_provider": "cohere",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

    if normalized_event in {"content-delta", "content_delta", "text-generation", "text_generation"}:
        delta = payload.get("delta", {}) or {}
        if isinstance(delta, dict):
            message_delta = delta.get("message", {}) or {}
            content = message_delta.get("content", {}) or {}
            text = (
                str(content.get("text", "") or "")
                or str(delta.get("text", "") or "")
                or str(payload.get("text", "") or "")
            )
        if not text:
            text = str(payload.get("text", "") or "")
        if not text:
            return None
        return {
            "byte_provider": "cohere",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }

    if normalized_event in {"message-end", "message_end", "stream-end", "stream_end"}:
        return {
            "byte_provider": "cohere",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": str(finish_reason or "stop")}
            ],
        }
    return None


def _stream_chunks(response, requested_model: str) -> Iterator[dict[str, Any]]:
    for event_name, payload in iter_json_sse_events(response):
        chunk = _cohere_stream_chunk(event_name, payload, requested_model)
        if chunk is not None:
            yield chunk


class ChatCompletion(BaseCacheLLM):
    @classmethod
    def _request(cls, *, model: str, timeout: float = 120.0, **llm_kwargs) -> Any:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return cls.llm(model=model, **llm_kwargs)
        stream = bool(llm_kwargs.pop("stream", False))
        api_key = _require_api_key(llm_kwargs.pop("api_key", None))
        request_timeout = float(llm_kwargs.pop("timeout", timeout) or timeout)
        max_retries = int(llm_kwargs.pop("max_retries", 2) or 0)
        retry_backoff_s = float(llm_kwargs.pop("retry_backoff_s", 0.5) or 0.5)
        messages = _normalize_messages(llm_kwargs.pop("messages", []))
        payload = {"model": model, "messages": messages, "stream": stream}
        payload.update(llm_kwargs)
        response = request_json(
            provider="cohere",
            url=_API_URL,
            headers=_headers(api_key),
            payload=payload,
            timeout=request_timeout,
            stream=stream,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        if stream:
            return _stream_chunks(response, model)
        return _response_to_openai(response.json(), model)

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        model = llm_kwargs.pop("model", "")
        return cls._request(model=model, **llm_kwargs)

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        model = llm_kwargs.pop("model", "")
        if llm_kwargs.get("stream", False):
            stream = cls._request(model=model, **llm_kwargs)
            if hasattr(stream, "__aiter__"):
                return stream
            return async_wrap_sync_iterator(stream)
        return await asyncio.to_thread(cls._request, model=model, **llm_kwargs)

    @staticmethod
    def _cache_data_convert(cache_data) -> dict[str, Any]:
        return {
            "byte": True,
            "byte_provider": "cohere",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": str(cache_data)},
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @staticmethod
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:
        if isinstance(llm_data, AsyncGenerator):

            async def hook(it) -> Any:
                total_answer = ""
                async for item in it:
                    total_answer += _stream_chunk_text(item)
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook(llm_data)
        if isinstance(llm_data, Iterator):

            def hook(it) -> Any:
                total_answer = ""
                for item in it:
                    total_answer += _stream_chunk_text(item)
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook(llm_data)
        choices = llm_data.get("choices", []) or []
        if choices:
            update_cache_func(Answer(choices[0]["message"]["content"], DataType.STR))
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs) -> Any:
        def cache_data_convert(cache_data) -> Any:
            if kwargs.get("stream", False):
                return replay_as_stream(
                    str(cache_data), model=str(kwargs.get("model", "") or "command-r-plus")
                )
            return cls._cache_data_convert(cache_data)

        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache("cohere", kwargs, kwargs.get("cache_obj", cache).config)
        return adapt(
            cls._llm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, **kwargs) -> Any:
        def cache_data_convert(cache_data) -> Any:
            if kwargs.get("stream", False):

                async def _aiter() -> Any:
                    for chunk in replay_as_stream(
                        str(cache_data), model=str(kwargs.get("model", "") or "command-r-plus")
                    ):
                        yield chunk

                return _aiter()
            return cls._cache_data_convert(cache_data)

        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache("cohere", kwargs, kwargs.get("cache_obj", cache).config)
        return await aadapt(
            cls._allm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )
