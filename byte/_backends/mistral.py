import asyncio
import os
import time
from collections.abc import AsyncGenerator, Iterator
from typing import Any

from byte import cache
from byte._backends.http_transport import (
    async_wrap_sync_iterator,
    iter_openai_sse_chunks,
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

_API_URL = "https://api.mistral.ai/v1/chat/completions"


def _require_api_key(api_key=None) -> Any:
    resolved = api_key or os.getenv("MISTRAL_API_KEY")
    if not resolved:
        raise ValueError("Missing Mistral credentials. Provide api_key or set MISTRAL_API_KEY.")
    return resolved


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _response_to_openai(payload: dict[str, Any], requested_model: str) -> dict[str, Any]:
    choices = payload.get("choices", []) or []
    normalized_choices = []
    for index, choice in enumerate(choices):
        message = dict(choice.get("message", {}) or {})
        normalized_choices.append(
            {
                "index": int(choice.get("index", index) or index),
                "finish_reason": choice.get("finish_reason"),
                "message": {
                    "role": str(message.get("role", "assistant") or "assistant"),
                    "content": message.get("content", "") or "",
                },
            }
        )
    usage = payload.get("usage", {}) or {}
    return {
        "byte_provider": "mistral",
        "id": payload.get("id", ""),
        "object": payload.get("object", "chat.completion"),
        "created": int(payload.get("created", int(time.time())) or int(time.time())),
        "model": payload.get("model", requested_model) or requested_model,
        "choices": normalized_choices,
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
    }


def _stream_chunk_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices", []) or []
    if not choices:
        return ""
    delta = dict(choices[0].get("delta", {}) or {})
    return str(delta.get("content", "") or "")


def _stream_chunks(response) -> Iterator[dict[str, Any]]:
    yield from iter_openai_sse_chunks(response, provider="mistral")


class ChatCompletion(BaseCacheLLM):
    @classmethod
    def _request(cls, *, model: str, stream: bool = False, timeout: float = 120.0, **llm_kwargs) -> Any:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return cls.llm(model=model, stream=stream, **llm_kwargs)
        api_key = _require_api_key(llm_kwargs.pop("api_key", None))
        request_timeout = float(llm_kwargs.pop("timeout", timeout) or timeout)
        max_retries = int(llm_kwargs.pop("max_retries", 2) or 0)
        retry_backoff_s = float(llm_kwargs.pop("retry_backoff_s", 0.5) or 0.5)
        payload = {"model": model, "stream": bool(stream)}
        payload.update(llm_kwargs)
        response = request_json(
            provider="mistral",
            url=_API_URL,
            headers=_headers(api_key),
            payload=payload,
            timeout=request_timeout,
            stream=bool(stream),
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        if stream:
            return _stream_chunks(response)
        return _response_to_openai(response.json(), model)

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        model = llm_kwargs.pop("model", "")
        stream = bool(llm_kwargs.pop("stream", False))
        return cls._request(model=model, stream=stream, **llm_kwargs)

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
            "byte_provider": "mistral",
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
                    str(cache_data), model=str(kwargs.get("model", "") or "mistral")
                )
            return cls._cache_data_convert(cache_data)

        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache("mistral", kwargs, kwargs.get("cache_obj", cache).config)
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
                        str(cache_data), model=str(kwargs.get("model", "") or "mistral")
                    ):
                        yield chunk

                return _aiter()
            return cls._cache_data_convert(cache_data)

        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache("mistral", kwargs, kwargs.get("cache_obj", cache).config)
        return await aadapt(
            cls._allm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )
