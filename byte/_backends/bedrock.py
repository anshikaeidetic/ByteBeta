import asyncio
import os
import time
from collections.abc import AsyncGenerator, Iterator
from typing import Any

from byte import cache
from byte._backends.http_transport import async_wrap_sync_iterator
from byte.adapter.adapter import aadapt, adapt
from byte.adapter.base import BaseCacheLLM
from byte.adapter.prompt_cache_bridge import (
    apply_native_prompt_cache,
    strip_native_prompt_cache_hints,
)
from byte.manager.scalar_data.base import Answer, DataType
from byte.processor.stream_cache import replay_as_stream
from byte.utils import import_boto3


def _normalize_messages(messages) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bedrock_messages: list[dict[str, Any]] = []
    system_blocks: list[dict[str, Any]] = []
    for message in messages or []:
        role = str(message.get("role", "user") or "user")
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "") or ""))
                elif not isinstance(part, dict):
                    text_parts.append(str(part or ""))
            content = "".join(text_parts)
        text_block = {"text": str(content or "")}
        if role == "system":
            system_blocks.append(text_block)
            continue
        bedrock_messages.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": [text_block],
            }
        )
    return bedrock_messages, system_blocks


def _usage_payload(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage", {}) or {}
    prompt_tokens = int(usage.get("inputTokens", 0) or 0)
    completion_tokens = int(usage.get("outputTokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _response_to_openai(payload: dict[str, Any], requested_model: str) -> dict[str, Any]:
    output = payload.get("output", {}) or {}
    message = output.get("message", {}) or {}
    content = message.get("content", []) or []
    text = ""
    for item in content:
        if isinstance(item, dict) and item.get("text") is not None:
            text += str(item.get("text") or "")
    stop_reason = payload.get("stopReason", "stop")
    return {
        "byte_provider": "bedrock",
        "id": payload.get("ResponseMetadata", {}).get("RequestId", ""),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": stop_reason,
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


def _client_kwargs(**llm_kwargs) -> Any:
    region_name = (
        llm_kwargs.pop("region_name", None)
        or llm_kwargs.pop("aws_region", None)
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    params = {"region_name": region_name}
    for field in (
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "profile_name",
    ):
        value = llm_kwargs.pop(field, None)
        if value not in (None, ""):
            params[field] = value
    return params


def _get_client(**llm_kwargs) -> Any:
    import_boto3()
    import boto3  # pylint: disable=C0415

    params = _client_kwargs(**llm_kwargs)
    profile_name = params.pop("profile_name", None)
    if profile_name:
        session = boto3.Session(profile_name=profile_name, region_name=params.get("region_name"))
        return session.client("bedrock-runtime", **params)
    return boto3.client("bedrock-runtime", **params)


def _stream_response_chunks(response: dict[str, Any], requested_model: str) -> Iterator[dict[str, Any]]:
    for event in response.get("stream", []) or []:
        if not isinstance(event, dict):
            continue
        if "messageStart" in event:
            yield {
                "byte_provider": "bedrock",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": requested_model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            continue
        if "contentBlockDelta" in event:
            delta_payload = event.get("contentBlockDelta", {}) or {}
            delta = delta_payload.get("delta", {}) or {}
            text = str(delta.get("text", "") or "")
            if not text:
                continue
            yield {
                "byte_provider": "bedrock",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": requested_model,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            continue
        if "messageStop" in event:
            stop = event.get("messageStop", {}) or {}
            yield {
                "byte_provider": "bedrock",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": str(stop.get("stopReason", "stop") or "stop"),
                    }
                ],
            }


class ChatCompletion(BaseCacheLLM):
    @classmethod
    def _request(cls, *, model: str, **llm_kwargs) -> Any:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return cls.llm(model=model, **llm_kwargs)
        stream = bool(llm_kwargs.pop("stream", False))
        client = _get_client(**llm_kwargs)
        messages, system_blocks = _normalize_messages(llm_kwargs.pop("messages", []))
        inference_config = {}
        for source_key, target_key in (
            ("max_tokens", "maxTokens"),
            ("temperature", "temperature"),
            ("top_p", "topP"),
            ("stop_sequences", "stopSequences"),
        ):
            if source_key in llm_kwargs and llm_kwargs[source_key] not in (None, "", [], {}):
                inference_config[target_key] = llm_kwargs.pop(source_key)
        payload = {
            "modelId": model,
            "messages": messages,
        }
        if system_blocks:
            payload["system"] = system_blocks
        if inference_config:
            payload["inferenceConfig"] = inference_config
        if stream:
            response = client.converse_stream(**payload)
            return _stream_response_chunks(response, model)
        response = client.converse(**payload)
        return _response_to_openai(response, model)

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
            "byte_provider": "bedrock",
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
                    str(cache_data), model=str(kwargs.get("model", "") or "bedrock-chat")
                )
            return cls._cache_data_convert(cache_data)

        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache("bedrock", kwargs, kwargs.get("cache_obj", cache).config)
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
                        str(cache_data), model=str(kwargs.get("model", "") or "bedrock-chat")
                    ):
                        yield chunk

                return _aiter()
            return cls._cache_data_convert(cache_data)

        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache("bedrock", kwargs, kwargs.get("cache_obj", cache).config)
        return await aadapt(
            cls._allm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )
