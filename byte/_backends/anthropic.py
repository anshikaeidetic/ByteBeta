import os
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from byte import cache
from byte.adapter.adapter import aadapt, adapt
from byte.adapter.base import BaseCacheLLM
from byte.adapter.client_pool import get_async_client as get_pooled_async_client
from byte.adapter.client_pool import get_sync_client as get_pooled_sync_client
from byte.adapter.prompt_cache_bridge import (
    apply_native_prompt_cache,
    strip_native_prompt_cache_hints,
)
from byte.manager.scalar_data.base import Answer, DataType
from byte.utils.multimodal import extract_content_parts, extract_text_content


def _create_client(**client_kwargs) -> Any:
    try:
        import anthropic  # pylint: disable=C0415
    except ImportError:
        from byte.utils import import_anthropic  # pylint: disable=C0415

        import_anthropic()
        import anthropic  # pylint: disable=C0415

    return anthropic.Anthropic(**client_kwargs)


def _create_async_client(**client_kwargs) -> Any:
    try:
        import anthropic  # pylint: disable=C0415
    except ImportError:
        from byte.utils import import_anthropic  # pylint: disable=C0415

        import_anthropic()
        import anthropic  # pylint: disable=C0415

    return anthropic.AsyncAnthropic(**client_kwargs)


def _get_client(api_key=None, **kwargs) -> Any:
    """Create or reuse an Anthropic client instance."""
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    elif os.getenv("ANTHROPIC_API_KEY"):
        client_kwargs["api_key"] = os.getenv("ANTHROPIC_API_KEY")

    return get_pooled_sync_client("anthropic", _create_client, **client_kwargs)


def _get_async_client(api_key=None, **kwargs) -> Any:
    """Create or reuse an async Anthropic client instance."""
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    elif os.getenv("ANTHROPIC_API_KEY"):
        client_kwargs["api_key"] = os.getenv("ANTHROPIC_API_KEY")

    return get_pooled_async_client("anthropic", _create_async_client, **client_kwargs)


def _response_to_openai_format(response) -> Any:
    """Convert Anthropic response to OpenAI-compatible dict format."""
    if isinstance(response, dict):
        return response

    # Extract text from content blocks
    text_content = ""
    if hasattr(response, "content") and response.content:
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

    return {
        "byte_provider": "anthropic",
        "choices": [
            {
                "message": {"role": "assistant", "content": text_content},
                "finish_reason": response.stop_reason
                if hasattr(response, "stop_reason")
                else "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "id": response.id if hasattr(response, "id") else "",
        "model": response.model if hasattr(response, "model") else "",
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": response.usage.input_tokens if hasattr(response, "usage") else 0,
            "completion_tokens": response.usage.output_tokens if hasattr(response, "usage") else 0,
            "total_tokens": (
                (response.usage.input_tokens + response.usage.output_tokens)
                if hasattr(response, "usage")
                else 0
            ),
        },
    }


def _anthropic_media_source(part: dict) -> dict:
    if part.get("bytes") is not None:
        import base64  # pylint: disable=C0415

        mime_type = str(part.get("mime_type") or "application/octet-stream")
        if mime_type == "text/plain":
            return {
                "type": "text",
                "media_type": "text/plain",
                "data": part["bytes"].decode("utf-8", errors="replace"),
            }
        return {
            "type": "base64",
            "media_type": mime_type,
            "data": base64.b64encode(part["bytes"]).decode("ascii"),
        }

    uri = str(part.get("uri") or "")
    if uri.startswith(("http://", "https://")):
        return {"type": "url", "url": uri}

    raise ValueError("Anthropic content parts must provide inline bytes or an HTTP(S) URL.")


def _convert_content(content: Any) -> Any:
    if isinstance(content, str):
        return content

    blocks = []
    for part in extract_content_parts(content):
        part_type = part.get("type")
        if part_type == "text":
            text = str(part.get("text") or "")
            if text:
                blocks.append({"type": "text", "text": text})
            continue

        if part_type == "image":
            blocks.append({"type": "image", "source": _anthropic_media_source(part)})
            continue

        if part_type == "file":
            blocks.append(
                {
                    "type": "document",
                    "source": _anthropic_media_source(part),
                    "title": str(part.get("name") or "document"),
                }
            )
            continue

        raise ValueError(f"Anthropic adapter does not support `{part_type}` input parts.")

    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return blocks[0]["text"]
    return blocks


def _convert_messages(messages) -> tuple[Any, ...]:
    anthropic_messages = []
    system_blocks: list[dict] = []

    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            system_text = extract_text_content(msg.get("content"))
            if system_text:
                system_blocks.append({"type": "text", "text": system_text})
            continue

        anthropic_role = "assistant" if role == "assistant" else "user"
        anthropic_messages.append(
            {
                "role": anthropic_role,
                "content": _convert_content(msg.get("content", "")),
            }
        )

    return anthropic_messages, system_blocks


def _apply_prompt_cache(system_blocks, anthropic_messages) -> None:
    if system_blocks:
        system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return
    if not anthropic_messages:
        return
    target_index = 0 if len(anthropic_messages) == 1 else max(0, len(anthropic_messages) - 2)
    content = anthropic_messages[target_index].get("content")
    if isinstance(content, str):
        anthropic_messages[target_index]["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
    elif isinstance(content, list) and content:
        content[-1]["cache_control"] = {"type": "ephemeral"}


def _stream_chunk_to_dict(chunk, model="") -> dict[str, Any] | None:
    """Convert an Anthropic stream event to OpenAI-compatible chunk dict."""
    delta = {}
    finish_reason = None

    event_type = getattr(chunk, "type", "")
    if event_type == "content_block_delta":
        delta = {"content": chunk.delta.text if hasattr(chunk.delta, "text") else ""}
    elif event_type == "message_start":
        delta = {"role": "assistant"}
        if hasattr(chunk, "message") and hasattr(chunk.message, "model"):
            model = chunk.message.model
    elif event_type == "message_delta":
        finish_reason = (
            getattr(chunk.delta, "stop_reason", "stop") if hasattr(chunk, "delta") else "stop"
        )
    elif event_type == "message_stop":
        finish_reason = "stop"
    else:
        return None

    return {
        "choices": [
            {
                "delta": delta,
                "finish_reason": finish_reason,
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "model": model,
        "object": "chat.completion.chunk",
    }


class ChatCompletion(BaseCacheLLM):
    """Anthropic Claude ChatCompletion Wrapper.

    Provides caching for Anthropic Claude models (Claude 3.5, Claude 4, etc.)
    with an OpenAI-compatible response format.

    Example:
        .. code-block:: python

            from byte import cache
            cache.init()

            from byte.adapter import anthropic as cache_anthropic
            response = cache_anthropic.ChatCompletion.create(
                model="claude-sonnet-4-20250514",
                messages=[
                    {"role": "user", "content": "What's GitHub?"}
                ],
                max_tokens=1024,
            )
            print(response['choices'][0]['message']['content'])
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        try:
            native_key = llm_kwargs.pop("native_prompt_cache_key", "")
            llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
            if cls.llm is not None:
                return cls.llm(*llm_args, **llm_kwargs)

            api_key = llm_kwargs.pop("api_key", None)
            client = _get_client(api_key=api_key)

            stream = llm_kwargs.pop("stream", False)
            messages = llm_kwargs.pop("messages", [])
            converted_messages, system_blocks = _convert_messages(messages)
            if native_key:
                _apply_prompt_cache(system_blocks, converted_messages)
            llm_kwargs["messages"] = converted_messages
            if system_blocks:
                llm_kwargs["system"] = system_blocks
            if stream:
                response = client.messages.create(stream=True, **llm_kwargs)
                return _sync_stream_generator(response, llm_kwargs.get("model", ""))
            else:
                response = client.messages.create(**llm_kwargs)
                return _response_to_openai_format(response)
        except Exception as e:
            from byte.utils.error import wrap_error  # pylint: disable=C0415

            raise wrap_error(e) from e

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        try:
            native_key = llm_kwargs.pop("native_prompt_cache_key", "")
            llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
            if cls.llm is not None:
                return await cls.llm(*llm_args, **llm_kwargs)

            api_key = llm_kwargs.pop("api_key", None)
            client = _get_async_client(api_key=api_key)

            stream = llm_kwargs.pop("stream", False)
            messages = llm_kwargs.pop("messages", [])
            converted_messages, system_blocks = _convert_messages(messages)
            if native_key:
                _apply_prompt_cache(system_blocks, converted_messages)
            llm_kwargs["messages"] = converted_messages
            if system_blocks:
                llm_kwargs["system"] = system_blocks
            if stream:
                response = await client.messages.create(stream=True, **llm_kwargs)
                return _async_stream_generator(response, llm_kwargs.get("model", ""))
            else:
                response = await client.messages.create(**llm_kwargs)
                return _response_to_openai_format(response)
        except Exception as e:
            from byte.utils.error import wrap_error  # pylint: disable=C0415

            raise wrap_error(e) from e

    @staticmethod
    def _cache_data_convert(cache_data) -> Any:
        return _construct_resp_from_cache(cache_data)

    @staticmethod
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:
        if isinstance(llm_data, AsyncIterator):

            async def hook_data(it) -> Any:
                total_answer = ""
                async for item in it:
                    content = item.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    total_answer += content
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_data(llm_data)
        if isinstance(llm_data, Iterator):

            def hook_data(it) -> Any:
                total_answer = ""
                for item in it:
                    content = item.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    total_answer += content
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_data(llm_data)
        else:
            content = llm_data["choices"][0]["message"]["content"]
            update_cache_func(Answer(content, DataType.STR))
            return llm_data

    @classmethod
    def create(cls, *args, **kwargs) -> Any:
        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache(
            "anthropic", kwargs, kwargs.get("cache_obj", cache).config
        )

        def cache_data_convert(cache_data) -> Any:
            if kwargs.get("stream", False):
                return _construct_stream_resp_from_cache(cache_data)
            return cls._cache_data_convert(cache_data)

        return adapt(
            cls._llm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, **kwargs) -> Any:
        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache(
            "anthropic", kwargs, kwargs.get("cache_obj", cache).config
        )

        def cache_data_convert(cache_data) -> Any:
            if kwargs.get("stream", False):
                return _async_iter(_construct_stream_resp_from_cache(cache_data))
            return cls._cache_data_convert(cache_data)

        return await aadapt(
            cls._allm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )


def _sync_stream_generator(stream, model="") -> Any:
    """Convert Anthropic stream events to OpenAI-compatible chunk dicts."""
    with stream as s:
        for event in s:
            chunk = _stream_chunk_to_dict(event, model)
            if chunk is not None:
                yield chunk


async def _async_stream_generator(stream, model="") -> Any:
    """Convert Anthropic async stream events to OpenAI-compatible chunk dicts."""
    async with stream as s:
        async for event in s:
            chunk = _stream_chunk_to_dict(event, model)
            if chunk is not None:
                yield chunk


def _construct_resp_from_cache(return_message) -> dict[str, Any]:
    return {
        "byte": True,
        "byte_provider": "anthropic",
        "choices": [
            {
                "message": {"role": "assistant", "content": return_message},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "chat.completion",
    }


def _construct_stream_resp_from_cache(return_message) -> list[Any]:
    created = int(time.time())
    return [
        {
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}],
            "created": created,
            "object": "chat.completion.chunk",
        },
        {
            "choices": [
                {
                    "delta": {"content": return_message},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": created,
            "object": "chat.completion.chunk",
        },
        {
            "byte": True,
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": created,
            "object": "chat.completion.chunk",
        },
    ]


async def _async_iter(items) -> Any:
    for item in items:
        yield item
