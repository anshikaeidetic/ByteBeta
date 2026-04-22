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
from byte.utils.multimodal import extract_content_parts


def _create_client(**client_kwargs) -> Any:
    try:
        import ollama  # pylint: disable=C0415
    except ImportError:
        from byte.utils import import_ollama  # pylint: disable=C0415

        import_ollama()
        import ollama  # pylint: disable=C0415

    return ollama.Client(**client_kwargs)


def _create_async_client(**client_kwargs) -> Any:
    try:
        import ollama  # pylint: disable=C0415
    except ImportError:
        from byte.utils import import_ollama  # pylint: disable=C0415

        import_ollama()
        import ollama  # pylint: disable=C0415

    return ollama.AsyncClient(**client_kwargs)


def _get_client(host=None, **kwargs) -> Any:
    """Create or reuse an Ollama client instance."""
    client_kwargs = {}
    _host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if _host:
        client_kwargs["host"] = _host

    return get_pooled_sync_client("ollama", _create_client, **client_kwargs)


def _get_async_client(host=None, **kwargs) -> Any:
    """Create or reuse an async Ollama client instance."""
    client_kwargs = {}
    _host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if _host:
        client_kwargs["host"] = _host

    return get_pooled_async_client("ollama", _create_async_client, **client_kwargs)


def _response_to_openai_format(response, model="") -> dict[str, Any]:
    """Convert Ollama response to OpenAI-compatible dict format."""
    if isinstance(response, dict):
        # Ollama already returns dicts, just normalize the format
        message = response.get("message", {})
        return {
            "byte_provider": "ollama",
            "choices": [
                {
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                    },
                    "finish_reason": "stop" if response.get("done", True) else None,
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "model": response.get("model", model),
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": response.get("prompt_eval_count", 0) or 0,
                "completion_tokens": response.get("eval_count", 0) or 0,
                "total_tokens": (
                    (response.get("prompt_eval_count", 0) or 0)
                    + (response.get("eval_count", 0) or 0)
                ),
            },
        }

    # Handle object-style response
    return {
        "byte_provider": "ollama",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": getattr(response, "message", {}).get("content", "")
                    if isinstance(getattr(response, "message", {}), dict)
                    else str(response),
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "model": model,
        "object": "chat.completion",
    }


def _convert_messages(messages) -> Any:
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        text_parts = []
        images = []
        for part in extract_content_parts(content):
            part_type = part.get("type")
            if part_type == "text":
                text = str(part.get("text") or "")
                if text:
                    text_parts.append(text)
                continue
            if part_type == "image":
                if part.get("bytes") is not None:
                    images.append(part["bytes"])
                    continue
                if part.get("uri"):
                    images.append(str(part["uri"]))
                    continue
            raise ValueError(f"Ollama adapter does not support `{part_type}` input parts.")

        converted.append(
            {
                "role": role,
                "content": "\n".join(text_parts).strip() or None,
                "images": images or None,
            }
        )

    return converted


class ChatCompletion(BaseCacheLLM):
    """Ollama ChatCompletion Wrapper.

    Provides caching for locally-hosted models via Ollama (Llama, Mistral,
    Phi, Qwen, DeepSeek, etc.) with an OpenAI-compatible response format.

    Example:
        .. code-block:: python

            from byte import cache
            cache.init()

            from byte.adapter import ollama as cache_ollama
            response = cache_ollama.ChatCompletion.create(
                model="llama3.2",
                messages=[
                    {"role": "user", "content": "What's GitHub?"}
                ],
            )
            print(response['choices'][0]['message']['content'])
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        try:
            llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
            if cls.llm is not None:
                return cls.llm(*llm_args, **llm_kwargs)

            host = llm_kwargs.pop("host", None)
            client = _get_client(host=host)

            model = llm_kwargs.pop("model", "llama3.2")
            messages = _convert_messages(llm_kwargs.pop("messages", []))
            stream = llm_kwargs.pop("stream", False)

            # Remove OpenAI-specific kwargs that Ollama doesn't support
            llm_kwargs.pop("api_key", None)
            llm_kwargs.pop("api_base", None)

            # Map OpenAI params to Ollama options
            options = llm_kwargs.pop("options", {})
            if "temperature" in llm_kwargs:
                options["temperature"] = llm_kwargs.pop("temperature")
            if "top_p" in llm_kwargs:
                options["top_p"] = llm_kwargs.pop("top_p")
            if "max_tokens" in llm_kwargs:
                options["num_predict"] = llm_kwargs.pop("max_tokens")

            if stream:
                response = client.chat(
                    model=model, messages=messages, stream=True, options=options or None
                )
                return _sync_stream_generator(response, model)
            else:
                response = client.chat(model=model, messages=messages, options=options or None)
                return _response_to_openai_format(response, model)
        except Exception as e:
            from byte.utils.error import wrap_error  # pylint: disable=C0415

            raise wrap_error(e) from e

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        try:
            llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
            if cls.llm is not None:
                return await cls.llm(*llm_args, **llm_kwargs)

            host = llm_kwargs.pop("host", None)
            client = _get_async_client(host=host)

            model = llm_kwargs.pop("model", "llama3.2")
            messages = _convert_messages(llm_kwargs.pop("messages", []))
            stream = llm_kwargs.pop("stream", False)

            llm_kwargs.pop("api_key", None)
            llm_kwargs.pop("api_base", None)

            options = llm_kwargs.pop("options", {})
            if "temperature" in llm_kwargs:
                options["temperature"] = llm_kwargs.pop("temperature")
            if "top_p" in llm_kwargs:
                options["top_p"] = llm_kwargs.pop("top_p")
            if "max_tokens" in llm_kwargs:
                options["num_predict"] = llm_kwargs.pop("max_tokens")

            if stream:
                response = await client.chat(
                    model=model, messages=messages, stream=True, options=options or None
                )
                return _async_stream_generator(response, model)
            else:
                response = await client.chat(
                    model=model, messages=messages, options=options or None
                )
                return _response_to_openai_format(response, model)
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
        kwargs = apply_native_prompt_cache("ollama", kwargs, kwargs.get("cache_obj", cache).config)

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
        kwargs = apply_native_prompt_cache("ollama", kwargs, kwargs.get("cache_obj", cache).config)

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


def _sync_stream_generator(response, model="") -> Any:
    """Convert Ollama stream chunks to OpenAI-compatible chunk dicts."""
    for chunk in response:
        message = chunk.get("message", {}) if isinstance(chunk, dict) else {}
        content = message.get("content", "")
        done = chunk.get("done", False) if isinstance(chunk, dict) else False

        yield {
            "choices": [
                {
                    "delta": {"content": content},
                    "finish_reason": "stop" if done else None,
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "model": model,
            "object": "chat.completion.chunk",
        }


async def _async_stream_generator(response, model="") -> Any:
    """Convert Ollama async stream chunks to OpenAI-compatible chunk dicts."""
    async for chunk in response:
        message = chunk.get("message", {}) if isinstance(chunk, dict) else {}
        content = message.get("content", "")
        done = chunk.get("done", False) if isinstance(chunk, dict) else False

        yield {
            "choices": [
                {
                    "delta": {"content": content},
                    "finish_reason": "stop" if done else None,
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "model": model,
            "object": "chat.completion.chunk",
        }


def _construct_resp_from_cache(return_message) -> dict[str, Any]:
    return {
        "byte": True,
        "byte_provider": "ollama",
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
