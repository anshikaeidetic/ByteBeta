"""ByteAI Cache adapter for OpenRouter (aggregated LLM gateway).

OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1,
so this adapter is a thin wrapper that sets the right base_url and routes
through the standard ByteAI Cache pipeline.

Supported models (examples):
  - google/gemini-2.0-flash-001
  - anthropic/claude-3-5-sonnet
  - meta-llama/llama-3.3-70b-instruct
  - mistralai/mixtral-8x7b-instruct
  - deepseek/deepseek-chat
  - ... 300+ models via single key

Usage::

    from byte import cache
    from byte.adapter import openrouter as cache_openrouter

    cache.init()

    response = cache_openrouter.ChatCompletion.create(
        model="google/gemini-2.0-flash-001",
        messages=[{"role": "user", "content": "What is ByteAI Cache?"}],
        api_key="<openrouter-api-key>",  # or set OPENROUTER_API_KEY in the environment
    )
    print(response["choices"][0]["message"]["content"])

Set OPENROUTER_API_KEY in your environment, or pass ``api_key`` directly.
"""

from __future__ import annotations

import base64
import json
import os
import time
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
from byte.utils import load_optional_attr
from byte.utils.log import byte_log

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_openai_clients() -> tuple[Any, ...]:
    openai_cls = load_optional_attr("openai", "OpenAI", package="openai")
    async_openai_cls = load_optional_attr("openai", "AsyncOpenAI", package="openai")
    return openai_cls, async_openai_cls


def _create_client(**client_kwargs) -> Any:
    openai_cls, _ = _load_openai_clients()
    return openai_cls(**client_kwargs)


def _create_async_client(**client_kwargs) -> Any:
    _, async_openai_cls = _load_openai_clients()
    return async_openai_cls(**client_kwargs)


def _get_client(api_key: str | None = None, **kwargs) -> Any:
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    return get_pooled_sync_client(
        "openrouter",
        _create_client,
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        **kwargs,
    )


def _get_async_client(api_key: str | None = None, **kwargs) -> Any:
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    return get_pooled_async_client(
        "openrouter",
        _create_async_client,
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        **kwargs,
    )


def _build_openai_compat(response, model: str) -> dict[str, Any]:
    """Convert an openai-SDK ChatCompletion object to a plain dict."""
    choices = []
    for choice in response.choices:
        msg = choice.message
        choice_dict: dict[str, Any] = {
            "index": choice.index,
            "finish_reason": choice.finish_reason,
            "message": {
                "role": msg.role,
                "content": msg.content or "",
            },
        }
        if getattr(msg, "tool_calls", None):
            choice_dict["message"]["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        choices.append(choice_dict)

    usage: dict[str, int] = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "byte_provider": "openrouter",
        "id": getattr(response, "id", ""),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": usage,
    }


def _response_to_dict(response) -> Any:
    if isinstance(response, dict):
        return response
    try:
        return response.model_dump()
    except AttributeError:
        return response


def _extract_image_payloads(response_dict) -> Any:
    images = []
    for item in response_dict.get("data", []):
        if item.get("b64_json"):
            images.append({"b64": item["b64_json"], "mime_type": "image/png"})
    return images


def _construct_image_from_cache(cache_data, response_format) -> Any:
    payload = json.loads(cache_data)
    result = {"byte": True, "byte_provider": "openrouter", "created": int(time.time()), "data": []}
    for index, item in enumerate(payload.get("images", [])):
        image_bytes = base64.b64decode(item["b64"])
        mime_type = item.get("mime_type") or "image/png"
        if response_format == "b64_json":
            result["data"].append({"b64_json": item["b64"]})
            continue
        extension = ".png"
        if "/" in mime_type:
            extension = f".{mime_type.split('/', 1)[1]}"
        target = os.path.abspath(f"{int(time.time())}_{index}{extension}")
        with open(target, "wb") as file_obj:
            file_obj.write(image_bytes)
        result["data"].append({"url": target})
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ChatCompletion(BaseCacheLLM):
    """Cached OpenRouter chat completions.

    Access 300+ models (GPT-4o, Claude, Gemini, Llama, Mistral, DeepSeek …)
    through a single API key, all transparently cached by ByteAI.

    Extra byte cache parameters on every call:

    - ``cache_obj``: per-agent :class:`~byte.core.Cache` instance
    - ``cache_skip``: ``True`` to bypass cache for this call
    - ``cache_factor``: float multiplier on the similarity threshold
    - ``session``: session id for multi-turn isolation
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> dict[str, Any]:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return cls.llm(*llm_args, **llm_kwargs)

        model = llm_kwargs.pop("model", "")
        api_key = llm_kwargs.pop("api_key", None)
        client = _get_client(api_key=api_key)
        response = client.chat.completions.create(model=model, **llm_kwargs)
        return _build_openai_compat(response, model)

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return await cls.llm(*llm_args, **llm_kwargs)

        model = llm_kwargs.pop("model", "")
        api_key = llm_kwargs.pop("api_key", None)
        client = _get_async_client(api_key=api_key)
        response = await client.chat.completions.create(model=model, **llm_kwargs)
        return _build_openai_compat(response, model)

    @staticmethod
    def _cache_data_convert(cache_data) -> dict[str, Any]:
        return {
            "byte": True,
            "byte_provider": "openrouter",
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
        try:
            choices = llm_data.get("choices", [])
            if choices:
                msg = choices[0]["message"]
                # Support caching tool calls as JSON string
                if "tool_calls" in msg:
                    import json

                    answer = json.dumps(msg["tool_calls"])
                else:
                    answer = msg.get("content") or ""
                update_cache_func(answer)
        except Exception as exc:  # pylint: disable=W0703
            byte_log.warning("openrouter cache update failed: %s", exc)
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs) -> dict[str, Any]:
        """Send a (possibly cached) chat request through OpenRouter.

        Example::

            response = ChatCompletion.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[{"role": "user", "content": "hi"}],
                api_key="<openrouter-api-key>",
            )
        """
        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache(
            "openrouter", kwargs, kwargs.get("cache_obj", cache).config
        )
        return adapt(
            cls._llm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, **kwargs) -> dict[str, Any]:
        """Async version of :meth:`create`."""
        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache(
            "openrouter", kwargs, kwargs.get("cache_obj", cache).config
        )
        return await aadapt(
            cls._allm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )


class Image:
    """Cached OpenRouter image generation wrapper."""

    llm = None

    @classmethod
    def create(cls, *args, **kwargs) -> dict[str, Any]:
        response_format = kwargs.pop("response_format", "b64_json")

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            if cls.llm is not None:
                return cls.llm(*llm_args, **llm_kwargs)

            api_key = llm_kwargs.pop("api_key", None)
            client = _get_client(api_key=api_key)
            response = client.images.generate(response_format="b64_json", **llm_kwargs)
            response_dict = _response_to_dict(response)
            if response_format == "b64_json":
                response_dict["byte_provider"] = "openrouter"
                return response_dict

            formatted = {"byte_provider": "openrouter", "created": int(time.time()), "data": []}
            for index, item in enumerate(response_dict.get("data", [])):
                b64_payload = item.get("b64_json")
                if not b64_payload:
                    continue
                image_bytes = base64.b64decode(b64_payload)
                target = os.path.abspath(f"{int(time.time())}_{index}.png")
                with open(target, "wb") as file_obj:
                    file_obj.write(image_bytes)
                formatted["data"].append({"url": target})
            return formatted

        def cache_data_convert(cache_data) -> Any:
            return _construct_image_from_cache(cache_data, response_format)

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            images = []
            for item in llm_data.get("data", []):
                if item.get("b64_json"):
                    images.append({"b64": item["b64_json"], "mime_type": "image/png"})
                    continue
                if item.get("url"):
                    with open(item["url"], "rb") as file_obj:
                        images.append(
                            {
                                "b64": base64.b64encode(file_obj.read()).decode("ascii"),
                                "mime_type": "image/png",
                            }
                        )
            update_cache_func(Answer(json.dumps({"images": images}), DataType.STR))
            return llm_data

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            *args,
            **kwargs,
        )
