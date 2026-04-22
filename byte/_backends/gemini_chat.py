
"""OpenAI-compatible chat wrapper for Google Gemini models."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from byte import cache
from byte.adapter.adapter import aadapt, adapt
from byte.adapter.base import BaseCacheLLM
from byte.adapter.prompt_cache_bridge import (
    apply_native_prompt_cache,
    strip_native_prompt_cache_hints,
)
from byte.manager.scalar_data.base import Answer, DataType

from .gemini_clients import (
    _get_async_client as _default_get_async_client,
)
from .gemini_clients import (
    _get_client as _default_get_client,
)
from .gemini_clients import (
    _resolve_backend_callable,
)
from .gemini_messages import _build_content_config, _convert_messages
from .gemini_responses import (
    _async_iter,
    _async_stream_generator,
    _construct_resp_from_cache,
    _construct_stream_resp_from_cache,
    _response_to_openai_format,
    _sync_stream_generator,
)


class ChatCompletion(BaseCacheLLM):
    """Google Gemini ChatCompletion Wrapper.

    Provides caching for Google Gemini models (Gemini 2.0, 2.5, etc.)
    with an OpenAI-compatible response format.

    Example:
        .. code-block:: python

            import os
            os.environ["GOOGLE_API_KEY"] = "<configure-in-env>"

            from byte import cache
            cache.init()

            from byte.adapter import gemini as cache_gemini
            response = cache_gemini.ChatCompletion.create(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "user", "content": "What's GitHub?"}
                ],
            )
            print(response['choices'][0]['message']['content'])
    """

    @classmethod
    def _convert_messages(cls, messages) -> Any:
        """Convert OpenAI-style messages to Gemini-style contents."""
        return _convert_messages(messages)

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        try:
            llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
            if cls.llm is not None:
                return cls.llm(*llm_args, **llm_kwargs)

            api_key = llm_kwargs.pop("api_key", None)
            model = llm_kwargs.pop("model", "gemini-2.0-flash")
            messages = llm_kwargs.pop("messages", [])
            stream = llm_kwargs.pop("stream", False)

            contents, system_instruction = cls._convert_messages(messages)
            config = _build_content_config(system_instruction=system_instruction, **llm_kwargs)
            get_client = _resolve_backend_callable("_get_client", _default_get_client)
            client = get_client(api_key=api_key)

            if stream:
                response = client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )
                return _sync_stream_generator(response, model)
            else:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
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

            api_key = llm_kwargs.pop("api_key", None)
            model = llm_kwargs.pop("model", "gemini-2.0-flash")
            messages = llm_kwargs.pop("messages", [])
            stream = llm_kwargs.pop("stream", False)

            contents, system_instruction = cls._convert_messages(messages)
            config = _build_content_config(system_instruction=system_instruction, **llm_kwargs)
            get_async_client = _resolve_backend_callable(
                "_get_async_client", _default_get_async_client
            )
            client = get_async_client(api_key=api_key)

            if stream:
                response = await client.aio.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )
                return _async_stream_generator(response, model)
            else:
                response = await client.aio.models.generate_content(
                    model=model, contents=contents, config=config
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
        kwargs = apply_native_prompt_cache("gemini", kwargs, kwargs.get("cache_obj", cache).config)

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
        kwargs = apply_native_prompt_cache("gemini", kwargs, kwargs.get("cache_obj", cache).config)

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

__all__ = ["ChatCompletion"]
