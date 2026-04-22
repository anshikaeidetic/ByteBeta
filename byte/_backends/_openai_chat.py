"""Chat and completion surfaces for the Byte OpenAI backend."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from typing import Any

from byte import cache
from byte.adapter.adapter import aadapt, adapt
from byte.adapter.base import BaseCacheLLM
from byte.adapter.prompt_cache_bridge import (
    apply_native_prompt_cache,
    strip_native_prompt_cache_hints,
)
from byte.manager.scalar_data.base import Answer, DataType
from byte.utils.response import (
    get_message_from_openai_answer,
    get_stream_message_from_openai_answer,
    get_text_from_openai_answer,
)
from byte.utils.token import token_counter

from ._openai_response import (
    _construct_resp_from_cache,
    _construct_stream_resp_from_cache,
    _construct_text_from_cache,
    _openai_response_to_dict,
)
from ._openai_support import (
    _async_stream_generator,
    _maybe_wrap_openai_error,
    _num_tokens_from_messages,
    _stream_generator,
    async_iter,
)


def _backend_module() -> Any:
    from byte._backends import openai as backend

    return backend


class ChatCompletion(BaseCacheLLM):
    """OpenAI ChatCompletion Wrapper (compatible with openai v1.x/v2.x)

    Example:
        .. code-block:: python

            from byte import cache
            from byte.processor.pre import get_prompt
            # init byte
            cache.init()
            cache.set_openai_key()

            from byte.adapter import openai
            # run ChatCompletion model with byte
            response = openai.ChatCompletion.create(
                          model='gpt-3.5-turbo',
                          messages=[
                            {
                                'role': 'user',
                                'content': "what's github"
                            }],
                        )
            response_content = response['choices'][0]['message']['content']
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return cls.llm(*llm_args, **llm_kwargs)

        try:
            api_key = llm_kwargs.pop("api_key", None)
            api_base = llm_kwargs.pop("api_base", None)
            client = _backend_module()._get_client(api_key=api_key, api_base=api_base)
            response = client.chat.completions.create(**llm_kwargs)

            if llm_kwargs.get("stream", False):
                return _stream_generator(response)
            return _openai_response_to_dict(response)
        except Exception as exc:
            wrapped = _maybe_wrap_openai_error(exc)
            if wrapped is not None:
                raise wrapped from exc
            raise

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return await cls.llm(*llm_args, **llm_kwargs)

        try:
            api_key = llm_kwargs.pop("api_key", None)
            api_base = llm_kwargs.pop("api_base", None)
            client = _backend_module()._get_async_client(api_key=api_key, api_base=api_base)
            response = await client.chat.completions.create(**llm_kwargs)

            if llm_kwargs.get("stream", False):
                return _async_stream_generator(response)
            return _openai_response_to_dict(response)
        except Exception as exc:
            wrapped = _maybe_wrap_openai_error(exc)
            if wrapped is not None:
                raise wrapped from exc
            raise

    @staticmethod
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
        if isinstance(llm_data, AsyncGenerator):

            async def hook_openai_data(it) -> Any:
                total_answer = ""
                async for item in it:
                    chunk_text = get_stream_message_from_openai_answer(item) or ""
                    total_answer += chunk_text
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_openai_data(llm_data)
        elif not isinstance(llm_data, Iterator):
            update_cache_func(Answer(get_message_from_openai_answer(llm_data), DataType.STR))
            return llm_data
        else:

            def hook_openai_data(it) -> Any:
                total_answer = ""
                for item in it:
                    chunk_text = get_stream_message_from_openai_answer(item) or ""
                    total_answer += chunk_text
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_openai_data(llm_data)

    @classmethod
    def create(cls, *args, **kwargs) -> Any:
        kwargs = cls.fill_base_args(**kwargs)
        chat_cache = kwargs.get("cache_obj", cache)
        kwargs = apply_native_prompt_cache("openai", kwargs, chat_cache.config)
        enable_token_counter = chat_cache.config.enable_token_counter

        def cache_data_convert(cache_data) -> Any:
            if enable_token_counter:
                input_token = _num_tokens_from_messages(kwargs.get("messages"))
                output_token = token_counter(cache_data)
                saved_token = [input_token, output_token]
            else:
                saved_token = [0, 0]
            if kwargs.get("stream", False):
                return _construct_stream_resp_from_cache(cache_data, saved_token)
            return _construct_resp_from_cache(cache_data, saved_token)

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
        chat_cache = kwargs.get("cache_obj", cache)
        kwargs = apply_native_prompt_cache("openai", kwargs, chat_cache.config)
        enable_token_counter = chat_cache.config.enable_token_counter

        def cache_data_convert(cache_data) -> Any:
            if enable_token_counter:
                input_token = _num_tokens_from_messages(kwargs.get("messages"))
                output_token = token_counter(cache_data)
                saved_token = [input_token, output_token]
            else:
                saved_token = [0, 0]
            if kwargs.get("stream", False):
                return async_iter(_construct_stream_resp_from_cache(cache_data, saved_token))
            return _construct_resp_from_cache(cache_data, saved_token)

        return await aadapt(
            cls._allm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )


class Completion(BaseCacheLLM):
    """OpenAI Completion Wrapper (compatible with openai v1.x/v2.x)

    Example:
        .. code-block:: python

            from byte import cache
            from byte.processor.pre import get_prompt
            # init byte
            cache.init()
            cache.set_openai_key()

            from byte.adapter import openai
            # run Completion model with byte
            response = openai.Completion.create(model="gpt-3.5-turbo-instruct",
                                                prompt="Hello world.")
            response_text = response["choices"][0]["text"]
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return cls.llm(*llm_args, **llm_kwargs)

        try:
            api_key = llm_kwargs.pop("api_key", None)
            api_base = llm_kwargs.pop("api_base", None)
            client = _backend_module()._get_client(api_key=api_key, api_base=api_base)
            response = client.completions.create(**llm_kwargs)
            return _openai_response_to_dict(response)
        except Exception as exc:
            wrapped = _maybe_wrap_openai_error(exc)
            if wrapped is not None:
                raise wrapped from exc
            raise

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return await cls.llm(*llm_args, **llm_kwargs)

        try:
            api_key = llm_kwargs.pop("api_key", None)
            api_base = llm_kwargs.pop("api_base", None)
            client = _backend_module()._get_async_client(api_key=api_key, api_base=api_base)
            response = await client.completions.create(**llm_kwargs)
            return _openai_response_to_dict(response)
        except Exception as exc:
            wrapped = _maybe_wrap_openai_error(exc)
            if wrapped is not None:
                raise wrapped from exc
            raise

    @staticmethod
    def _cache_data_convert(cache_data) -> Any:
        return _construct_text_from_cache(cache_data)

    @staticmethod
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
        update_cache_func(Answer(get_text_from_openai_answer(llm_data), DataType.STR))
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs) -> Any:
        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache("openai", kwargs, kwargs.get("cache_obj", cache).config)
        return adapt(
            cls._llm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, **kwargs) -> Any:
        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache("openai", kwargs, kwargs.get("cache_obj", cache).config)
        return await aadapt(
            cls._allm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )
