"""ByteAI Cache adapter for Groq-hosted models (Llama-3, Mixtral, Gemma, etc.)

Groq uses an OpenAI-compatible REST API, so this adapter wraps `groq.Groq`
and passes calls through the standard ByteAI Cache pipeline.

Usage::

    from byte import cache
    from byte.adapter import groq as cache_groq

    cache.init()

    response = cache_groq.ChatCompletion.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Explain transformers in one sentence"}],
    )
    print(response["choices"][0]["message"]["content"])

Multi-agent / per-session isolation::

    from byte import Cache
    from byte.adapter import groq as cache_groq

    agent_cache = Cache()
    agent_cache.init()

    response = cache_groq.ChatCompletion.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": "What is RAG?"}],
        cache_obj=agent_cache,       # isolated cache for this agent
    )

Set GROQ_API_KEY in your environment or pass ``api_key`` directly.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncGenerator, Generator
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
from byte.utils.multimodal import materialize_upload, open_upload
from byte.utils.response import (
    get_message_from_openai_answer,
    get_stream_message_from_openai_answer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_client(**client_kwargs) -> Any:
    groq_cls = load_optional_attr("groq", "Groq", package="groq")
    return groq_cls(**client_kwargs)


def _create_async_client(**client_kwargs) -> Any:
    async_groq_cls = load_optional_attr("groq", "AsyncGroq", package="groq")
    return async_groq_cls(**client_kwargs)


def _get_client(api_key: str | None = None, **kwargs) -> Any:
    return get_pooled_sync_client("groq", _create_client, api_key=api_key, **kwargs)


def _get_async_client(api_key: str | None = None, **kwargs) -> Any:
    return get_pooled_async_client("groq", _create_async_client, api_key=api_key, **kwargs)


def _build_openai_compat(response, model: str) -> dict[str, Any]:
    """Convert a Groq ChatCompletion object to an OpenAI-compatible dict."""
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
        # tool / function calls
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

    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "byte_provider": "groq",
        "id": response.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": usage,
    }


def _extract_binary_response_bytes(response) -> bytes:
    if isinstance(response, bytes):
        return response
    if hasattr(response, "read"):
        return response.read()
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    if isinstance(response, dict):
        payload = response.get("audio") or response.get("data") or response.get("content")
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            return base64.b64decode(payload)
    raise ValueError("Unsupported Groq binary response type")


def _extract_text_response(response) -> str:
    if isinstance(response, dict):
        return str(response.get("text", "") or "")
    return str(getattr(response, "text", "") or "")


def _extract_chunk_usage(chunk) -> dict | None:
    """Extract usage from a Groq streaming chunk (present in final chunk)."""
    # chunk.usage set when stream_options={"include_usage":true}, or in chunk.x_groq.usage always
    raw = getattr(chunk, "usage", None)
    if raw is None:
        x_groq = getattr(chunk, "x_groq", None)
        if x_groq:
            raw = getattr(x_groq, "usage", None)
    if raw is None:
        return None
    return {
        "prompt_tokens": getattr(raw, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(raw, "completion_tokens", 0) or 0,
        "total_tokens": getattr(raw, "total_tokens", 0) or 0,
    }


def _iter_stream_chunks(groq_stream, model: str) -> Generator:
    """Yield OpenAI-compatible chunk dicts from a synchronous Groq stream."""
    for chunk in groq_stream:
        choices = []
        for c in chunk.choices or []:
            delta_content = ""
            if c.delta:
                delta_content = getattr(c.delta, "content", None) or ""
            choices.append({
                "index": c.index,
                "delta": {"role": "assistant", "content": delta_content},
                "finish_reason": c.finish_reason,
            })
        out: dict[str, Any] = {
            "id": getattr(chunk, "id", ""),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
        }
        usage = _extract_chunk_usage(chunk)
        if usage:
            out["usage"] = usage
        yield out


async def _aiter_stream_chunks(groq_stream, model: str) -> AsyncGenerator:
    """Yield OpenAI-compatible chunk dicts from an asynchronous Groq stream."""
    async for chunk in groq_stream:
        choices = []
        for c in chunk.choices or []:
            delta_content = ""
            if c.delta:
                delta_content = getattr(c.delta, "content", None) or ""
            choices.append({
                "index": c.index,
                "delta": {"role": "assistant", "content": delta_content},
                "finish_reason": c.finish_reason,
            })
        out: dict[str, Any] = {
            "id": getattr(chunk, "id", ""),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
        }
        usage = _extract_chunk_usage(chunk)
        if usage:
            out["usage"] = usage
        yield out


def _construct_audio_text_from_cache(return_text) -> dict[str, Any]:
    return {
        "byte": True,
        "text": return_text,
    }


def _construct_speech_from_cache(cache_data, response_format) -> dict[str, Any]:
    payload = json.loads(cache_data)
    return {
        "byte": True,
        "byte_provider": "groq",
        "audio": base64.b64decode(payload["audio"]),
        "format": payload.get("format", response_format),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ChatCompletion(BaseCacheLLM):
    """Cached Groq chat completions.

    Drop-in replacement for ``groq.Client().chat.completions.create()``,
    with transparent ByteAI caching on top.
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return cls.llm(*llm_args, **llm_kwargs)

        stream  = llm_kwargs.pop("stream", False)
        model   = llm_kwargs.pop("model", "")
        api_key = llm_kwargs.pop("api_key", None)
        llm_kwargs.pop("stream_options", None)  # Groq SDK does not support stream_options
        client  = _get_client(api_key=api_key)

        if stream:
            # Return a generator of OpenAI-compatible chunk dicts.
            # The Byte pipeline detects IteratorABC and wraps it for memory/caching.
            groq_stream = client.chat.completions.create(model=model, stream=True, **llm_kwargs)
            return _iter_stream_chunks(groq_stream, model)

        response = client.chat.completions.create(model=model, **llm_kwargs)
        return _build_openai_compat(response, model)

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        llm_kwargs = strip_native_prompt_cache_hints(llm_kwargs)
        if cls.llm is not None:
            return await cls.llm(*llm_args, **llm_kwargs)

        stream  = llm_kwargs.pop("stream", False)
        model   = llm_kwargs.pop("model", "")
        api_key = llm_kwargs.pop("api_key", None)
        llm_kwargs.pop("stream_options", None)  # Groq SDK does not support stream_options
        client  = _get_async_client(api_key=api_key)

        if stream:
            groq_stream = await client.chat.completions.create(model=model, stream=True, **llm_kwargs)
            return _aiter_stream_chunks(groq_stream, model)

        response = await client.chat.completions.create(model=model, **llm_kwargs)
        return _build_openai_compat(response, model)

    @staticmethod
    def _cache_data_convert(cache_data) -> dict[str, Any]:
        return {
            "byte": True,
            "byte_provider": "groq",
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
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
        # Streaming (async): materialize chunks as they pass through, cache full answer at end.
        if isinstance(llm_data, AsyncGenerator):

            async def _hook_async(it):
                total = ""
                async for item in it:
                    total += get_stream_message_from_openai_answer(item) or ""
                    yield item
                try:
                    update_cache_func(Answer(total, DataType.STR))
                except Exception as exc:  # pylint: disable=W0703
                    byte_log.warning("groq cache update failed (async stream): %s", exc)

            return _hook_async(llm_data)

        # Streaming (sync): same pattern with a sync generator.
        from collections.abc import (
            Iterator as _IteratorABC,  # local import to avoid top-level dependency change
        )
        if isinstance(llm_data, _IteratorABC):

            def _hook_sync(it):
                total = ""
                for item in it:
                    total += get_stream_message_from_openai_answer(item) or ""
                    yield item
                try:
                    update_cache_func(Answer(total, DataType.STR))
                except Exception as exc:  # pylint: disable=W0703
                    byte_log.warning("groq cache update failed (sync stream): %s", exc)

            return _hook_sync(llm_data)

        # Non-streaming dict response — safe to .get().
        try:
            content = get_message_from_openai_answer(llm_data)
            if content:
                update_cache_func(Answer(content, DataType.STR))
        except Exception as exc:  # pylint: disable=W0703
            byte_log.warning("groq cache update failed: %s", exc)
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs) -> dict[str, Any]:
        """Send a (possibly cached) chat request to Groq.

        All standard Groq parameters (``temperature``, ``max_tokens``,
        ``tools``, ``tool_choice``, etc.) are forwarded as-is when the cache
        misses.

        Extra byte cache parameters:

        - ``cache_obj``: per-agent :class:`~byte.core.Cache` instance
        - ``cache_skip``: ``True`` to bypass cache for this call
        - ``cache_factor``: float multiplier on the similarity threshold
        - ``session``: session id for multi-turn isolation
        """
        kwargs = cls.fill_base_args(**kwargs)
        kwargs = apply_native_prompt_cache("groq", kwargs, kwargs.get("cache_obj", cache).config)
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
        kwargs = apply_native_prompt_cache("groq", kwargs, kwargs.get("cache_obj", cache).config)
        return await aadapt(
            cls._allm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )


class Audio:
    """Groq speech-to-text wrappers with Byte caching."""

    llm = None

    @classmethod
    def transcribe(cls, model: str, file: Any, *args, **kwargs) -> Any:
        upload = materialize_upload(file, default_name="audio.bin")

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            target_model = llm_kwargs.pop("model", model)
            file_payload = llm_kwargs.pop("file", upload)
            if cls.llm is not None:
                return cls.llm(model=target_model, file=file_payload, **llm_kwargs)

            api_key = llm_kwargs.pop("api_key", None)
            client = _get_client(api_key=api_key)
            response = client.audio.transcriptions.create(
                model=target_model,
                file=open_upload(file_payload),
                **llm_kwargs,
            )
            return {
                "byte_provider": "groq",
                "text": _extract_text_response(response),
            }

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            update_cache_func(Answer(llm_data.get("text", ""), DataType.STR))
            return llm_data

        return adapt(
            llm_handler,
            _construct_audio_text_from_cache,
            update_cache_callback,
            model=model,
            *args,
            file=upload,
            **kwargs,
        )

    @classmethod
    def translate(cls, model: str, file: Any, *args, **kwargs) -> Any:
        upload = materialize_upload(file, default_name="audio.bin")

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            target_model = llm_kwargs.pop("model", model)
            file_payload = llm_kwargs.pop("file", upload)
            if cls.llm is not None:
                return cls.llm(model=target_model, file=file_payload, **llm_kwargs)

            api_key = llm_kwargs.pop("api_key", None)
            client = _get_client(api_key=api_key)
            response = client.audio.translations.create(
                model=target_model,
                file=open_upload(file_payload),
                **llm_kwargs,
            )
            return {
                "byte_provider": "groq",
                "text": _extract_text_response(response),
            }

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            update_cache_func(Answer(llm_data.get("text", ""), DataType.STR))
            return llm_data

        return adapt(
            llm_handler,
            _construct_audio_text_from_cache,
            update_cache_callback,
            model=model,
            *args,
            file=upload,
            **kwargs,
        )


class Speech:
    """Groq text-to-speech wrapper with cacheable audio payloads."""

    llm = None

    @classmethod
    def create(cls, model: str, input: str, voice: str, *args, **kwargs) -> Any:
        response_format = kwargs.pop("response_format", "mp3")

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            target_model = llm_kwargs.pop("model", model)
            input_text = llm_kwargs.pop("input", input)
            voice_name = llm_kwargs.pop("voice", voice)
            fmt = llm_kwargs.pop("response_format", response_format)
            if cls.llm is not None:
                return cls.llm(
                    model=target_model,
                    input=input_text,
                    voice=voice_name,
                    response_format=fmt,
                    **llm_kwargs,
                )

            api_key = llm_kwargs.pop("api_key", None)
            client = _get_client(api_key=api_key)
            response = client.audio.speech.create(
                model=target_model,
                input=input_text,
                voice=voice_name,
                response_format=fmt,
                **llm_kwargs,
            )
            return {
                "byte_provider": "groq",
                "audio": _extract_binary_response_bytes(response),
                "format": fmt,
            }

        def cache_data_convert(cache_data) -> Any:
            return _construct_speech_from_cache(cache_data, response_format)

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            update_cache_func(
                Answer(
                    json.dumps(
                        {
                            "audio": base64.b64encode(llm_data["audio"]).decode("ascii"),
                            "format": llm_data.get("format", response_format),
                        }
                    ),
                    DataType.STR,
                )
            )
            return llm_data

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            model=model,
            input=input,
            voice=voice,
            *args,
            response_format=response_format,
            **kwargs,
        )
