import time
from collections.abc import AsyncGenerator, Iterator
from typing import Any

from byte import cache
from byte.adapter.adapter import aadapt, adapt
from byte.adapter.base import BaseCacheLLM
from byte.h2o.runtime import describe_huggingface_runtime, get_huggingface_runtime
from byte.manager.scalar_data.base import Answer, DataType
from byte.utils.async_ops import run_sync


def _runtime_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "tokenizer_name": kwargs.pop("tokenizer_name", None),
        "revision": kwargs.pop("revision", None),
        "trust_remote_code": bool(kwargs.pop("trust_remote_code", False)),
        "device": kwargs.pop("device", None),
        "device_map": kwargs.pop("device_map", None),
        "torch_dtype": kwargs.pop("torch_dtype", None),
        "local_files_only": bool(kwargs.pop("local_files_only", False)),
        "attn_implementation": kwargs.pop("attn_implementation", None),
    }


def _normalize_h2o_transport_kwargs(kwargs: dict[str, Any], *, chat_cache) -> dict[str, Any]:
    request_kwargs = dict(kwargs)
    request_kwargs["h2o_enabled"] = bool(
        request_kwargs.get(
            "byte_h2o_enabled",
            request_kwargs.get("h2o_enabled", getattr(chat_cache.config, "h2o_enabled", False)),
        )
    )
    request_kwargs["h2o_heavy_ratio"] = float(
        request_kwargs.get(
            "byte_h2o_heavy_ratio",
            request_kwargs.get(
                "h2o_heavy_ratio",
                getattr(chat_cache.config, "h2o_heavy_ratio", 0.1),
            ),
        )
        or 0.0
    )
    request_kwargs["h2o_recent_ratio"] = float(
        request_kwargs.get(
            "byte_h2o_recent_ratio",
            request_kwargs.get(
                "h2o_recent_ratio",
                getattr(chat_cache.config, "h2o_recent_ratio", 0.1),
            ),
        )
        or 0.0
    )
    request_kwargs["kv_codec"] = str(
        request_kwargs.get(
            "byte_kv_codec",
            request_kwargs.get("kv_codec", getattr(chat_cache.config, "kv_codec", "disabled")),
        )
        or "disabled"
    )
    request_kwargs["kv_bits"] = int(
        request_kwargs.get(
            "byte_kv_bits",
            request_kwargs.get("kv_bits", getattr(chat_cache.config, "kv_bits", 8)),
        )
        or 8
    )
    request_kwargs["kv_hot_window_ratio"] = float(
        request_kwargs.get(
            "byte_kv_hot_window_ratio",
            request_kwargs.get(
                "kv_hot_window_ratio",
                getattr(chat_cache.config, "kv_hot_window_ratio", 0.25),
            ),
        )
        or 0.0
    )
    request_kwargs["compression_mode"] = str(
        request_kwargs.get(
            "byte_compression_mode",
            request_kwargs.get(
                "compression_mode", getattr(chat_cache.config, "compression_mode", "shadow")
            ),
        )
        or "shadow"
    )
    request_kwargs["compression_backend_policy"] = str(
        request_kwargs.get(
            "compression_backend_policy",
            getattr(chat_cache.config, "compression_backend_policy", "auto"),
        )
        or "auto"
    )
    request_kwargs["compression_verify_shadow_rate"] = float(
        request_kwargs.get(
            "compression_verify_shadow_rate",
            getattr(chat_cache.config, "compression_verify_shadow_rate", 0.1),
        )
        or 0.0
    )
    return request_kwargs


def _runtime_request_kwargs(kwargs: dict[str, Any], *, chat_cache) -> dict[str, Any]:
    request_kwargs = dict(kwargs)
    request_kwargs.pop("api_key", None)
    request_kwargs.pop("api_base", None)
    request_kwargs.pop("base_url", None)
    request_kwargs.pop("host", None)
    request_kwargs["byte_h2o_enabled"] = bool(
        request_kwargs.pop(
            "h2o_enabled",
            request_kwargs.pop("byte_h2o_enabled", getattr(chat_cache.config, "h2o_enabled", False)),
        )
    )
    request_kwargs["byte_h2o_heavy_ratio"] = float(
        request_kwargs.pop(
            "h2o_heavy_ratio",
            request_kwargs.pop("byte_h2o_heavy_ratio", getattr(chat_cache.config, "h2o_heavy_ratio", 0.1)),
        )
        or 0.0
    )
    request_kwargs["byte_h2o_recent_ratio"] = float(
        request_kwargs.pop(
            "h2o_recent_ratio",
            request_kwargs.pop("byte_h2o_recent_ratio", getattr(chat_cache.config, "h2o_recent_ratio", 0.1)),
        )
        or 0.0
    )
    request_kwargs["byte_kv_codec"] = str(
        request_kwargs.pop(
            "kv_codec",
            request_kwargs.pop("byte_kv_codec", getattr(chat_cache.config, "kv_codec", "disabled")),
        )
        or "disabled"
    )
    request_kwargs["byte_kv_bits"] = int(
        request_kwargs.pop(
            "kv_bits",
            request_kwargs.pop("byte_kv_bits", getattr(chat_cache.config, "kv_bits", 8)),
        )
        or 8
    )
    request_kwargs["byte_kv_hot_window_ratio"] = float(
        request_kwargs.pop(
            "kv_hot_window_ratio",
            request_kwargs.pop(
                "byte_kv_hot_window_ratio",
                getattr(chat_cache.config, "kv_hot_window_ratio", 0.25),
            ),
        )
        or 0.0
    )
    request_kwargs["byte_compression_mode"] = str(
        request_kwargs.pop(
            "compression_mode",
            request_kwargs.pop(
                "byte_compression_mode",
                getattr(chat_cache.config, "compression_mode", "shadow"),
            ),
        )
        or "shadow"
    )
    request_kwargs["compression_backend_policy"] = str(
        request_kwargs.pop(
            "compression_backend_policy",
            getattr(chat_cache.config, "compression_backend_policy", "auto"),
        )
        or "auto"
    )
    request_kwargs["compression_verify_shadow_rate"] = float(
        request_kwargs.pop(
            "compression_verify_shadow_rate",
            getattr(chat_cache.config, "compression_verify_shadow_rate", 0.1),
        )
        or 0.0
    )
    return request_kwargs


def _guess_model_family(model_name: str) -> str:
    lowered = str(model_name or "").strip().lower()
    if "mistral" in lowered:
        return "mistral"
    if "qwen2" in lowered or "qwen/" in lowered or "qwen-" in lowered:
        return "qwen2"
    if "phi-3" in lowered or "phi3" in lowered:
        return "phi3"
    if "gemma" in lowered:
        return "gemma"
    if "gpt-neox" in lowered or "pythia" in lowered:
        return "gpt_neox"
    if "opt" in lowered:
        return "opt"
    if "llama" in lowered:
        return "llama"
    return ""


def _runtime_descriptor_from_request(
    *,
    model_name: str,
    request_kwargs: dict[str, Any],
    cache_hit: bool,
    chat_cache,
) -> dict[str, Any]:
    fallback_reason = "cache_hit" if cache_hit else ""
    prompt_tokens = 0
    descriptor = describe_huggingface_runtime(
        model_name=model_name,
        model_family=_guess_model_family(model_name),
        prompt_tokens=prompt_tokens,
        cache_hit=cache_hit,
        h2o_enabled=bool(
            request_kwargs.get(
                "byte_h2o_enabled",
                request_kwargs.get("h2o_enabled", getattr(chat_cache.config, "h2o_enabled", False)),
            )
        ),
        h2o_heavy_ratio=float(
            request_kwargs.get(
                "byte_h2o_heavy_ratio",
                request_kwargs.get("h2o_heavy_ratio", getattr(chat_cache.config, "h2o_heavy_ratio", 0.1)),
            )
            or 0.0
        ),
        h2o_recent_ratio=float(
            request_kwargs.get(
                "byte_h2o_recent_ratio",
                request_kwargs.get(
                    "h2o_recent_ratio", getattr(chat_cache.config, "h2o_recent_ratio", 0.1)
                ),
            )
            or 0.0
        ),
        kv_codec=str(
            request_kwargs.get(
                "byte_kv_codec",
                request_kwargs.get("kv_codec", getattr(chat_cache.config, "kv_codec", "disabled")),
            )
            or "disabled"
        ),
        kv_bits=int(
            request_kwargs.get(
                "byte_kv_bits",
                request_kwargs.get("kv_bits", getattr(chat_cache.config, "kv_bits", 8)),
            )
            or 8
        ),
        kv_hot_window_ratio=float(
            request_kwargs.get(
                "byte_kv_hot_window_ratio",
                request_kwargs.get(
                    "kv_hot_window_ratio",
                    getattr(chat_cache.config, "kv_hot_window_ratio", 0.25),
                ),
            )
            or 0.0
        ),
        compression_mode=str(
            request_kwargs.get(
                "byte_compression_mode",
                request_kwargs.get(
                    "compression_mode", getattr(chat_cache.config, "compression_mode", "shadow")
                ),
            )
            or "shadow"
        ),
        compression_backend_policy=str(
            request_kwargs.get(
                "compression_backend_policy",
                getattr(chat_cache.config, "compression_backend_policy", "auto"),
            )
            or "auto"
        ),
        compression_verify_shadow_rate=float(
            request_kwargs.get(
                "compression_verify_shadow_rate",
                getattr(chat_cache.config, "compression_verify_shadow_rate", 0.1),
            )
            or 0.0
        ),
    ).to_dict()
    if fallback_reason:
        descriptor["h2o_fallback_reason"] = fallback_reason
    descriptor["cache_hit"] = cache_hit
    return descriptor


def _chat_response_from_cache(return_message: str, saved_token: list[int], runtime_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "byte": True,
        "saved_token": saved_token,
        "byte_provider": "huggingface",
        "byte_runtime": runtime_payload,
        "choices": [
            {
                "message": {"role": "assistant", "content": return_message},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {
            "completion_tokens": saved_token[1],
            "prompt_tokens": saved_token[0],
            "total_tokens": saved_token[0] + saved_token[1],
        },
        "object": "chat.completion",
    }


def _chat_stream_from_cache(
    return_message: str,
    saved_token: list[int],
    runtime_payload: dict[str, Any],
) -> list[Any]:
    created = int(time.time())
    return [
        {
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}],
            "created": created,
            "object": "chat.completion.chunk",
        },
        {
            "choices": [{"delta": {"content": return_message}, "finish_reason": None, "index": 0}],
            "created": created,
            "object": "chat.completion.chunk",
        },
        {
            "byte": True,
            "byte_provider": "huggingface",
            "byte_runtime": runtime_payload,
            "saved_token": saved_token,
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": created,
            "usage": {
                "completion_tokens": saved_token[1],
                "prompt_tokens": saved_token[0],
                "total_tokens": saved_token[0] + saved_token[1],
            },
            "object": "chat.completion.chunk",
        },
    ]


def _completion_response_from_cache(
    return_text: str,
    runtime_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "byte": True,
        "byte_provider": "huggingface",
        "byte_runtime": runtime_payload,
        "choices": [
            {
                "text": return_text,
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "text_completion",
    }


def _completion_stream_from_cache(
    return_text: str,
    runtime_payload: dict[str, Any],
) -> list[Any]:
    return [
        {
            "byte": True,
            "byte_provider": "huggingface",
            "byte_runtime": runtime_payload,
            "choices": [
                {
                    "text": return_text,
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
            "object": "text_completion",
        }
    ]


async def _async_iter(input_list) -> Any:
    for item in input_list:
        yield item


class ChatCompletion(BaseCacheLLM):
    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        runtime_options = _runtime_options(llm_kwargs)
        chat_cache = llm_kwargs.get("cache_obj", cache)
        request_kwargs = _runtime_request_kwargs(llm_kwargs, chat_cache=chat_cache)
        if cls.llm is not None:
            return cls.llm(*llm_args, **request_kwargs)
        runtime = get_huggingface_runtime(
            request_kwargs["model"],
            **runtime_options,
        )
        return runtime.generate_chat(**request_kwargs)

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        if llm_kwargs.get("stream", False):
            return cls._llm_handler(*llm_args, **llm_kwargs)
        return await run_sync(cls._llm_handler, *llm_args, **llm_kwargs)

    @staticmethod
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:
        if isinstance(llm_data, AsyncGenerator):

            async def hook_data(it) -> Any:
                total_answer = ""
                async for item in it:
                    delta = (((item.get("choices") or [{}])[0] or {}).get("delta") or {}).get(
                        "content", ""
                    )
                    total_answer += delta if isinstance(delta, str) else ""
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_data(llm_data)
        if not isinstance(llm_data, Iterator):
            update_cache_func(
                Answer(llm_data["choices"][0]["message"]["content"], DataType.STR)
            )
            return llm_data

        def hook_data(it) -> Any:
            total_answer = ""
            for item in it:
                delta = (((item.get("choices") or [{}])[0] or {}).get("delta") or {}).get(
                    "content", ""
                )
                total_answer += delta if isinstance(delta, str) else ""
                yield item
            update_cache_func(Answer(total_answer, DataType.STR))

        return hook_data(llm_data)

    @classmethod
    def create(cls, *args, **kwargs) -> Any:
        kwargs = cls.fill_base_args(**kwargs)
        chat_cache = kwargs.get("cache_obj", cache)
        kwargs = _normalize_h2o_transport_kwargs(kwargs, chat_cache=chat_cache)

        def cache_data_convert(cache_data) -> Any:
            saved_token = [0, 0]
            runtime_payload = _runtime_descriptor_from_request(
                model_name=str(kwargs.get("model", "") or ""),
                request_kwargs=kwargs,
                cache_hit=True,
                chat_cache=chat_cache,
            )
            if kwargs.get("stream", False):
                return _chat_stream_from_cache(cache_data, saved_token, runtime_payload)
            return _chat_response_from_cache(cache_data, saved_token, runtime_payload)

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
        kwargs = _normalize_h2o_transport_kwargs(kwargs, chat_cache=chat_cache)

        def cache_data_convert(cache_data) -> Any:
            saved_token = [0, 0]
            runtime_payload = _runtime_descriptor_from_request(
                model_name=str(kwargs.get("model", "") or ""),
                request_kwargs=kwargs,
                cache_hit=True,
                chat_cache=chat_cache,
            )
            if kwargs.get("stream", False):
                return _async_iter(_chat_stream_from_cache(cache_data, saved_token, runtime_payload))
            return _chat_response_from_cache(cache_data, saved_token, runtime_payload)

        return await aadapt(
            cls._allm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )


class Completion(BaseCacheLLM):
    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        runtime_options = _runtime_options(llm_kwargs)
        chat_cache = llm_kwargs.get("cache_obj", cache)
        request_kwargs = _runtime_request_kwargs(llm_kwargs, chat_cache=chat_cache)
        if cls.llm is not None:
            return cls.llm(*llm_args, **request_kwargs)
        runtime = get_huggingface_runtime(
            request_kwargs["model"],
            **runtime_options,
        )
        return runtime.generate_completion(**request_kwargs)

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        if llm_kwargs.get("stream", False):
            return cls._llm_handler(*llm_args, **llm_kwargs)
        return await run_sync(cls._llm_handler, *llm_args, **llm_kwargs)

    @staticmethod
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:
        if not isinstance(llm_data, Iterator):
            update_cache_func(Answer(llm_data["choices"][0]["text"], DataType.STR))
            return llm_data

        def hook_data(it) -> Any:
            total_answer = ""
            for item in it:
                text = (((item.get("choices") or [{}])[0] or {}).get("text") or "")
                total_answer += text if isinstance(text, str) else ""
                yield item
            update_cache_func(Answer(total_answer, DataType.STR))

        return hook_data(llm_data)

    @classmethod
    def create(cls, *args, **kwargs) -> Any:
        kwargs = cls.fill_base_args(**kwargs)
        chat_cache = kwargs.get("cache_obj", cache)
        kwargs = _normalize_h2o_transport_kwargs(kwargs, chat_cache=chat_cache)

        def cache_data_convert(cache_data) -> Any:
            runtime_payload = _runtime_descriptor_from_request(
                model_name=str(kwargs.get("model", "") or ""),
                request_kwargs=kwargs,
                cache_hit=True,
                chat_cache=chat_cache,
            )
            if kwargs.get("stream", False):
                return _completion_stream_from_cache(cache_data, runtime_payload)
            return _completion_response_from_cache(cache_data, runtime_payload)

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
        kwargs = _normalize_h2o_transport_kwargs(kwargs, chat_cache=chat_cache)

        def cache_data_convert(cache_data) -> Any:
            runtime_payload = _runtime_descriptor_from_request(
                model_name=str(kwargs.get("model", "") or ""),
                request_kwargs=kwargs,
                cache_hit=True,
                chat_cache=chat_cache,
            )
            if kwargs.get("stream", False):
                return _async_iter(_completion_stream_from_cache(cache_data, runtime_payload))
            return _completion_response_from_cache(cache_data, runtime_payload)

        return await aadapt(
            cls._allm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )
