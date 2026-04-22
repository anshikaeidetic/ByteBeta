from unittest.mock import patch

import pytest

from byte import Cache, Config
from byte._backends import huggingface
from byte.adapter.api import init_exact_cache
from byte.processor.pre import get_prompt, last_content


class _FakeRuntime:
    def __init__(self) -> None:
        self.chat_calls = []
        self.completion_calls = []

    def generate_chat(self, **kwargs) -> object:
        self.chat_calls.append(kwargs)
        if kwargs.get("stream", False):
            return iter(
                [
                    {
                        "choices": [{"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}],
                        "object": "chat.completion.chunk",
                    },
                    {
                        "choices": [{"delta": {"content": "Byte"}, "finish_reason": None, "index": 0}],
                        "object": "chat.completion.chunk",
                    },
                    {
                        "choices": [{"delta": {"content": " H2O"}, "finish_reason": None, "index": 0}],
                        "object": "chat.completion.chunk",
                    },
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                        "object": "chat.completion.chunk",
                        "byte_provider": "huggingface",
                        "byte_runtime": {
                            "provider": "huggingface",
                            "model_family": "llama",
                            "h2o_applied": True,
                        },
                    },
                ]
            )
        return {
            "byte_provider": "huggingface",
            "byte_runtime": {
                "provider": "huggingface",
                "model_family": "llama",
                "h2o_requested": kwargs.get("byte_h2o_enabled", False),
                "h2o_applied": kwargs.get("byte_h2o_enabled", False),
                "byte_compression": {
                    "requested": kwargs.get("byte_kv_codec", "disabled") not in {"", "disabled"},
                    "applied_codec": kwargs.get("byte_kv_codec", "disabled"),
                    "bits": kwargs.get("byte_kv_bits", 8),
                },
            },
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Byte H2O works"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            "object": "chat.completion",
        }

    def generate_completion(self, **kwargs) -> object:
        self.completion_calls.append(kwargs)
        if kwargs.get("stream", False):
            return iter(
                [
                    {
                        "choices": [{"text": "Byte", "finish_reason": None, "index": 0}],
                        "object": "text_completion",
                    },
                    {
                        "choices": [{"text": " H2O", "finish_reason": None, "index": 0}],
                        "object": "text_completion",
                    },
                ]
            )
        return {
            "byte_provider": "huggingface",
            "byte_runtime": {
                "provider": "huggingface",
                "model_family": "llama",
                "h2o_requested": kwargs.get("byte_h2o_enabled", False),
                "h2o_applied": kwargs.get("byte_h2o_enabled", False),
                "byte_compression": {
                    "requested": kwargs.get("byte_kv_codec", "disabled") not in {"", "disabled"},
                    "applied_codec": kwargs.get("byte_kv_codec", "disabled"),
                    "bits": kwargs.get("byte_kv_bits", 8),
                },
            },
            "choices": [{"text": "Byte H2O works", "finish_reason": "stop", "index": 0}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            "object": "text_completion",
        }


def _chat_cache(tmp_path) -> object:
    cache_obj = Cache()
    init_exact_cache(
        data_dir=str(tmp_path / "chat"),
        cache_obj=cache_obj,
        pre_func=last_content,
        config=Config(enable_token_counter=False, h2o_enabled=True),
    )
    return cache_obj


def _completion_cache(tmp_path) -> object:
    cache_obj = Cache()
    init_exact_cache(
        data_dir=str(tmp_path / "completion"),
        cache_obj=cache_obj,
        pre_func=get_prompt,
        config=Config(enable_token_counter=False, h2o_enabled=True),
    )
    return cache_obj


@pytest.fixture(autouse=True)
def reset_adapter_state() -> object:
    huggingface.ChatCompletion.llm = None
    huggingface.Completion.llm = None
    yield
    huggingface.ChatCompletion.llm = None
    huggingface.Completion.llm = None


def test_huggingface_chat_completion_uses_runtime_and_caches(tmp_path) -> None:
    cache_obj = _chat_cache(tmp_path)
    runtime = _FakeRuntime()

    with patch("byte._backends.huggingface.get_huggingface_runtime", return_value=runtime):
        response = huggingface.ChatCompletion.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": "Say hello"}],
            cache_obj=cache_obj,
            byte_h2o_enabled=True,
            byte_h2o_heavy_ratio=0.2,
            byte_h2o_recent_ratio=0.2,
            byte_kv_codec="turboquant",
            byte_kv_bits=6,
            byte_compression_mode="guarded",
        )

    assert response["choices"][0]["message"]["content"] == "Byte H2O works"
    assert response["byte_provider"] == "huggingface"
    assert response["byte_runtime"]["h2o_applied"] is True
    assert runtime.chat_calls[0]["byte_h2o_enabled"] is True
    assert runtime.chat_calls[0]["byte_kv_codec"] == "turboquant"
    assert runtime.chat_calls[0]["byte_kv_bits"] == 6
    assert runtime.chat_calls[0]["byte_compression_mode"] == "guarded"

    cached = huggingface.ChatCompletion.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[{"role": "user", "content": "Say hello"}],
        cache_obj=cache_obj,
        byte_h2o_enabled=True,
        byte_h2o_heavy_ratio=0.2,
        byte_h2o_recent_ratio=0.2,
        byte_kv_codec="turboquant",
    )

    assert cached["byte"] is True
    assert cached["byte_runtime"]["provider"] == "huggingface"
    assert cached["byte_runtime"]["h2o_requested"] is True
    assert cached["byte_runtime"]["cache_hit"] is True
    assert cached["byte_runtime"]["byte_compression"]["requested"] is True
    assert cached["byte_runtime"]["byte_compression"]["requested_codec"] == "turboquant"


def test_huggingface_chat_streaming_caches_full_answer(tmp_path) -> None:
    cache_obj = _chat_cache(tmp_path)
    runtime = _FakeRuntime()

    with patch("byte._backends.huggingface.get_huggingface_runtime", return_value=runtime):
        response = huggingface.ChatCompletion.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": "Stream hello"}],
            cache_obj=cache_obj,
            stream=True,
            byte_h2o_enabled=True,
        )
        chunks = list(response)

    assert "".join(
        (((chunk.get("choices") or [{}])[0] or {}).get("delta") or {}).get("content", "")
        for chunk in chunks
    ) == "Byte H2O"

    cached = list(
        huggingface.ChatCompletion.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": "Stream hello"}],
            cache_obj=cache_obj,
            stream=True,
            byte_h2o_enabled=True,
        )
    )

    assert cached[-1]["byte"] is True
    assert cached[-1]["byte_runtime"]["provider"] == "huggingface"


def test_huggingface_completion_uses_runtime_and_caches(tmp_path) -> None:
    cache_obj = _completion_cache(tmp_path)
    runtime = _FakeRuntime()

    with patch("byte._backends.huggingface.get_huggingface_runtime", return_value=runtime):
        response = huggingface.Completion.create(
            model="facebook/opt-125m",
            prompt="Complete this sentence",
            cache_obj=cache_obj,
            byte_h2o_enabled=True,
        )

    assert response["choices"][0]["text"] == "Byte H2O works"
    assert runtime.completion_calls[0]["byte_h2o_enabled"] is True

    cached = huggingface.Completion.create(
        model="facebook/opt-125m",
        prompt="Complete this sentence",
        cache_obj=cache_obj,
        byte_h2o_enabled=True,
    )

    assert cached["byte"] is True
    assert cached["byte_runtime"]["provider"] == "huggingface"


@pytest.mark.asyncio
async def test_huggingface_async_chat_completion(tmp_path) -> None:
    cache_obj = _chat_cache(tmp_path)
    runtime = _FakeRuntime()

    with patch("byte._backends.huggingface.get_huggingface_runtime", return_value=runtime):
        response = await huggingface.ChatCompletion.acreate(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[{"role": "user", "content": "Say hello"}],
            cache_obj=cache_obj,
            byte_h2o_enabled=True,
        )

    assert response["choices"][0]["message"]["content"] == "Byte H2O works"
    assert response["byte_runtime"]["provider"] == "huggingface"
