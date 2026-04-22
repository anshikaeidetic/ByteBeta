import importlib
import time
from pathlib import Path

import pytest

from byte import Cache, Config
from byte._backends import openai as cache_openai
from byte.manager import manager_factory
from byte.utils.response import get_stream_message_from_openai_answer

CHAT_CASES = [
    ("byte.adapter.openai", "gpt-4o-mini", True),
    ("byte.adapter.anthropic", "claude-sonnet-4-20250514", True),
    ("byte.adapter.gemini", "gemini-2.0-flash", True),
    ("byte.adapter.groq", "llama-3.3-70b-versatile", True),
    ("byte.adapter.openrouter", "meta-llama/llama-3.3-70b-instruct", True),
    ("byte.adapter.ollama", "llama3.2", True),
    ("byte.adapter.deepseek", "deepseek-chat", False),
    ("byte.adapter.mistral", "mistral-small-latest", False),
    ("byte.adapter.cohere", "command-r-plus", False),
    ("byte.adapter.bedrock", "anthropic.claude-3-haiku-20240307-v1:0", False),
]

ASYNC_STREAM_CASES = [
    ("byte.adapter.openai", "gpt-4o-mini"),
    ("byte.adapter.anthropic", "claude-sonnet-4-20250514"),
    ("byte.adapter.gemini", "gemini-2.0-flash"),
    ("byte.adapter.ollama", "llama3.2"),
]

THREADED_ASYNC_STREAM_CASES = [
    ("byte.adapter.deepseek", "deepseek-chat"),
    ("byte.adapter.mistral", "mistral-small-latest"),
    ("byte.adapter.cohere", "command-r-plus"),
    ("byte.adapter.bedrock", "anthropic.claude-3-haiku-20240307-v1:0"),
]

CACHED_STREAM_ONLY_CASES = [
    ("byte.adapter.deepseek", "deepseek-chat"),
    ("byte.adapter.mistral", "mistral-small-latest"),
]


def _load_adapter(module_name: str) -> object:
    return importlib.import_module(module_name)


def _make_cache(tmp_path: Path) -> Cache:
    cache_obj = Cache()
    cache_obj.init(
        data_manager=manager_factory("map", data_dir=str(tmp_path)),
        config=Config(
            enable_token_counter=False,
            ambiguity_detection=False,
            planner_enabled=False,
        ),
    )
    return cache_obj


def _chat_response(provider: str, model: str, text: str) -> object:
    return {
        "byte_provider": provider,
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "id": f"{provider}-async",
        "model": model,
        "object": "chat.completion",
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }


async def _streaming_chunks(*parts: str) -> object:
    for part in parts:
        yield {
            "choices": [
                {
                    "delta": {"content": part},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "object": "chat.completion.chunk",
        }
    yield {
        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
        "created": int(time.time()),
        "object": "chat.completion.chunk",
    }


async def _collect_stream_text(stream) -> str:
    parts = []
    async for chunk in stream:
        parts.append(get_stream_message_from_openai_answer(chunk))
    return "".join(parts)


def _sync_streaming_chunks(*parts: str) -> object:
    for part in parts:
        yield {
            "choices": [
                {
                    "delta": {"content": part},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "object": "chat.completion.chunk",
        }
    yield {
        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
        "created": int(time.time()),
        "object": "chat.completion.chunk",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("module_name", "model", "awaits_llm"), CHAT_CASES)
async def test_chat_acreate_hits_cache_across_adapters(
    tmp_path, monkeypatch, module_name, model, awaits_llm
) -> object:
    adapter = _load_adapter(module_name)
    cache_obj = _make_cache(tmp_path / module_name.rsplit(".", 1)[-1])
    expected = f"answer::{module_name.rsplit('.', 1)[-1]}"
    call_count = 0

    if awaits_llm:

        async def fake_llm(*args, **kwargs) -> object:
            nonlocal call_count
            call_count += 1
            return _chat_response(module_name.rsplit(".", 1)[-1], model, expected)
    else:

        def fake_llm(*args, **kwargs) -> object:
            nonlocal call_count
            call_count += 1
            return _chat_response(module_name.rsplit(".", 1)[-1], model, expected)

    monkeypatch.setattr(adapter.ChatCompletion, "llm", fake_llm, raising=False)
    monkeypatch.setattr(adapter.ChatCompletion, "cache_args", {}, raising=False)

    messages = [{"role": "user", "content": "Explain Byte in one line."}]

    first = await adapter.ChatCompletion.acreate(
        model=model, messages=messages, cache_obj=cache_obj
    )
    second = await adapter.ChatCompletion.acreate(
        model=model, messages=messages, cache_obj=cache_obj
    )

    assert first["choices"][0]["message"]["content"] == expected
    assert second["choices"][0]["message"]["content"] == expected
    assert second.get("byte") is True
    assert call_count == 1


@pytest.mark.asyncio
async def test_openai_completion_acreate_hits_cache(tmp_path, monkeypatch) -> object:
    cache_obj = _make_cache(tmp_path / "openai-completion")
    expected = "completion::byte"
    call_count = 0

    async def fake_llm(*args, **kwargs) -> object:
        nonlocal call_count
        call_count += 1
        return {
            "choices": [{"text": expected, "finish_reason": "stop", "index": 0}],
            "created": int(time.time()),
            "id": "cmpl-async",
            "model": "gpt-3.5-turbo-instruct",
            "object": "text_completion",
        }

    monkeypatch.setattr(cache_openai.Completion, "llm", fake_llm, raising=False)
    monkeypatch.setattr(cache_openai.Completion, "cache_args", {}, raising=False)

    first = await cache_openai.Completion.acreate(
        model="gpt-3.5-turbo-instruct",
        prompt="Return completion::byte",
        cache_obj=cache_obj,
    )
    second = await cache_openai.Completion.acreate(
        model="gpt-3.5-turbo-instruct",
        prompt="Return completion::byte",
        cache_obj=cache_obj,
    )

    assert first["choices"][0]["text"] == expected
    assert second["choices"][0]["text"] == expected
    assert second.get("byte") is True
    assert call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("module_name", "model"), ASYNC_STREAM_CASES)
async def test_chat_acreate_async_streams_fill_cache(tmp_path, monkeypatch, module_name, model) -> object:
    adapter = _load_adapter(module_name)
    cache_obj = _make_cache(tmp_path / f"{module_name.rsplit('.', 1)[-1]}-stream")
    expected = "Byte async stream"
    call_count = 0

    async def fake_llm(*args, **kwargs) -> object:
        nonlocal call_count
        call_count += 1
        return _streaming_chunks("Byte ", "async ", "stream")

    monkeypatch.setattr(adapter.ChatCompletion, "llm", fake_llm, raising=False)
    monkeypatch.setattr(adapter.ChatCompletion, "cache_args", {}, raising=False)

    messages = [{"role": "user", "content": "Stream a short answer."}]

    first = await adapter.ChatCompletion.acreate(
        model=model,
        messages=messages,
        stream=True,
        cache_obj=cache_obj,
    )

    assert await _collect_stream_text(first) == expected
    second = await adapter.ChatCompletion.acreate(
        model=model,
        messages=messages,
        stream=True,
        cache_obj=cache_obj,
    )
    assert await _collect_stream_text(second) == expected
    assert call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("module_name", "model"), THREADED_ASYNC_STREAM_CASES)
async def test_chat_acreate_threaded_async_streams_fill_cache(
    tmp_path, monkeypatch, module_name, model
) -> object:
    adapter = _load_adapter(module_name)
    cache_obj = _make_cache(tmp_path / f"{module_name.rsplit('.', 1)[-1]}-threaded-stream")
    expected = "Byte threaded stream"
    call_count = 0

    def fake_llm(*args, **kwargs) -> object:
        nonlocal call_count
        call_count += 1
        return _sync_streaming_chunks("Byte ", "threaded ", "stream")

    monkeypatch.setattr(adapter.ChatCompletion, "llm", fake_llm, raising=False)
    monkeypatch.setattr(adapter.ChatCompletion, "cache_args", {}, raising=False)

    messages = [{"role": "user", "content": "Stream a short answer."}]

    first = await adapter.ChatCompletion.acreate(
        model=model,
        messages=messages,
        stream=True,
        cache_obj=cache_obj,
    )

    assert await _collect_stream_text(first) == expected
    second = await adapter.ChatCompletion.acreate(
        model=model,
        messages=messages,
        stream=True,
        cache_obj=cache_obj,
    )
    assert await _collect_stream_text(second) == expected
    assert call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("module_name", "model"), CACHED_STREAM_ONLY_CASES)
async def test_chat_acreate_cached_stream_works_for_threaded_async_adapters(
    tmp_path, monkeypatch, module_name, model
) -> object:
    adapter = _load_adapter(module_name)
    cache_obj = _make_cache(tmp_path / f"{module_name.rsplit('.', 1)[-1]}-cached-stream")
    expected = f"cached-stream::{module_name.rsplit('.', 1)[-1]}"

    def seed_llm(*args, **kwargs) -> object:
        return _chat_response(module_name.rsplit(".", 1)[-1], model, expected)

    def fail_llm(*args, **kwargs) -> None:
        raise AssertionError("cached async stream unexpectedly called upstream llm")

    monkeypatch.setattr(adapter.ChatCompletion, "llm", seed_llm, raising=False)
    monkeypatch.setattr(adapter.ChatCompletion, "cache_args", {}, raising=False)

    adapter.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "Prime the cache"}],
        cache_obj=cache_obj,
    )

    monkeypatch.setattr(adapter.ChatCompletion, "llm", fail_llm, raising=False)

    cached_stream = await adapter.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "user", "content": "Prime the cache"}],
        stream=True,
        cache_obj=cache_obj,
    )

    assert await _collect_stream_text(cached_stream) == expected
