"""Unit tests for the Groq and OpenRouter adapters."""

import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("groq")

from byte import Cache
from byte._backends import groq as cache_groq
from byte._backends import openrouter as cache_openrouter
from byte.config import Config
from byte.manager.factory import get_data_manager
from byte.similarity_evaluation.exact_match import ExactMatchEvaluation


def _make_compat_response(text, provider="groq") -> object:
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
        "model": "test-model",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_cache_obj(ttl=None, model_namespace=False) -> object:
    cache_obj = Cache()
    import uuid

    cache_obj.init(
        data_manager=get_data_manager(data_path=f"data_map_{uuid.uuid4().hex}.txt"),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(
            ttl=ttl,
            model_namespace=model_namespace,
            ambiguity_detection=False,
            planner_enabled=False,
        ),
    )
    return cache_obj


@pytest.fixture(autouse=True)
def reset_state() -> object:
    cache_groq.ChatCompletion.llm = None
    cache_groq.ChatCompletion.cache_args = {}
    cache_groq.Audio.llm = None
    cache_groq.Speech.llm = None
    cache_openrouter.ChatCompletion.llm = None
    cache_openrouter.ChatCompletion.cache_args = {}
    cache_openrouter.Image.llm = None
    yield


class TestGroqChatCaching:
    def test_cache_miss_calls_llm(self) -> object:
        cache_obj = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_compat_response("Groq answer")

        cache_groq.ChatCompletion.llm = fake_llm

        response = cache_groq.ChatCompletion.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "What is Groq?"}],
            cache_obj=cache_obj,
        )

        assert call_count[0] == 1
        assert response["choices"][0]["message"]["content"] == "Groq answer"
        assert response["byte_provider"] == "groq"

    def test_cache_hit_skips_llm(self) -> object:
        cache_obj = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_compat_response("Cached Groq")

        cache_groq.ChatCompletion.llm = fake_llm
        messages = [{"role": "user", "content": "LPU stands for?"}]

        cache_groq.ChatCompletion.create(
            model="llama-3.3-70b-versatile", messages=messages, cache_obj=cache_obj
        )
        hit = cache_groq.ChatCompletion.create(
            model="llama-3.3-70b-versatile", messages=messages, cache_obj=cache_obj
        )

        assert call_count[0] == 1
        assert hit["byte"] is True
        assert hit["usage"]["total_tokens"] == 0

    def test_two_agents_are_isolated(self) -> object:
        agent_a = _make_cache_obj()
        agent_b = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_compat_response("answer")

        cache_groq.ChatCompletion.llm = fake_llm
        messages = [{"role": "user", "content": "Same question"}]

        cache_groq.ChatCompletion.create(
            model="llama-3.3-70b-versatile", messages=messages, cache_obj=agent_a
        )
        cache_groq.ChatCompletion.create(
            model="llama-3.3-70b-versatile", messages=messages, cache_obj=agent_b
        )

        assert call_count[0] == 2


def test_groq_audio_and_speech_surfaces_cache() -> None:
    cache_obj = _make_cache_obj()
    cache_groq.Audio.llm = lambda **kwargs: {"byte_provider": "groq", "text": "hello from groq"}
    cache_groq.Speech.llm = lambda **kwargs: {
        "byte_provider": "groq",
        "audio": b"groq-speech",
        "format": "mp3",
    }

    audio_hit = cache_groq.Audio.transcribe(
        "whisper-large-v3-turbo",
        {"name": "clip.wav", "bytes": b"RIFFtest", "mime_type": "audio/wav"},
        cache_obj=cache_obj,
    )
    audio_cached = cache_groq.Audio.transcribe(
        "whisper-large-v3-turbo",
        {"name": "clip.wav", "bytes": b"RIFFtest", "mime_type": "audio/wav"},
        cache_obj=cache_obj,
    )
    speech_first = cache_groq.Speech.create(
        "playai-tts",
        "Byte saves inference cost.",
        "Fritz-PlayAI",
        cache_obj=cache_obj,
    )
    speech_cached = cache_groq.Speech.create(
        "playai-tts",
        "Byte saves inference cost.",
        "Fritz-PlayAI",
        cache_obj=cache_obj,
    )

    assert audio_hit["text"] == "hello from groq"
    assert audio_cached["byte"] is True
    assert speech_first["audio"] == b"groq-speech"
    assert speech_cached["byte"] is True


class TestOpenRouterCaching:
    def test_cache_miss_calls_llm(self) -> object:
        cache_obj = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_compat_response("OpenRouter answer", provider="openrouter")

        cache_openrouter.ChatCompletion.llm = fake_llm

        response = cache_openrouter.ChatCompletion.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": "What is OpenRouter?"}],
            cache_obj=cache_obj,
        )

        assert call_count[0] == 1
        assert response["choices"][0]["message"]["content"] == "OpenRouter answer"
        assert response["byte_provider"] == "openrouter"

    def test_cache_hit_skips_llm(self) -> object:
        cache_obj = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_compat_response("Cached OR", provider="openrouter")

        cache_openrouter.ChatCompletion.llm = fake_llm
        messages = [{"role": "user", "content": "Explain caching"}]

        cache_openrouter.ChatCompletion.create(
            model="google/gemini-2.0-flash-001", messages=messages, cache_obj=cache_obj
        )
        hit = cache_openrouter.ChatCompletion.create(
            model="google/gemini-2.0-flash-001", messages=messages, cache_obj=cache_obj
        )

        assert call_count[0] == 1
        assert hit["byte"] is True
        assert hit["usage"]["total_tokens"] == 0

    def test_tool_calls_are_cached(self) -> object:
        cache_obj = _make_cache_obj()
        call_count = [0]

        tool_call_resp = {
            "byte_provider": "openrouter",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "model": "gpt-4o",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
        }

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return tool_call_resp

        cache_openrouter.ChatCompletion.llm = fake_llm
        messages = [{"role": "user", "content": "What's the weather in Paris?"}]

        cache_openrouter.ChatCompletion.create(
            model="openai/gpt-4o", messages=messages, cache_obj=cache_obj
        )
        hit = cache_openrouter.ChatCompletion.create(
            model="openai/gpt-4o", messages=messages, cache_obj=cache_obj
        )

        assert call_count[0] == 1
        assert hit["byte"] is True


def test_openrouter_image_generation_is_cacheable() -> None:
    cache_obj = _make_cache_obj()
    mock_client = MagicMock()
    mock_client.images.generate = MagicMock(
        return_value={"data": [{"b64_json": "aW1hZ2UtYnl0ZXM="}]}
    )

    with patch("byte.adapter.openrouter._get_client", return_value=mock_client):
        first = cache_openrouter.Image.create(
            model="openai/gpt-image-1",
            prompt="A futuristic cache server",
            response_format="b64_json",
            cache_obj=cache_obj,
        )
        second = cache_openrouter.Image.create(
            model="openai/gpt-image-1",
            prompt="A futuristic cache server",
            response_format="b64_json",
            cache_obj=cache_obj,
        )

    assert first["data"][0]["b64_json"]
    assert second["byte"] is True


def test_ttl_recent_entry_is_served() -> object:
    cache_obj = _make_cache_obj(ttl=3600)
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_compat_response("fresh answer", provider="groq")

    cache_groq.ChatCompletion.llm = fake_llm
    messages = [{"role": "user", "content": "TTL fresh"}]

    cache_groq.ChatCompletion.create(
        model="llama-3.3-70b-versatile", messages=messages, cache_obj=cache_obj
    )
    hit = cache_groq.ChatCompletion.create(
        model="llama-3.3-70b-versatile", messages=messages, cache_obj=cache_obj
    )

    assert call_count[0] == 1
    assert hit["byte"] is True


def test_ttl_expired_entry_triggers_fresh_call() -> object:
    cache_obj = _make_cache_obj(ttl=0)
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_compat_response("answer", provider="groq")

    cache_groq.ChatCompletion.llm = fake_llm
    messages = [{"role": "user", "content": "TTL expired"}]

    cache_groq.ChatCompletion.create(
        model="llama-3.3-70b-versatile", messages=messages, cache_obj=cache_obj
    )
    time.sleep(0.05)
    cache_groq.ChatCompletion.create(
        model="llama-3.3-70b-versatile", messages=messages, cache_obj=cache_obj
    )

    assert call_count[0] == 2


def test_model_namespace_keeps_models_separate() -> object:
    cache_obj = _make_cache_obj(model_namespace=True)
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_compat_response(f"answer {call_count[0]}", provider="openrouter")

    cache_openrouter.ChatCompletion.llm = fake_llm
    question = "What is the speed of light?"

    cache_openrouter.ChatCompletion.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": question}],
        cache_obj=cache_obj,
    )
    cache_openrouter.ChatCompletion.create(
        model="anthropic/claude-3-5-sonnet",
        messages=[{"role": "user", "content": question}],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 2


def test_model_namespace_off_allows_cross_model_hits() -> object:
    cache_obj = _make_cache_obj(model_namespace=False)
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_compat_response("shared answer", provider="openrouter")

    cache_openrouter.ChatCompletion.llm = fake_llm
    question = "What is Python?"

    cache_openrouter.ChatCompletion.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": question}],
        cache_obj=cache_obj,
    )
    hit = cache_openrouter.ChatCompletion.create(
        model="anthropic/claude-3-5-sonnet",
        messages=[{"role": "user", "content": question}],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 1
    assert hit["byte"] is True
