"""Unit tests for byte.adapter.gemini (Google Gemini)."""

import time
from unittest.mock import MagicMock, patch

import pytest

from byte import Cache, Config
from byte._backends import gemini as cache_gemini
from byte.manager.factory import get_data_manager
from byte.similarity_evaluation.exact_match import ExactMatchEvaluation

QUESTION = "What does GPU stand for?"
ANSWER = "GPU stands for Graphics Processing Unit."


def _make_openai_compat_response(text, model="gemini-2.0-flash") -> object:
    return {
        "byte_provider": "gemini",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "model": model,
        "object": "chat.completion",
        "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
    }


def _make_cache_obj() -> object:
    cache_obj = Cache()
    import uuid

    cache_obj.init(
        data_manager=get_data_manager(data_path=f"data_map_{uuid.uuid4().hex}.txt"),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(ambiguity_detection=False, planner_enabled=False),
    )
    return cache_obj


@pytest.fixture(autouse=True)
def reset_state() -> object:
    cache_gemini.ChatCompletion.llm = None
    cache_gemini.ChatCompletion.cache_args = {}
    cache_gemini.Audio.llm = None
    cache_gemini.Speech.llm = None
    cache_gemini.Image.llm = None
    yield


class TestGeminiChatCaching:
    def test_cache_miss_calls_llm(self) -> object:
        cache_obj = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response(ANSWER)

        cache_gemini.ChatCompletion.llm = fake_llm

        response = cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": QUESTION}],
            cache_obj=cache_obj,
        )

        assert call_count[0] == 1
        assert response["choices"][0]["message"]["content"] == ANSWER

    def test_cache_hit_skips_llm(self) -> object:
        cache_obj = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response(ANSWER)

        cache_gemini.ChatCompletion.llm = fake_llm
        messages = [{"role": "user", "content": QUESTION}]

        cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash", messages=messages, cache_obj=cache_obj
        )
        response = cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash", messages=messages, cache_obj=cache_obj
        )

        assert call_count[0] == 1
        assert response["choices"][0]["message"]["content"] == ANSWER
        assert response["byte"] is True

    def test_system_prompt_does_not_break_caching(self) -> object:
        cache_obj = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response("System-aware answer")

        cache_gemini.ChatCompletion.llm = fake_llm

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "System-prompted question"},
        ]

        cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash", messages=messages, cache_obj=cache_obj
        )
        cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash", messages=messages, cache_obj=cache_obj
        )

        assert call_count[0] == 1

    def test_two_agents_are_isolated(self) -> object:
        agent_a = _make_cache_obj()
        agent_b = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response(f"Answer {call_count[0]}")

        cache_gemini.ChatCompletion.llm = fake_llm
        messages = [{"role": "user", "content": "Shared question?"}]

        cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash", messages=messages, cache_obj=agent_a
        )
        cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash", messages=messages, cache_obj=agent_b
        )

        assert call_count[0] == 2


def test_multimodal_messages_are_converted_for_gemini() -> object:
    cache_obj = _make_cache_obj()
    captured = {}
    mock_client = MagicMock()

    def fake_generate_content(**kwargs) -> object:
        captured.update(kwargs)
        return _make_openai_compat_response("Gemini multimodal", model="gemini-2.0-flash")

    mock_client.models.generate_content = fake_generate_content

    with patch("byte.adapter.gemini._get_client", return_value=mock_client):
        response = cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "Be concise."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image and transcribe the clip."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                        },
                        {
                            "type": "input_audio",
                            "audio": {
                                "data": "UklGRg==",
                                "mime_type": "audio/wav",
                                "name": "clip.wav",
                            },
                        },
                        {
                            "type": "file",
                            "file": {
                                "file_data": "JVBERi0xLjQgdGVzdA==",
                                "mime_type": "application/pdf",
                                "filename": "spec.pdf",
                            },
                        },
                    ],
                },
            ],
            cache_obj=cache_obj,
        )

    assert response["choices"][0]["message"]["content"] == "Gemini multimodal"
    assert captured["config"].system_instruction == "Be concise."
    assert len(captured["contents"][0]["parts"]) == 4


def test_gemini_audio_transcribe_caches_text() -> None:
    cache_obj = _make_cache_obj()
    cache_gemini.Audio.llm = lambda **kwargs: {"byte_provider": "gemini", "text": "hello world"}

    first = cache_gemini.Audio.transcribe(
        "gemini-2.0-flash",
        {"name": "clip.wav", "bytes": b"RIFFtest", "mime_type": "audio/wav"},
        cache_obj=cache_obj,
    )
    second = cache_gemini.Audio.transcribe(
        "gemini-2.0-flash",
        {"name": "clip.wav", "bytes": b"RIFFtest", "mime_type": "audio/wav"},
        cache_obj=cache_obj,
    )

    assert first["text"] == "hello world"
    assert second["text"] == "hello world"
    assert second["byte"] is True


def test_gemini_speech_caches_audio_payload() -> None:
    cache_obj = _make_cache_obj()
    cache_gemini.Speech.llm = lambda **kwargs: {
        "byte_provider": "gemini",
        "audio": b"speech-bytes",
        "format": "wav",
        "mime_type": "audio/wav",
    }

    first = cache_gemini.Speech.create(
        "gemini-2.5-flash-preview-tts",
        "Hello from Byte",
        "Kore",
        cache_obj=cache_obj,
    )
    second = cache_gemini.Speech.create(
        "gemini-2.5-flash-preview-tts",
        "Hello from Byte",
        "Kore",
        cache_obj=cache_obj,
    )

    assert first["audio"] == b"speech-bytes"
    assert second["audio"] == b"speech-bytes"
    assert second["byte"] is True


def test_gemini_image_generation_returns_cacheable_images() -> None:
    cache_obj = _make_cache_obj()
    mock_client = MagicMock()
    mock_client.models.generate_images = MagicMock(
        return_value={
            "generated_images": [
                {
                    "image": {
                        "image_bytes": b"image-bytes",
                        "mime_type": "image/png",
                    }
                }
            ]
        }
    )

    with patch("byte.adapter.gemini._get_client", return_value=mock_client):
        first = cache_gemini.Image.create(
            model="imagen-3.0-generate-002",
            prompt="A cache rocket",
            response_format="b64_json",
            cache_obj=cache_obj,
        )
        second = cache_gemini.Image.create(
            model="imagen-3.0-generate-002",
            prompt="A cache rocket",
            response_format="b64_json",
            cache_obj=cache_obj,
        )

    assert first["data"][0]["b64_json"]
    assert second["byte"] is True
