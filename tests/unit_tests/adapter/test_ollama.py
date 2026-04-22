"""Unit tests for byte.adapter.ollama."""

import time
from unittest.mock import MagicMock, patch

import pytest

from byte import Cache, Config
from byte._backends import ollama as cache_ollama
from byte.manager.factory import get_data_manager
from byte.similarity_evaluation.exact_match import ExactMatchEvaluation

QUESTION = "What is 2 + 2?"
ANSWER = "2 + 2 equals 4."


def _make_openai_compat_response(text, model="llama3.2") -> object:
    return {
        "byte_provider": "ollama",
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
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
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
    cache_ollama.ChatCompletion.llm = None
    cache_ollama.ChatCompletion.cache_args = {}
    yield


class TestOllamaCaching:
    def test_cache_miss_calls_llm(self) -> object:
        cache_obj = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response(ANSWER)

        cache_ollama.ChatCompletion.llm = fake_llm

        response = cache_ollama.ChatCompletion.create(
            model="llama3.2",
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

        cache_ollama.ChatCompletion.llm = fake_llm
        messages = [{"role": "user", "content": QUESTION}]

        cache_ollama.ChatCompletion.create(model="llama3.2", messages=messages, cache_obj=cache_obj)
        response = cache_ollama.ChatCompletion.create(
            model="llama3.2", messages=messages, cache_obj=cache_obj
        )

        assert call_count[0] == 1
        assert response["choices"][0]["message"]["content"] == ANSWER
        assert response["byte"] is True

    def test_two_agents_are_isolated(self) -> object:
        agent_a = _make_cache_obj()
        agent_b = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response("Answer")

        cache_ollama.ChatCompletion.llm = fake_llm
        messages = [{"role": "user", "content": "Multi-agent test question"}]

        cache_ollama.ChatCompletion.create(model="llama3.2", messages=messages, cache_obj=agent_a)
        cache_ollama.ChatCompletion.create(model="llama3.2", messages=messages, cache_obj=agent_b)

        assert call_count[0] == 2


def test_ollama_multimodal_messages_map_to_images_field() -> object:
    cache_obj = _make_cache_obj()
    captured = {}
    mock_client = MagicMock()

    def fake_chat(**kwargs) -> object:
        captured.update(kwargs)
        return {
            "message": {"role": "assistant", "content": "It is a chart."},
            "done": True,
            "model": "llama3.2-vision",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

    mock_client.chat = fake_chat

    with patch("byte.adapter.ollama._get_client", return_value=mock_client):
        response = cache_ollama.ChatCompletion.create(
            model="llama3.2-vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                        },
                    ],
                }
            ],
            cache_obj=cache_obj,
        )

    assert response["choices"][0]["message"]["content"] == "It is a chart."
    assert captured["messages"][0]["images"]
    assert captured["messages"][0]["content"] == "Describe the image."


def test_ollama_security_mode_blocks_client_host_override() -> None:
    cache_obj = Cache()
    import uuid

    cache_obj.init(
        data_manager=get_data_manager(data_path=f"data_map_{uuid.uuid4().hex}.txt"),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(
            ambiguity_detection=False,
            planner_enabled=False,
            security_mode=True,
        ),
    )

    cache_ollama.ChatCompletion.llm = lambda **kwargs: _make_openai_compat_response("ok")

    with pytest.raises(Exception, match="host overrides are disabled"):
        cache_ollama.ChatCompletion.create(
            model="llama3.2",
            host="http://localhost:11434",
            messages=[{"role": "user", "content": QUESTION}],
            cache_obj=cache_obj,
        )
