"""Unit tests for byte.adapter.deepseek."""

import time
from unittest.mock import patch

import pytest

from byte import Cache, Config
from byte._backends import deepseek as cache_deepseek
from byte.manager.factory import get_data_manager
from byte.similarity_evaluation.exact_match import ExactMatchEvaluation

QUESTION = "What is Byte?"
ANSWER = "Byte caches repeated AI requests."


class _MockResponse:
    def __init__(self, payload, status_code=200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> object:
        return self._payload


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
    cache_deepseek.ChatCompletion.llm = None
    cache_deepseek.ChatCompletion.cache_args = {}
    yield
    cache_deepseek.ChatCompletion.llm = None
    cache_deepseek.ChatCompletion.cache_args = {}


def test_deepseek_adapter_returns_openai_compatible_payload() -> None:
    payload = {
        "id": "deepseek-1",
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": ANSWER,
                    "reasoning_content": "internal scratchpad",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 4,
            "total_tokens": 13,
            "prompt_tokens_details": {"cached_tokens": 3},
            "prompt_cache_hit_tokens": 3,
            "prompt_cache_miss_tokens": 6,
        },
    }

    with patch("byte._backends.deepseek.request_json", return_value=_MockResponse(payload)):
        response = cache_deepseek.ChatCompletion._llm_handler(
            model="deepseek-chat",
            api_key="test-key",
            messages=[{"role": "user", "content": QUESTION}],
        )

    assert response["byte_provider"] == "deepseek"
    assert response["choices"][0]["message"]["content"] == ANSWER
    assert response["choices"][0]["message"]["reasoning_content"] == "internal scratchpad"
    assert response["usage"]["total_tokens"] == 13
    assert response["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
    assert response["usage"]["prompt_cache_hit_tokens"] == 3
    assert response["usage"]["prompt_cache_miss_tokens"] == 6


def test_deepseek_cache_hit_skips_llm() -> object:
    cache_obj = _make_cache_obj()
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return {
            "byte_provider": "deepseek",
            "choices": [
                {
                    "message": {"role": "assistant", "content": ANSWER},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "model": kwargs.get("model", "deepseek-chat"),
            "object": "chat.completion",
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }

    cache_deepseek.ChatCompletion.llm = fake_llm
    messages = [{"role": "user", "content": QUESTION}]

    cache_deepseek.ChatCompletion.create(
        model="deepseek-chat",
        messages=messages,
        cache_obj=cache_obj,
    )
    hit = cache_deepseek.ChatCompletion.create(
        model="deepseek-chat",
        messages=messages,
        cache_obj=cache_obj,
    )

    assert call_count[0] == 1
    assert hit["byte"] is True
    assert hit["choices"][0]["message"]["content"] == ANSWER
