"""Unit tests for byte.adapter.anthropic (Claude) with full mock-based coverage.

Tests verify:
- Cache miss â†’ calls LLM â†’ stores in cache
- Cache hit â†’ returns cached answer without calling LLM
- Cost savings: second identical call is FREE (hits cache)
- Multi-agent isolation: two agents with separate cache_obj don't share results
- Response format normalization (Anthropic â†’ OpenAI-compatible)

NOTE: The .llm attribute bypasses the network call entirely; our fake_llm must
return the same dict structure that _llm_handler produces after calling
_response_to_openai_format(), because _update_cache_callback reads from that dict.
"""

import base64
import time
from unittest.mock import MagicMock, patch

import pytest

from byte import Cache, Config
from byte._backends import anthropic as cache_anthropic
from byte.adapter.api import init_cache
from byte.manager.factory import get_data_manager
from byte.processor.pre import last_content, normalized_last_content
from byte.similarity_evaluation.exact_match import ExactMatchEvaluation

QUESTION = "What is the capital of France?"
ANSWER = "The capital of France is Paris."


def _make_openai_compat_response(text, provider="anthropic") -> object:
    """Return an OpenAI-compatible dict (what _llm_handler produces after conversion)."""
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
        "id": "msg_abc123",
        "model": "claude-sonnet-4-20250514",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


def _make_cache_obj() -> object:
    c = Cache()
    import uuid

    c.init(
        data_manager=get_data_manager(data_path=f"data_map_{uuid.uuid4().hex}.txt"),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(ambiguity_detection=False, planner_enabled=False),
    )
    return c


@pytest.fixture(autouse=True)
def reset_state() -> object:
    """Reset Anthropic adapter state before every test."""
    cache_anthropic.ChatCompletion.llm = None
    cache_anthropic.ChatCompletion.cache_args = {}
    yield


class TestAnthropicCacheMiss:
    def test_cache_miss_calls_llm(self) -> object:
        """On a cold cache, the LLM should be called and the result cached."""
        cache_obj = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response(ANSWER)

        cache_anthropic.ChatCompletion.llm = fake_llm

        response = cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": QUESTION}],
            max_tokens=256,
            cache_obj=cache_obj,
        )

        assert call_count[0] == 1
        assert response["choices"][0]["message"]["content"] == ANSWER

    def test_cache_hit_skips_llm(self) -> object:
        """Second identical request should hit cache â€” LLM NOT called again."""
        cache_obj = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response(ANSWER)

        cache_anthropic.ChatCompletion.llm = fake_llm

        msgs = [{"role": "user", "content": QUESTION}]

        # First call â€” miss
        cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=msgs,
            max_tokens=256,
            cache_obj=cache_obj,
        )
        # Second call â€” should be a cache hit
        response = cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=msgs,
            max_tokens=256,
            cache_obj=cache_obj,
        )

        assert call_count[0] == 1, "LLM should only be called ONCE (second call is a hit)"
        assert response["choices"][0]["message"]["content"] == ANSWER

    def test_response_format_is_openai_compatible(self) -> None:
        """Response from a cache miss must be OpenAI-compatible dict format."""
        cache_obj = _make_cache_obj()
        cache_anthropic.ChatCompletion.llm = lambda **kw: _make_openai_compat_response(ANSWER)

        response = cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": QUESTION}],
            cache_obj=cache_obj,
        )

        assert "choices" in response
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert response["object"] == "chat.completion"

    def test_cache_hit_response_has_byte_flag(self) -> None:
        """Cache hit response must include byte=True to distinguish from live LLM calls."""
        cache_obj = _make_cache_obj()
        cache_anthropic.ChatCompletion.llm = lambda **kw: _make_openai_compat_response(ANSWER)

        msgs = [{"role": "user", "content": QUESTION}]
        cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=msgs,
            cache_obj=cache_obj,
        )
        hit_response = cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=msgs,
            cache_obj=cache_obj,
        )

        assert hit_response.get("byte") is True


class TestAnthropicMultiAgent:
    def test_separate_agents_have_isolated_caches(self) -> object:
        """Two agents with separate cache_obj instances must NOT share results."""
        agent_a_cache = _make_cache_obj()
        agent_b_cache = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response(f"Response {call_count[0]}")

        cache_anthropic.ChatCompletion.llm = fake_llm

        msgs = [{"role": "user", "content": "Hello from agent!"}]

        # Agent A fills its cache
        cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=msgs,
            cache_obj=agent_a_cache,
        )
        assert call_count[0] == 1

        # Agent B should be a miss (different cache) â†’ LLM called again
        cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=msgs,
            cache_obj=agent_b_cache,
        )
        assert call_count[0] == 2, "Agent B should have its own cache â€” must call LLM"

    def test_same_agent_cache_hits(self) -> object:
        """Same agent hitting the cache twice should only call LLM once."""
        agent_cache = _make_cache_obj()
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response("Cached answer")

        cache_anthropic.ChatCompletion.llm = fake_llm
        msgs = [{"role": "user", "content": "Agent question repeated"}]

        cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514", messages=msgs, cache_obj=agent_cache
        )
        cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514", messages=msgs, cache_obj=agent_cache
        )

        assert call_count[0] == 1


class TestAnthropicCostSavings:
    def test_cost_savings_counter(self) -> None:
        """Cache hits return usage tokens=0 â€” demonstrating API cost savings."""
        cache_obj = _make_cache_obj()
        cache_anthropic.ChatCompletion.llm = lambda **kw: _make_openai_compat_response("Saved!")

        msgs = [{"role": "user", "content": "Expensive call"}]
        cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514", messages=msgs, cache_obj=cache_obj
        )
        hit = cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514", messages=msgs, cache_obj=cache_obj
        )

        # Cache hits report zero token usage â†’ zero API cost
        usage = hit.get("usage", {})
        assert usage.get("total_tokens", 0) == 0, "Cache hit should report 0 tokens (= $0 API cost)"

    def test_normalized_mode_is_provider_agnostic(self, tmp_path) -> object:
        """Normalized caching should work with Anthropic via the shared cache API."""
        cache_obj = Cache()
        init_cache(
            mode="normalized",
            data_dir=str(tmp_path),
            cache_obj=cache_obj,
            pre_func=last_content,
            normalized_pre_func=normalized_last_content,
            config=Config(enable_token_counter=False),
        )
        call_count = [0]

        def fake_llm(**kwargs) -> object:
            call_count[0] += 1
            return _make_openai_compat_response("Normalized Anthropic hit")

        cache_anthropic.ChatCompletion.llm = fake_llm

        cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "  WHAT is Byte Cache??!!  "}],
            cache_obj=cache_obj,
        )
        hit = cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "what is byte cache"}],
            cache_obj=cache_obj,
        )

        assert call_count[0] == 1
        assert hit.get("byte") is True

    def test_multimodal_messages_are_converted_to_anthropic_blocks(self) -> object:
        cache_obj = _make_cache_obj()
        captured = {}
        mock_client = MagicMock()

        def fake_create(**kwargs) -> object:
            captured.update(kwargs)
            return _make_openai_compat_response("Vision answer")

        mock_client.messages.create = fake_create
        pdf_b64 = base64.b64encode(b"%PDF-1.4 test").decode("ascii")

        with patch("byte.adapter.anthropic._get_client", return_value=mock_client):
            response = cache_anthropic.ChatCompletion.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[
                    {"role": "system", "content": "Be concise."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Summarize these inputs."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,aGVsbG8=",
                                    "name": "diagram.png",
                                },
                            },
                            {
                                "type": "file",
                                "file": {
                                    "filename": "spec.pdf",
                                    "file_data": pdf_b64,
                                    "mime_type": "application/pdf",
                                },
                            },
                        ],
                    },
                ],
                cache_obj=cache_obj,
            )

        assert response["choices"][0]["message"]["content"] == "Vision answer"
        assert captured["system"][0]["text"] == "Be concise."
        blocks = captured["messages"][0]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image"
        assert blocks[2]["type"] == "document"
