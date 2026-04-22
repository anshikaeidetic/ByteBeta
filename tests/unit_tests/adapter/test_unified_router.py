import time

import pytest

import byte.adapter as byte_adapter
from byte import Cache, Config
from byte._backends import deepseek as cache_deepseek
from byte._backends import gemini as cache_gemini
from byte._backends import huggingface as cache_huggingface
from byte._backends import openai as cache_openai
from byte.adapter.api import init_exact_cache
from byte.adapter.router_runtime import (
    clear_model_aliases,
    clear_route_runtime_stats,
    register_model_alias,
    route_runtime_stats,
)
from byte.utils.error import CacheError


def _make_response(text, provider, model) -> object:
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
        "model": model,
        "object": "chat.completion",
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    }


@pytest.fixture(autouse=True)
def reset_state() -> object:
    clear_model_aliases()
    clear_route_runtime_stats()
    cache_openai.ChatCompletion.llm = None
    cache_deepseek.ChatCompletion.llm = None
    cache_gemini.ChatCompletion.llm = None
    cache_huggingface.ChatCompletion.llm = None
    cache_gemini.Image.llm = None
    yield
    clear_model_aliases()
    clear_route_runtime_stats()
    cache_openai.ChatCompletion.llm = None
    cache_deepseek.ChatCompletion.llm = None
    cache_gemini.ChatCompletion.llm = None
    cache_huggingface.ChatCompletion.llm = None
    cache_gemini.Image.llm = None


def _cache_obj(tmp_path) -> object:
    cache_obj = Cache()
    init_exact_cache(
        data_dir=str(tmp_path),
        cache_obj=cache_obj,
        config=Config(ambiguity_detection=False, planner_enabled=False),
    )
    return cache_obj


def test_chat_routes_byte_route_to_openai_backend(tmp_path) -> object:
    cache_obj = _cache_obj(tmp_path / "openai-route")
    register_model_alias("chat-default", ["openai/gpt-4o-mini"])
    calls = []

    def fake_openai(**kwargs) -> object:
        calls.append(kwargs["model"])
        return _make_response("openai-path", "openai", kwargs["model"])

    cache_openai.ChatCompletion.llm = fake_openai

    response = byte_adapter.ChatCompletion.create(
        model="chat-default",
        messages=[{"role": "user", "content": "Say openai-path"}],
        cache_obj=cache_obj,
    )

    assert calls == ["gpt-4o-mini"]
    assert response["choices"][0]["message"]["content"] == "openai-path"
    assert response["byte_router"]["selected_provider"] == "openai"
    assert response["byte_router"]["selected_target"] == "openai/gpt-4o-mini"


def test_chat_routes_byte_route_to_deepseek_backend(tmp_path) -> object:
    cache_obj = _cache_obj(tmp_path / "deepseek-route")
    register_model_alias("reasoning-default", ["deepseek/deepseek-chat"])
    calls = []

    def fake_deepseek(**kwargs) -> object:
        calls.append(kwargs["model"])
        return _make_response("deepseek-path", "deepseek", kwargs["model"])

    cache_deepseek.ChatCompletion.llm = fake_deepseek

    response = byte_adapter.ChatCompletion.create(
        model="reasoning-default",
        messages=[{"role": "user", "content": "Say deepseek-path"}],
        cache_obj=cache_obj,
    )

    assert calls == ["deepseek-chat"]
    assert response["choices"][0]["message"]["content"] == "deepseek-path"
    assert response["byte_router"]["selected_provider"] == "deepseek"
    assert response["byte_router"]["selected_target"] == "deepseek/deepseek-chat"


def test_alias_round_robin_rotates_targets(tmp_path) -> object:
    cache_obj = _cache_obj(tmp_path / "round-robin")
    register_model_alias("fast-chat", ["openai/gpt-4o-mini", "gemini/gemini-2.0-flash"])
    seen = []

    def fake_openai(**kwargs) -> object:
        seen.append(("openai", kwargs["model"]))
        return _make_response("openai", "openai", kwargs["model"])

    def fake_gemini(**kwargs) -> object:
        seen.append(("gemini", kwargs["model"]))
        return _make_response("gemini", "gemini", kwargs["model"])

    cache_openai.ChatCompletion.llm = fake_openai
    cache_gemini.ChatCompletion.llm = fake_gemini

    first = byte_adapter.ChatCompletion.create(
        model="fast-chat",
        messages=[{"role": "user", "content": "hello one"}],
        cache_obj=cache_obj,
        cache_skip=True,
        byte_routing_strategy="round_robin",
    )
    second = byte_adapter.ChatCompletion.create(
        model="fast-chat",
        messages=[{"role": "user", "content": "hello two"}],
        cache_obj=cache_obj,
        cache_skip=True,
        byte_routing_strategy="round_robin",
    )

    assert seen == [("openai", "gpt-4o-mini"), ("gemini", "gemini-2.0-flash")]
    assert first["byte_router"]["selected_provider"] == "openai"
    assert second["byte_router"]["selected_provider"] == "gemini"


def test_router_falls_back_to_next_provider_on_retryable_error(tmp_path) -> object:
    cache_obj = _cache_obj(tmp_path / "fallback")
    register_model_alias("resilient-chat", ["openai/gpt-4o-mini", "gemini/gemini-2.0-flash"])
    seen = []

    def failing_openai(**kwargs) -> None:
        seen.append(("openai", kwargs["model"]))
        raise RuntimeError("429 rate limit from upstream")

    def fake_gemini(**kwargs) -> object:
        seen.append(("gemini", kwargs["model"]))
        return _make_response("gemini-fallback", "gemini", kwargs["model"])

    cache_openai.ChatCompletion.llm = failing_openai
    cache_gemini.ChatCompletion.llm = fake_gemini

    response = byte_adapter.ChatCompletion.create(
        model="resilient-chat",
        messages=[{"role": "user", "content": "hello fallback"}],
        cache_obj=cache_obj,
    )

    assert seen == [("openai", "gpt-4o-mini"), ("gemini", "gemini-2.0-flash")]
    assert response["choices"][0]["message"]["content"] == "gemini-fallback"
    assert response["byte_router"]["fallback_used"] is True
    assert response["byte_router"]["attempted_targets"] == [
        "openai/gpt-4o-mini",
        "gemini/gemini-2.0-flash",
    ]
    stats = route_runtime_stats()["targets"]
    assert stats["openai/gpt-4o-mini"]["failures"] == 1
    assert stats["gemini/gemini-2.0-flash"]["successes"] == 1


def test_image_dispatches_supported_media_surface(tmp_path) -> object:
    cache_obj = _cache_obj(tmp_path / "image")
    register_model_alias(
        "image-default",
        ["gemini/gemini-2.0-flash-preview-image-generation"],
    )
    calls = []

    def fake_gemini_image(**kwargs) -> object:
        calls.append(kwargs["model"])
        return {
            "byte_provider": "gemini",
            "created": int(time.time()),
            "data": [{"b64_json": "aGVsbG8="}],
        }

    cache_gemini.Image.llm = fake_gemini_image

    response = byte_adapter.Image.create(
        model="image-default",
        prompt="Draw a small cat",
        response_format="b64_json",
        cache_obj=cache_obj,
    )

    assert calls == ["gemini-2.0-flash-preview-image-generation"]
    assert response["byte_router"]["selected_provider"] == "gemini"
    assert response["data"][0]["b64_json"] == "aGVsbG8="


def test_cost_strategy_prefers_cheapest_known_target(tmp_path) -> object:
    cache_obj = _cache_obj(tmp_path / "cost")
    register_model_alias("cost-chat", ["openai/gpt-4o", "openai/gpt-4o-mini"])
    seen = []

    def fake_openai(**kwargs) -> object:
        seen.append(kwargs["model"])
        return _make_response("cost-route", "openai", kwargs["model"])

    cache_openai.ChatCompletion.llm = fake_openai

    response = byte_adapter.ChatCompletion.create(
        model="cost-chat",
        messages=[{"role": "user", "content": "Return exactly cost-route"}],
        cache_obj=cache_obj,
        cache_skip=True,
        byte_routing_strategy="cost",
        max_tokens=16,
    )

    assert seen == ["gpt-4o-mini"]
    assert response["byte_router"]["selected_model"] == "gpt-4o-mini"
    assert response["byte_router"]["estimated_cost_usd"] is not None


def test_health_weighted_strategy_avoids_unhealthy_target(tmp_path) -> object:
    cache_obj = _cache_obj(tmp_path / "health")
    register_model_alias("health-chat", ["openai/gpt-4o-mini", "gemini/gemini-2.0-flash"])
    seen = []

    def failing_openai(**kwargs) -> None:
        seen.append(("openai", kwargs["model"]))
        raise RuntimeError("429 rate limit from upstream")

    def fake_gemini(**kwargs) -> object:
        seen.append(("gemini", kwargs["model"]))
        return _make_response("healthy", "gemini", kwargs["model"])

    cache_openai.ChatCompletion.llm = failing_openai
    cache_gemini.ChatCompletion.llm = fake_gemini

    first = byte_adapter.ChatCompletion.create(
        model="health-chat",
        messages=[{"role": "user", "content": "hello health one"}],
        cache_obj=cache_obj,
        cache_skip=True,
    )
    second = byte_adapter.ChatCompletion.create(
        model="health-chat",
        messages=[{"role": "user", "content": "hello health two"}],
        cache_obj=cache_obj,
        cache_skip=True,
        byte_routing_strategy="health_weighted",
    )

    assert first["byte_router"]["selected_provider"] == "gemini"
    assert second["byte_router"]["selected_provider"] == "gemini"
    assert seen[0] == ("openai", "gpt-4o-mini")
    assert seen[1] == ("gemini", "gemini-2.0-flash")
    assert seen[2] == ("gemini", "gemini-2.0-flash")


def test_speculative_routing_returns_fastest_success(tmp_path) -> object:
    cache_obj = _cache_obj(tmp_path / "speculative")
    cache_obj.config.speculative_routing = True
    cache_obj.config.speculative_max_parallel = 2
    register_model_alias("latency-chat", ["openai/gpt-4o-mini", "gemini/gemini-2.0-flash"])

    def slow_openai(**kwargs) -> object:
        time.sleep(0.05)
        return _make_response("slow-openai", "openai", kwargs["model"])

    def fast_gemini(**kwargs) -> object:
        return _make_response("fast-gemini", "gemini", kwargs["model"])

    cache_openai.ChatCompletion.llm = slow_openai
    cache_gemini.ChatCompletion.llm = fast_gemini

    response = byte_adapter.ChatCompletion.create(
        model="latency-chat",
        messages=[{"role": "user", "content": "Return the fastest provider"}],
        cache_obj=cache_obj,
        cache_skip=True,
    )

    assert response["choices"][0]["message"]["content"] == "fast-gemini"
    assert response["byte_router"]["strategy"].endswith(":speculative")


def test_backend_qualified_model_selector_is_rejected(tmp_path) -> None:
    cache_obj = _cache_obj(tmp_path / "public-api")

    with pytest.raises(CacheError):
        byte_adapter.ChatCompletion.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "This should fail"}],
            cache_obj=cache_obj,
        )


def test_chat_routes_local_huggingface_route_without_hosted_credentials(tmp_path) -> object:
    cache_obj = _cache_obj(tmp_path / "huggingface-route")
    register_model_alias("local-chat", ["huggingface/meta-llama/Llama-3.2-1B-Instruct"])
    seen = []

    def fake_huggingface(**kwargs) -> object:
        seen.append(kwargs["model"])
        return _make_response("local-h2o", "huggingface", kwargs["model"])

    cache_huggingface.ChatCompletion.llm = fake_huggingface

    response = byte_adapter.ChatCompletion.create(
        model="local-chat",
        messages=[{"role": "user", "content": "Say local-h2o"}],
        cache_obj=cache_obj,
        byte_h2o_enabled=True,
    )

    assert seen == ["meta-llama/Llama-3.2-1B-Instruct"]
    assert response["choices"][0]["message"]["content"] == "local-h2o"
    assert response["byte_router"]["selected_provider"] == "huggingface"
    assert response["byte_router"]["selected_target"] == "huggingface/meta-llama/Llama-3.2-1B-Instruct"
