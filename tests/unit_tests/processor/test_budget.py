from byte.processor.budget import (
    estimate_request_cost,
    estimate_tokens_for_request,
    get_model_pricing,
    normalize_model_for_pricing,
)


def test_normalize_model_for_pricing_handles_provider_prefix_and_version_suffix() -> None:
    assert normalize_model_for_pricing("openai/gpt-4o-mini") == "gpt-4o-mini"
    assert normalize_model_for_pricing("gpt-4o-2024-08-06") == "gpt-4o"


def test_estimate_tokens_for_request_uses_messages_and_max_tokens() -> None:
    estimated = estimate_tokens_for_request(
        {
            "messages": [{"role": "user", "content": "Hello world from Byte"}],
            "max_tokens": 12,
        }
    )

    assert estimated["prompt_tokens"] >= 1
    assert estimated["completion_tokens"] == 12


def test_estimate_request_cost_returns_known_cost_for_prefixed_model() -> None:
    cost = estimate_request_cost(
        "openai/gpt-4o-mini",
        {
            "messages": [{"role": "user", "content": "Return exactly BYTE_OK and nothing else."}],
            "max_tokens": 8,
        },
    )

    assert cost is not None
    assert cost > 0


def test_get_model_pricing_returns_none_for_unknown_model() -> None:
    assert get_model_pricing("unknown-provider/unknown-model") is None
