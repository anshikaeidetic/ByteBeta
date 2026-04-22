import pytest

pytest.importorskip("groq")

from byte import Cache, Config
from byte._backends import anthropic as cache_anthropic
from byte._backends import bedrock as cache_bedrock
from byte._backends import cohere as cache_cohere
from byte._backends import deepseek as cache_deepseek
from byte._backends import gemini as cache_gemini
from byte._backends import groq as cache_groq
from byte._backends import mistral as cache_mistral
from byte._backends import ollama as cache_ollama
from byte._backends import openai as cache_openai
from byte._backends import openrouter as cache_openrouter
from byte.adapter.api import init_cache
from byte.processor.pre import last_content, normalized_last_content

ADAPTER_CASES = [
    (cache_openai, "openai", "gpt-4o-mini", True),
    (cache_deepseek, "deepseek", "deepseek-chat", False),
    (cache_anthropic, "anthropic", "claude-sonnet-4-20250514", False),
    (cache_gemini, "gemini", "gemini-2.0-flash", False),
    (cache_groq, "groq", "llama-3.3-70b-versatile", False),
    (cache_openrouter, "openrouter", "meta-llama/llama-3.3-70b-instruct", False),
    (cache_ollama, "ollama", "llama3.2", False),
    (cache_mistral, "mistral", "mistral-small-latest", False),
    (cache_cohere, "cohere", "command-r", False),
    (cache_bedrock, "bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0", False),
]


@pytest.fixture(autouse=True)
def reset_adapters() -> object:
    modules = [case[0] for case in ADAPTER_CASES]
    for adapter in modules:
        adapter.ChatCompletion.llm = None
        adapter.ChatCompletion.cache_args = {}
    yield
    for adapter in modules:
        adapter.ChatCompletion.llm = None
        adapter.ChatCompletion.cache_args = {}


@pytest.mark.parametrize("adapter_module,provider,model,expects_openai_key", ADAPTER_CASES)
def test_native_prompt_cache_bridge_is_safe_for_all_adapters(
    tmp_path, adapter_module, provider, model, expects_openai_key
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="exact",
        data_dir=str(tmp_path / provider),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            native_prompt_caching=True,
            native_prompt_cache_min_chars=1,
        ),
    )
    observed_kwargs = {}

    def fake_llm(**kwargs) -> object:
        observed_kwargs.update(kwargs)
        return {
            "byte_provider": provider,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "model": kwargs.get("model", model),
            "object": "chat.completion",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    adapter_module.ChatCompletion.llm = fake_llm
    adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a support classifier."},
            {
                "role": "user",
                "content": (
                    "Classify the billing ticket with the standard support taxonomy and reply with exactly one label. "
                    "This prompt is intentionally long enough to trigger native prompt cache bridge behavior."
                ),
            },
        ],
        cache_obj=cache_obj,
    )

    assert "native_prompt_cache_key" not in observed_kwargs
    assert "native_prompt_cache_ttl" not in observed_kwargs
    if expects_openai_key:
        assert observed_kwargs.get("prompt_cache_key")
    else:
        assert "prompt_cache_key" not in observed_kwargs
