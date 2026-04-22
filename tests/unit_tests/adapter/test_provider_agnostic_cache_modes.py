from byte import Cache, Config
from byte._backends import gemini as cache_gemini
from byte.adapter.api import init_cache
from byte.processor.pre import last_content, normalized_last_content


def _make_gemini_response(text, model="gemini-2.0-flash") -> object:
    return {
        "byte_provider": "gemini",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": 0,
        "model": model,
        "object": "chat.completion",
        "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
    }


def test_normalized_mode_saves_cost_for_gemini_too(tmp_path) -> object:
    """The shared normalized mode should work for non-OpenAI adapters."""
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
        return _make_gemini_response("Normalized Gemini hit")

    cache_gemini.ChatCompletion.llm = fake_llm
    cache_gemini.ChatCompletion.cache_args = {}
    try:
        cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "  WHAT is Byte Cache??!!  "}],
            cache_obj=cache_obj,
        )
        hit = cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "what is byte cache"}],
            cache_obj=cache_obj,
        )
    finally:
        cache_gemini.ChatCompletion.llm = None
        cache_gemini.ChatCompletion.cache_args = {}

    assert call_count[0] == 1
    assert hit.get("byte") is True
    assert hit["usage"]["total_tokens"] == 0
