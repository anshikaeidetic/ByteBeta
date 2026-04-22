import time

from byte import Cache, Config
from byte._backends import anthropic as cache_anthropic
from byte._backends import gemini as cache_gemini
from byte.adapter.api import init_cache
from byte.processor.pre import last_content, normalized_last_content
from byte.processor.shared_memory import clear_shared_memory
from byte.session import Session


def _anthropic_response(text) -> object:
    return {
        "byte_provider": "anthropic",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "model": "claude-sonnet-4-20250514",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 18, "completion_tokens": 6, "total_tokens": 24},
    }


def _gemini_response(text) -> object:
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
        "model": "gemini-2.0-flash",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
    }


def test_memory_scope_shares_intent_graph_across_providers(tmp_path) -> None:
    clear_shared_memory("provider-shared")
    anthropic_cache = Cache()
    gemini_cache = Cache()
    config = Config(enable_token_counter=False, memory_scope="provider-shared", intent_memory=True)

    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / "anthropic"),
        cache_obj=anthropic_cache,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=config,
    )
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / "gemini"),
        cache_obj=gemini_cache,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=config,
    )

    cache_anthropic.ChatCompletion.llm = lambda **kwargs: _anthropic_response("POSITIVE")
    cache_gemini.ChatCompletion.llm = lambda **kwargs: _gemini_response("Byte Cache is fast.")
    cache_anthropic.ChatCompletion.cache_args = {}
    cache_gemini.ChatCompletion.cache_args = {}

    anthropic_session = Session(name="shared-flow", data_manager=anthropic_cache.data_manager)
    gemini_session = Session(name="shared-flow", data_manager=gemini_cache.data_manager)

    try:
        cache_anthropic.ChatCompletion.create(
            model="claude-sonnet-4-20250514",
            messages=[
                {
                    "role": "user",
                    "content": 'Classify the sentiment. Labels: POSITIVE, NEGATIVE, NEUTRAL Review: "I loved it."',
                }
            ],
            cache_obj=anthropic_cache,
            session=anthropic_session,
        )
        cache_gemini.ChatCompletion.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": 'Summarize the following article in one sentence. Article: "Byte Cache reduces repeated LLM calls."',
                }
            ],
            cache_obj=gemini_cache,
            session=gemini_session,
        )
    finally:
        cache_anthropic.ChatCompletion.llm = None
        cache_gemini.ChatCompletion.llm = None
        cache_anthropic.ChatCompletion.cache_args = {}
        cache_gemini.ChatCompletion.cache_args = {}

    anthropic_stats = anthropic_cache.intent_stats()
    gemini_stats = gemini_cache.intent_stats()
    anthropic_ai_memory = anthropic_cache.ai_memory_stats()
    gemini_ai_memory = gemini_cache.ai_memory_stats()

    assert anthropic_stats == gemini_stats
    assert anthropic_stats["total_records"] == 2
    assert anthropic_stats["transition_count"] == 1
    assert anthropic_stats["top_transitions"][0]["from"] == "classification"
    assert anthropic_stats["top_transitions"][0]["to"] == "summarization::one_sentence"
    assert anthropic_ai_memory == gemini_ai_memory
    assert anthropic_ai_memory["total_entries"] == 2
    assert anthropic_ai_memory["categories"]["classification"] == 1
    assert anthropic_ai_memory["categories"]["summarization"] == 1
