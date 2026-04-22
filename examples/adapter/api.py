from byte import Cache, Config, cache
from byte.adapter.api import get, init_cache, init_similar_cache, put
from byte.processor.post import nop
from byte.processor.pre import get_prompt, last_content


def run_basic() -> None:
    cache.init(pre_embedding_func=get_prompt)
    put("hello", "foo")
    print(get("hello"))
    # output: foo


def run_similar_match() -> None:
    inner_cache = Cache()
    init_similar_cache(cache_obj=inner_cache, post_func=nop, config=Config(similarity_threshold=0))

    put("hello1", "foo1", cache_obj=inner_cache)
    put("hello2", "foo2", cache_obj=inner_cache)
    put("hello3", "foo3", cache_obj=inner_cache)

    messages = get("hello", cache_obj=inner_cache, top_k=3)
    print(messages)
    # output: ['foo1', 'foo2', 'foo3']


def run_provider_agnostic_normalized_mode() -> None:
    chat_cache = Cache()
    init_cache(
        mode="normalized",
        cache_obj=chat_cache,
        pre_func=last_content,
        config=Config(enable_token_counter=False),
    )

    put(
        "  WHAT is Byte Cache??!!  ",
        "A cache layer for repeated LLM work.",
        cache_obj=chat_cache,
    )
    print(get("what is byte cache", cache_obj=chat_cache))
    # output: A cache layer for repeated LLM work.


if __name__ == "__main__":
    run_basic()
    run_similar_match()
    run_provider_agnostic_normalized_mode()
