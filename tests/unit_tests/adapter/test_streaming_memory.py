from byte import Cache, Config
from byte._backends import openai
from byte.adapter.api import init_exact_cache
from byte.processor.pre import last_content


def _read_stream_text(chunks) -> object:
    output = []
    for chunk in chunks:
        choices = chunk.get("choices", []) or []
        if not choices:
            continue
        delta = choices[0].get("delta", {}) or {}
        if delta.get("content"):
            output.append(delta["content"])
    return "".join(output)


def test_streaming_miss_records_execution_memory_and_enables_verified_reuse(tmp_path) -> object:
    cache_obj = Cache()
    init_exact_cache(
        data_dir=str(tmp_path),
        cache_obj=cache_obj,
        pre_func=last_content,
        config=Config(
            enable_token_counter=False,
            execution_memory=True,
            verified_reuse_for_all=True,
        ),
    )

    call_count = {"value": 0}

    def fake_streaming_llm(*args, **kwargs) -> object:
        call_count["value"] += 1

        def generator() -> object:
            yield {
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "model": kwargs.get("model", ""),
            }
            yield {
                "choices": [{"index": 0, "delta": {"content": "STREAM_OK"}, "finish_reason": None}],
                "model": kwargs.get("model", ""),
            }
            yield {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "model": kwargs.get("model", ""),
            }

        return generator()

    original_llm = openai.ChatCompletion.llm
    try:
        openai.ChatCompletion.llm = fake_streaming_llm

        first_chunks = list(
            openai.ChatCompletion.create(
                model="gpt-4o-mini",
                stream=True,
                cache_obj=cache_obj,
                messages=[{"role": "user", "content": "Reply with STREAM_OK"}],
                byte_test_result={"passed": True},
                byte_schema_validation={"passed": True},
                byte_tool_checks={"passed": True},
            )
        )
        assert _read_stream_text(first_chunks) == "STREAM_OK"
        assert cache_obj.execution_memory_stats()["verified_entries"] == 1
        assert cache_obj.ai_memory_stats()["total_entries"] == 1

        second_chunks = list(
            openai.ChatCompletion.create(
                model="gpt-4o-mini",
                stream=True,
                cache_obj=cache_obj,
                messages=[{"role": "user", "content": "Reply with STREAM_OK"}],
            )
        )

        assert _read_stream_text(second_chunks) == "STREAM_OK"
        assert call_count["value"] == 1
        assert cache_obj.report.hint_cache_count == 1
    finally:
        openai.ChatCompletion.llm = original_llm


def test_streaming_update_cache_callback_ignores_non_content_chunks(tmp_path) -> object:
    cache_obj = Cache()
    init_exact_cache(
        data_dir=str(tmp_path),
        cache_obj=cache_obj,
        pre_func=last_content,
        config=Config(
            enable_token_counter=False,
            execution_memory=True,
        ),
    )

    def fake_streaming_llm(*args, **kwargs) -> object:
        def generator() -> object:
            yield {
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "model": kwargs.get("model", ""),
            }
            yield {
                "choices": [{"index": 0, "delta": {"content": "STREAM_OK"}, "finish_reason": None}],
                "model": kwargs.get("model", ""),
            }
            yield {
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                "model": kwargs.get("model", ""),
            }
            yield {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "model": kwargs.get("model", ""),
            }

        return generator()

    original_llm = openai.ChatCompletion.llm
    try:
        openai.ChatCompletion.llm = fake_streaming_llm
        chunks = list(
            openai.ChatCompletion.create(
                model="gpt-4o-mini",
                stream=True,
                cache_obj=cache_obj,
                messages=[{"role": "user", "content": "Reply with STREAM_OK"}],
            )
        )
        assert _read_stream_text(chunks) == "STREAM_OK"
        assert cache_obj.ai_memory_stats()["total_entries"] == 1
    finally:
        openai.ChatCompletion.llm = original_llm
