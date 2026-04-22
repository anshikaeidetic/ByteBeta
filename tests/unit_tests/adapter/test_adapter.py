import asyncio
import os
import random
import time

import numpy
import pytest

from byte import Cache, Config, cache
from byte.adapter.adapter import aadapt, adapt
from byte.adapter.api import get, put
from byte.manager import get_data_manager, manager_factory
from byte.processor.post import first, nop
from byte.processor.pre import get_prompt, last_content
from byte.similarity_evaluation import ExactMatchEvaluation
from byte.utils.error import NotInitError
from byte.utils.time import time_cal

data_map_path = "data_map.txt"


def test_adapt() -> object:
    def llm_handler(*llm_args, **llm_kwargs) -> object:
        a = llm_kwargs.get("a", 0)
        b = llm_kwargs.get("b", 0)
        time.sleep(1)
        return a + b

    def cache_data_convert(cache_data) -> object:
        return int(cache_data)

    def custom_update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> object:
        update_cache_func(str(llm_data))
        return llm_data

    def custom_add_llm(*args, **kwargs) -> object:
        return adapt(llm_handler, cache_data_convert, custom_update_cache_callback, *args, **kwargs)

    def pre_embedding(data, **kwargs) -> object:
        a = data.get("a", 0)
        b = data.get("b", 0)
        return f"{a}+{b}"

    if os.path.isfile(data_map_path):
        os.remove(data_map_path)
    map_manager = get_data_manager()
    cache.init(pre_embedding_func=pre_embedding, data_manager=map_manager)

    def report_func(delta_time) -> None:
        assert 0.9 < delta_time < 1.1, delta_time

    def add1(**kwargs) -> None:
        res = custom_add_llm(a=1, b=2, **kwargs)
        assert res == 3, res

    time_cal(add1, report_func=report_func)()

    def delay_embedding(data, **kwargs) -> object:
        time.sleep(0.5)
        return data

    cache.init(
        pre_embedding_func=pre_embedding,
        embedding_func=delay_embedding,
        data_manager=map_manager,
        post_process_messages_func=first,
    )

    def report_func(delta_time) -> None:
        assert 1.4 < delta_time < 1.6, delta_time

    time_cal(add1, report_func=report_func)(cache_skip=True)

    def report_func(delta_time) -> None:
        assert delta_time < 0.6, delta_time

    time_cal(add1, report_func=report_func)()
    time_cal(add1, report_func=report_func)(cache_factor=0)
    time_cal(add1, report_func=report_func)(cache_factor=10)

    def custom_update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> object:
        time.sleep(0.5)
        update_cache_func(str(llm_data))
        return llm_data

    def disable_cache(*args, **kwargs) -> object:
        return False

    def report_func(delta_time) -> None:
        assert 0.9 < delta_time < 1.1, delta_time

    def custom_add_llm(*args, **kwargs) -> object:
        return adapt(llm_handler, cache_data_convert, custom_update_cache_callback, *args, **kwargs)

    def add2(**kwargs) -> None:
        res = custom_add_llm(a=1, b=2, **kwargs)
        assert res == 3, res

    cache.init(
        cache_enable_func=disable_cache,
        pre_embedding_func=pre_embedding,
        embedding_func=delay_embedding,
        data_manager=map_manager,
    )
    time_cal(add2, report_func=report_func)()


def test_not_init_cache() -> None:
    foo_cache = Cache()
    is_exception = False
    try:
        adapt(None, None, None, cache_obj=foo_cache)
    except NotInitError:
        is_exception = True

    assert is_exception


@pytest.mark.requires_feature("sqlalchemy", "faiss")
def test_cache_temperature() -> None:
    if os.path.exists("faiss.index"):
        os.remove("faiss.index")
    if os.path.exists("sqlite.db"):
        try:
            os.remove("sqlite.db")
        except PermissionError:
            pass
    data_manager = manager_factory("sqlite,faiss", vector_params={"dimension": 3, "top_k": 2})
    cache.init(
        pre_embedding_func=get_prompt,
        embedding_func=lambda x, **_: numpy.ones((3,)).astype("float32"),
        data_manager=data_manager,
        post_process_messages_func=nop,
    )
    assert cache.data_manager.v._top_k == 2
    prompt = "test"
    answer = "test answer"
    for _ in range(5):
        put(prompt=prompt, data=answer)

    answers = get(prompt=prompt, temperature=2.0)
    assert answers is None

    answers = get(prompt=prompt, temperature=1.5)
    assert answers in [None, [answer] * 5]

    answers = get(prompt=prompt, temperature=0.0, top_k=3)
    assert len(answers) == 3

    answers = get(prompt=prompt, temperature=0.0)
    assert len(answers) == 5

    answers = get(prompt=prompt)
    assert len(answers) == 2


@pytest.mark.requires_feature("transformers")
def test_input_summarization() -> object:
    cache_obj = Cache()

    def embedding_func(x, **_) -> object:
        assert len(x.split()) < 40
        return x

    cache_obj.init(
        pre_func=lambda x, **_: x.get("text"),
        embedding_func=embedding_func,
        data_manager=manager_factory(data_dir=str(random.random())),
        config=Config(input_summary_len=40),
    )
    adapt(
        lambda **_: 0,
        lambda **_: 0,
        lambda **_: 0,
        text="A large language model (LLM) is a language model consisting of a neural network with many parameters (typically billions of weights or more), trained on large quantities of unlabeled text using self-supervised learning or semi-supervised learning. LLMs emerged around 2018 and perform well at a wide variety of tasks. This has shifted the focus of natural language processing research away from the previous paradigm of training specialized supervised models for specific tasks.",
        cache_obj=cache_obj,
    )

    adapt(
        lambda **_: 0,
        lambda **_: 0,
        lambda **_: 0,
        text="A large language model (LLM)",
        cache_obj=cache_obj,
    )


def test_tool_namespace_prevents_cross_tool_collisions() -> object:
    cache_obj = Cache()
    cache_obj.init(
        pre_embedding_func=last_content,
        embedding_func=lambda value, **_: value,
        data_manager=manager_factory("map", data_dir=str(random.random())),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(tool_namespace=True),
    )
    call_count = [0]

    def llm_handler(**kwargs) -> object:
        call_count[0] += 1
        return kwargs["tools"][0]["function"]["name"]

    def custom_update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> object:
        update_cache_func(llm_data)
        return llm_data

    weather_request = {
        "messages": [{"role": "user", "content": "What is the answer?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    stocks_request = {
        "messages": [{"role": "user", "content": "What is the answer?"}],
        "tools": [{"type": "function", "function": {"name": "get_stock_price"}}],
    }

    weather = adapt(
        llm_handler,
        lambda data: data,
        custom_update_cache_callback,
        cache_obj=cache_obj,
        **weather_request,
    )
    stocks = adapt(
        llm_handler,
        lambda data: data,
        custom_update_cache_callback,
        cache_obj=cache_obj,
        **stocks_request,
    )
    weather_hit = adapt(
        llm_handler,
        lambda data: data,
        custom_update_cache_callback,
        cache_obj=cache_obj,
        **weather_request,
    )

    assert weather == "get_weather"
    assert stocks == "get_stock_price"
    assert weather_hit == "get_weather"
    assert call_count[0] == 2


def test_context_fingerprint_partitions_multi_turn_queries() -> object:
    cache_obj = Cache()
    cache_obj.init(
        pre_embedding_func=last_content,
        embedding_func=lambda value, **_: value,
        data_manager=manager_factory("map", data_dir=str(random.random())),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(context_fingerprint=True),
    )
    call_count = [0]

    def llm_handler(**kwargs) -> object:
        call_count[0] += 1
        return kwargs["messages"][0]["content"]

    def custom_update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> object:
        update_cache_func(llm_data)
        return llm_data

    convo_a = {
        "messages": [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a language."},
            {"role": "user", "content": "Tell me more"},
        ]
    }
    convo_b = {
        "messages": [
            {"role": "user", "content": "Tell me about Rust"},
            {"role": "assistant", "content": "Rust is a language."},
            {"role": "user", "content": "Tell me more"},
        ]
    }

    first = adapt(
        llm_handler,
        lambda data: data,
        custom_update_cache_callback,
        cache_obj=cache_obj,
        **convo_a,
    )
    second = adapt(
        llm_handler,
        lambda data: data,
        custom_update_cache_callback,
        cache_obj=cache_obj,
        **convo_b,
    )
    repeat = adapt(
        llm_handler,
        lambda data: data,
        custom_update_cache_callback,
        cache_obj=cache_obj,
        **convo_a,
    )

    assert first == "Tell me about Python"
    assert second == "Tell me about Rust"
    assert repeat == "Tell me about Python"
    assert call_count[0] == 2


def test_async_retrieval_namespace_prevents_cross_context_collisions() -> object:
    cache_obj = Cache()
    cache_obj.init(
        pre_embedding_func=get_prompt,
        embedding_func=lambda value, **_: value,
        data_manager=manager_factory("map", data_dir=str(random.random())),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(retrieval_namespace_fields=["retrieval_context"]),
    )
    call_count = [0]

    async def llm_handler(**kwargs) -> object:
        call_count[0] += 1
        return kwargs["retrieval_context"]["doc_id"]

    def custom_update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> object:
        update_cache_func(str(llm_data))
        return llm_data

    async def run_calls() -> object:
        first = await aadapt(
            llm_handler,
            lambda data: int(data),
            custom_update_cache_callback,
            cache_obj=cache_obj,
            prompt="What changed?",
            retrieval_context={"doc_id": 1},
        )
        second = await aadapt(
            llm_handler,
            lambda data: int(data),
            custom_update_cache_callback,
            cache_obj=cache_obj,
            prompt="What changed?",
            retrieval_context={"doc_id": 2},
        )
        repeat = await aadapt(
            llm_handler,
            lambda data: int(data),
            custom_update_cache_callback,
            cache_obj=cache_obj,
            prompt="What changed?",
            retrieval_context={"doc_id": 1},
        )
        return first, second, repeat

    first, second, repeat = asyncio.run(run_calls())

    assert first == 1
    assert second == 2
    assert repeat == 1
    assert call_count[0] == 2
