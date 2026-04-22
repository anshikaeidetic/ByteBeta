"""Provider-agnostic coding workload tests across supported adapter backends."""

import time

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
from byte.processor.model_router import clear_route_performance
from byte.processor.policy import clear_global_policy
from byte.processor.pre import last_content, normalized_last_content

ADAPTER_CASES = [
    (cache_openai, "openai", "gpt-4o-mini"),
    (cache_deepseek, "deepseek", "deepseek-chat"),
    (cache_anthropic, "anthropic", "claude-sonnet-4-20250514"),
    (cache_gemini, "gemini", "gemini-2.0-flash"),
    (cache_groq, "groq", "llama-3.3-70b-versatile"),
    (cache_openrouter, "openrouter", "meta-llama/llama-3.3-70b-instruct"),
    (cache_ollama, "ollama", "llama3.2"),
    (cache_mistral, "mistral", "mistral-small-latest"),
    (cache_cohere, "cohere", "command-r-plus"),
    (cache_bedrock, "bedrock", "anthropic.claude-3-5-sonnet"),
]


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
        "usage": {"prompt_tokens": 16, "completion_tokens": 8, "total_tokens": 24},
    }


@pytest.fixture(autouse=True)
def reset_adapter_state() -> object:
    clear_route_performance()
    clear_global_policy()
    adapters = [
        cache_openai,
        cache_deepseek,
        cache_anthropic,
        cache_gemini,
        cache_groq,
        cache_openrouter,
        cache_ollama,
        cache_mistral,
        cache_cohere,
        cache_bedrock,
    ]
    for adapter in adapters:
        adapter.ChatCompletion.llm = None
        adapter.ChatCompletion.cache_args = {}
    yield
    clear_route_performance()
    clear_global_policy()
    for adapter in adapters:
        adapter.ChatCompletion.llm = None
        adapter.ChatCompletion.cache_args = {}


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_code_fix_templates_hit_normalized_cache_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / provider),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            semantic_allowed_categories=["question_answer"],
        ),
    )
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("MUTABLE_DEFAULT", provider, model)

    adapter_module.ChatCompletion.llm = fake_llm

    first_prompt = (
        "Fix the bug in this Python function.\n"
        "File: src/cart.py:14\n"
        "Diagnostic: mutable default argument\n"
        "```python\n"
        "def add_item(item, items=[]):\n"
        "    items.append(item)\n"
        "    return items\n"
        "```\n"
        "Return exactly MUTABLE_DEFAULT and nothing else."
    )
    second_prompt = (
        "You are debugging the selected code.\n"
        "Path: app/cart.py\n"
        "Line 27\n"
        "Error: mutable default argument\n"
        "```python\n"
        "def add_item(item, items=[]):\n"
        "    items.append(item)\n"
        "    return items\n"
        "```\n"
        "Reply with exactly MUTABLE_DEFAULT and nothing else."
    )

    adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": first_prompt}],
        cache_obj=cache_obj,
    )
    hit = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": second_prompt}],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 1
    assert hit.get("byte") is True
    assert hit["usage"]["total_tokens"] == 0


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_code_test_templates_hit_normalized_cache_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-tests"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            semantic_allowed_categories=["question_answer"],
        ),
    )
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("PYTEST", provider, model)

    adapter_module.ChatCompletion.llm = fake_llm

    first_prompt = (
        "Write pytest tests for this helper.\n"
        "File: src/helpers.py\n"
        "```python\n"
        "def slugify(value):\n"
        "    return value.strip().lower().replace(' ', '-')\n"
        "```\n"
        "Return exactly PYTEST and nothing else."
    )
    second_prompt = (
        "Add unit tests using pytest for the selected function.\n"
        "Path: app/helpers.py:32\n"
        "```python\n"
        "def slugify(value):\n"
        "    return value.strip().lower().replace(' ', '-')\n"
        "```\n"
        "Reply with exactly PYTEST and nothing else."
    )

    adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": first_prompt}],
        cache_obj=cache_obj,
    )
    hit = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": second_prompt}],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 1
    assert hit.get("byte") is True
    assert hit["usage"]["total_tokens"] == 0


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_warm_data_respects_model_namespace_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    first_prompt = (
        "Fix the bug in this Python function.\n"
        "File: src/cart.py:14\n"
        "Diagnostic: mutable default argument\n"
        "```python\n"
        "def add_item(item, items=[]):\n"
        "    items.append(item)\n"
        "    return items\n"
        "```\n"
        "Return exactly MUTABLE_DEFAULT and nothing else."
    )
    second_prompt = (
        "You are debugging the selected code.\n"
        "Path: app/cart.py\n"
        "Line 27\n"
        "Error: mutable default argument\n"
        "```python\n"
        "def add_item(item, items=[]):\n"
        "    items.append(item)\n"
        "    return items\n"
        "```\n"
        "Reply with exactly MUTABLE_DEFAULT and nothing else."
    )
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-warm"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            model_namespace=True,
            semantic_allowed_categories=["question_answer"],
        ),
        warm_data=[{"question": first_prompt, "answer": "MUTABLE_DEFAULT", "model": model}],
    )
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("MUTABLE_DEFAULT", provider, model)

    adapter_module.ChatCompletion.llm = fake_llm

    hit = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": second_prompt}],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert hit.get("byte") is True
    assert hit["usage"]["total_tokens"] == 0


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_structured_answer_repair_and_cache_reuse_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-repair"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False),
    )
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("The correct label is MUTABLE_DEFAULT.", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    prompt = (
        "Fix the bug in this Python function.\n"
        "File: src/cart.py:14\n"
        "Diagnostic: mutable default argument\n"
        "```python\n"
        "def add_item(item, items=[]):\n"
        "    items.append(item)\n"
        "    return items\n"
        "```\n"
        "Return exactly MUTABLE_DEFAULT and nothing else."
    )

    first = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        cache_obj=cache_obj,
    )
    second = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        cache_obj=cache_obj,
    )

    assert first["choices"][0]["message"]["content"] == "MUTABLE_DEFAULT"
    assert second["choices"][0]["message"]["content"] == "MUTABLE_DEFAULT"
    assert call_count[0] == 1
    assert second.get("byte") is True


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_invalid_cheap_route_escalates_to_expensive_model_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-escalate"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            model_routing=True,
            routing_cheap_model="cheap-model",
            routing_expensive_model="expensive-model",
            routing_default_model="expensive-model",
        ),
    )
    seen_models = []

    def fake_llm(**kwargs) -> object:
        seen_models.append(kwargs["model"])
        if kwargs["model"] == "cheap-model":
            return _make_response("Not sure.", provider, kwargs["model"])
        return _make_response("POSITIVE", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    prompt = (
        "Classify the sentiment.\n"
        "Labels: POSITIVE, NEGATIVE, NEUTRAL\n"
        'Review: "I absolutely loved this movie."\n'
        "Answer with exactly one label."
    )

    adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        cache_obj=cache_obj,
    )

    assert seen_models == ["cheap-model", "expensive-model"]


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_coder_route_can_escalate_to_reasoning_after_verifier_rejection_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-orchestration"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            model_routing=True,
            routing_coder_model="coder-model",
            routing_reasoning_model="reasoning-model",
            routing_expensive_model="expensive-model",
            routing_default_model="cheap-model",
            routing_verifier_model="verifier-model",
        ),
    )
    seen_models = []
    verifier_calls = [0]

    def fake_llm(**kwargs) -> object:
        seen_models.append(kwargs["model"])
        if kwargs["model"] == "coder-model":
            return _make_response("MUTABLE_DEFAULT", provider, kwargs["model"])
        if kwargs["model"] == "reasoning-model":
            return _make_response("MUTABLE_DEFAULT", provider, kwargs["model"])
        if kwargs["model"] == "verifier-model":
            verifier_calls[0] += 1
            verdict = "REJECT" if verifier_calls[0] == 1 else "ACCEPT"
            return _make_response(verdict, provider, kwargs["model"])
        return _make_response("UNEXPECTED", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    prompt = (
        "Fix the bug in this Python function.\n"
        "File: src/cart.py:14\n"
        "Diagnostic: mutable default argument\n"
        "```python\n"
        "def add_item(item, items=[]):\n"
        "    items.append(item)\n"
        "    return items\n"
        "```\n"
        "Return exactly MUTABLE_DEFAULT and nothing else."
    )

    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        cache_obj=cache_obj,
    )

    assert response["choices"][0]["message"]["content"] == "MUTABLE_DEFAULT"
    assert seen_models == ["coder-model", "verifier-model", "reasoning-model", "verifier-model"]


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_internal_byte_route_hints_do_not_leak_to_provider_calls(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="exact",
        data_dir=str(tmp_path / f"{provider}-provider-safe"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False),
    )

    def fake_llm(**kwargs) -> object:
        assert "byte_route_preference" not in kwargs
        return _make_response("SAFE", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "This is a Byte benchmark safety check. Return exactly SAFE and nothing else.",
            }
        ],
        cache_obj=cache_obj,
        byte_route_preference="cheap",
    )

    assert response["choices"][0]["message"]["content"] == "SAFE"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_unique_output_guard_disables_answer_reuse_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-unique-guard"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False, unique_output_guard=True, context_only_unique_prompts=True
        ),
    )
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("STRESS_UNIQUE_0001", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    prompt = (
        "Single-token benchmark request 0001.\n"
        "Topic: release note draft 0001.\n"
        "Reply exactly STRESS_UNIQUE_0001 and nothing else."
    )

    first = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        cache_obj=cache_obj,
    )
    second = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        cache_obj=cache_obj,
    )

    assert first["choices"][0]["message"]["content"] == "STRESS_UNIQUE_0001"
    assert second["choices"][0]["message"]["content"] == "STRESS_UNIQUE_0001"
    if second.get("byte_reason") == "contract_shortcut":
        assert call_count[0] == 0
    else:
        assert second.get("byte") is not True
        assert call_count[0] == 2


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_exact_prompt_with_code_topic_word_does_not_clarify_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-exact-no-clarify"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False, planner_enabled=True, ambiguity_detection=True),
    )

    def fake_llm(**kwargs) -> object:
        return _make_response("STRESS_UNIQUE_0005", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Single-token benchmark request 0005.\n"
                    "Topic: code review follow-up 0005.\n"
                    "Reply exactly STRESS_UNIQUE_0005 and nothing else."
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert response.get("byte_reason") != "clarification_required"
    assert response["choices"][0]["message"]["content"] == "STRESS_UNIQUE_0005"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_docstring_output_contract_is_enforced_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-docstring-contract"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            model_routing=True,
            routing_cheap_model="cheap-model",
            routing_expensive_model="expensive-model",
            routing_default_model="cheap-model",
        ),
    )

    def fake_llm(**kwargs) -> object:
        system_messages = [
            str(message.get("content", "") or "")
            for message in (kwargs.get("messages") or [])
            if str(message.get("role", "") or "") == "system"
        ]
        assert any("DOCSTRING_READY" in message for message in system_messages)
        return _make_response(
            '"""Normalize the provided customer name."""', provider, kwargs["model"]
        )

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Add a docstring for this function.\n"
                    "File: src/cleanup.py\n"
                    "```python\n"
                    "def normalize_name(value):\n"
                    "    return value.strip().title()\n"
                    "```\n"
                    "Return exactly DOCSTRING_READY and nothing else."
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert response["choices"][0]["message"]["content"] == "DOCSTRING_READY"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_complexity_semantic_label_repair_applies_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-complexity-semantics"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False),
    )

    def fake_llm(**kwargs) -> object:
        return _make_response("O_N", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Walk me through what this selected code does.\n"
                    "Path: app/matrix.py\n"
                    "Line 11\n"
                    "```python\n"
                    "def pair_sum(matrix):\n"
                    "    total = 0\n"
                    "    for row in matrix:\n"
                    "        for value in row:\n"
                    "            total += value\n"
                    "    return total\n"
                    "```\n"
                    "Answer with exactly one complexity label from {O_1, O_N, O_N_SQUARED}."
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert response["choices"][0]["message"]["content"] == "O_N_SQUARED"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_framework_semantic_label_repair_applies_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-framework-semantics"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False),
    )

    def fake_llm(**kwargs) -> object:
        return _make_response("UNITTEST", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Write pytest tests for this helper.\n"
                    "File: src/helpers.py\n"
                    "```python\n"
                    "def slugify(value):\n"
                    "    return value.strip().lower().replace(' ', '-')\n"
                    "```\n"
                    "Return exactly one framework label from {PYTEST, UNITTEST, JEST}."
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert response["choices"][0]["message"]["content"] == "PYTEST"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_bug_semantic_label_repair_applies_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-bug-semantics"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False),
    )

    def fake_llm(**kwargs) -> object:
        return _make_response("NONE", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Fix the bug in this Python function.\n"
                    "File: src/unique_d.py\n"
                    "Diagnostic: broad exception clause\n"
                    "```python\n"
                    "def load_port(value):\n"
                    "    try:\n"
                    "        return int(value)\n"
                    "    except:\n"
                    "        return 8080\n"
                    "```\n"
                    "Return exactly one label from {MUTABLE_DEFAULT, OFF_BY_ONE, SYNTAX_ERROR, BROAD_EXCEPTION, NONE}."
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert response["choices"][0]["message"]["content"] == "BROAD_EXCEPTION"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_retrieval_context_partitions_cache_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-retrieval"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False),
    )
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response(f"answer-{call_count[0]}", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    prompt = "Summarize this repository health status in one sentence."

    first = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        byte_retrieval_context={"docs": ["repository health answer-1"]},
        cache_obj=cache_obj,
    )
    second = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        byte_retrieval_context={"docs": ["repository health answer-2"]},
        cache_obj=cache_obj,
    )
    third = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        byte_retrieval_context={"docs": ["repository health answer-1"]},
        cache_obj=cache_obj,
    )

    assert first["choices"][0]["message"]["content"] == "answer-1"
    assert second["choices"][0]["message"]["content"] == "answer-2"
    assert third["choices"][0]["message"]["content"] == "answer-1"
    assert call_count[0] == 2
    assert third.get("byte") is True


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_ambiguous_code_request_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-clarify"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False, planner_enabled=True, ambiguity_detection=True),
    )
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("should-not-run", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "Fix this bug for me."}],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert response.get("byte_reason") == "clarification_required"
    assert (
        "code" in response["choices"][0]["message"]["content"].lower()
        or "file" in response["choices"][0]["message"]["content"].lower()
    )


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_repo_summary_context_avoids_unnecessary_clarification_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-repo-context"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False, planner_enabled=True, ambiguity_detection=True),
    )
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("PATCH_READY", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "Fix the selected bug in the nearby file."}],
        byte_repo_summary="The repository contains a checkout flow and a cart service with a mutable default bug.",
        byte_changed_files=["src/cart.py"],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 1
    assert response.get("byte_reason") != "clarification_required"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_unverified_code_answers_do_not_reuse_cache_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-verified"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False, execution_memory=True, verified_reuse_for_coding=True
        ),
    )
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("MUTABLE_DEFAULT", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    prompt = (
        "Fix the bug in this Python function.\n"
        "File: src/cart.py:14\n"
        "Diagnostic: mutable default argument\n"
        "```python\n"
        "def add_item(item, items=[]):\n"
        "    items.append(item)\n"
        "    return items\n"
        "```\n"
        "Return exactly MUTABLE_DEFAULT and nothing else."
    )

    adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        byte_memory={"verification": {"verified": False}},
        cache_obj=cache_obj,
    )
    adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 2


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_verified_patch_reuse_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path / f"{provider}-patch"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False, planner_enabled=True, delta_generation=True),
    )
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("should-not-run", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    prompt = (
        "Apply the same fix to this nearby file.\n"
        "```python\n"
        "def add_item(item, items=[]):\n"
        "    items.append(item)\n"
        "    return items\n"
        "```\n"
        "Return the patch only."
    )
    patch = (
        "--- original\n"
        "+++ fixed\n"
        "@@\n"
        "-def add_item(item, items=[]):\n"
        "+def add_item(item, items=None):\n"
        "+    if items is None:\n"
        "+        items = []\n"
    )
    cache_obj.remember_execution_result(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        answer=patch,
        patch=patch,
        verification={"verified": True},
        repo_fingerprint="repo-1",
        model=model,
        provider=provider,
    )

    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        byte_repo_fingerprint="repo-1",
        byte_allow_patch_reuse=True,
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert response.get("byte_reason") == "verified_patch_reuse"
    assert "byte_suggested" in response["choices"][0]["message"]["content"]
