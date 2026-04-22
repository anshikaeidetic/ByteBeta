"""Provider-agnostic reasoning reuse tests across supported adapter backends."""

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
    for adapter in adapters:
        adapter.ChatCompletion.llm = None
        adapter.ChatCompletion.cache_args = {}


def _init_reasoning_cache(tmp_path, provider, *, reasoning_reuse=True) -> object:
    cache_obj = Cache()
    init_cache(
        mode="hybrid",
        data_dir=str(tmp_path / provider),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            reasoning_reuse=reasoning_reuse,
            coding_reasoning_shortcut=True,
            reasoning_memory=True,
            reasoning_repair=True,
            ambiguity_detection=True,
        ),
    )
    return cache_obj


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_profit_margin_reasoning_shortcut_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-margin")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("30%", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "A company sells a product for $120.\n"
                    "Production cost = $60\n"
                    "Marketing cost = $20\n"
                    "Shipping cost = $10\n"
                    "Calculate profit margin percentage."
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert response.get("byte_reason") == "deterministic_reasoning"
    assert response["choices"][0]["message"]["content"] == "25%"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_exact_contract_shortcut_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-exact-contract")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("WRONG", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Unique benchmark request 001. Reply exactly UNIQUE_001 and nothing else.",
            }
        ],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert response.get("byte_reason") == "contract_shortcut"
    assert response["choices"][0]["message"]["content"] == "UNIQUE_001"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_grounded_retrieval_shortcut_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-grounded-rag")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("WRONG", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Return exactly the invoice identifier from the retrieval context and nothing else.",
            }
        ],
        byte_retrieval_context=[
            {
                "title": "Invoice note",
                "snippet": "Invoice INV-4101 belongs to owner FINANCE and the open amount is $4,237.",
            },
            {"title": "Schedule note", "snippet": "The follow-up date for INV-4101 is 2026-06-11."},
        ],
        byte_document_context=[
            {
                "title": "Invoice packet INV-4101",
                "snippet": "Invoice INV-4101 amount due $4,237, due date 2026-06-11, owner FINANCE.",
            },
        ],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert response.get("byte_reason") == "grounded_context_shortcut"
    assert response["choices"][0]["message"]["content"] == "INV-4101"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_grounded_long_context_shortcut_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-grounded-long")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("ARCH_01", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Return exactly the validated primary service identifier and nothing else.",
            }
        ],
        byte_repo_summary={
            "repo": "deepseek-architecture-01",
            "services": ["svc-01", "fallback-01"],
            "queue": "queue-01",
            "policy_label": "ARCH_01",
        },
        byte_document_context=[
            {
                "title": "Architecture overview 01",
                "snippet": "The distributed system uses API gateway -> svc-01 -> workers -> queue-01. Fallback traffic routes to fallback-01.",
            },
            {
                "title": "Reliability note 01",
                "snippet": "Policy label ARCH_01 identifies the architecture packet for svc-01.",
            },
        ],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert response.get("byte_reason") == "grounded_context_shortcut"
    assert response["choices"][0]["message"]["content"] == "svc-01"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_grounded_workflow_shortcut_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-grounded-workflow")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("ESCALATE_ENGINEERING", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Use the workflow policy context to decide the final action. Return exactly the prescribed action label and nothing else.",
            }
        ],
        byte_support_articles=[
            {
                "title": "Workflow policy 01A",
                "snippet": "When the correct owner is billing, the final action label should be REFUND_APPROVE.",
            },
            {
                "title": "Workflow policy 01B",
                "snippet": "Case summary 01: preserve deterministic policy labels and avoid extra prose.",
            },
        ],
        byte_document_context=[
            {
                "title": "Workflow packet 01",
                "snippet": "Escalation owner billing, prescribed action REFUND_APPROVE.",
            },
        ],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert response.get("byte_reason") == "grounded_context_shortcut"
    assert response["choices"][0]["message"]["content"] == "REFUND_APPROVE"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_code_fix_shortcut_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-code-fix")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("NONE", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are debugging the selected code.\n"
                    "Path: app/parser.py\n"
                    "Line 27\n"
                    "Error: invalid syntax\n"
                    "```python\n"
                    "def parse(value)\n"
                    "    return value.strip()\n"
                    "```\n"
                    "Reply with exactly one label from {MUTABLE_DEFAULT, OFF_BY_ONE, SYNTAX_ERROR, BROAD_EXCEPTION, NONE}."
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert response.get("byte_reason") == "coding_analysis_shortcut"
    assert response["choices"][0]["message"]["content"] == "SYNTAX_ERROR"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_code_explanation_shortcut_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-code-explain")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
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

    assert call_count[0] == 0
    assert response.get("byte_reason") == "coding_analysis_shortcut"
    assert response["choices"][0]["message"]["content"] == "O_N_SQUARED"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_test_generation_shortcut_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-code-tests")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
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

    assert call_count[0] == 0
    assert response.get("byte_reason") == "coding_analysis_shortcut"
    assert response["choices"][0]["message"]["content"] == "PYTEST"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_coding_exact_contract_shortcut_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-code-contract")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("WRONG", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Refactor this function for readability, compare tradeoffs, preserve behavior, "
                    "and after your analysis reply with exactly BYTE_CODE_REFACTOR_B and nothing else.\n"
                    "File: src/orders.py:48\n"
                    "```python\n"
                    "def build_message(name, total):\n"
                    "    return 'Customer ' + name + ' owes ' + str(total)\n"
                    "```"
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert response.get("byte_reason") == "coding_contract_shortcut"
    assert response["choices"][0]["message"]["content"] == "BYTE_CODE_REFACTOR_B"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_refund_policy_reasoning_shortcut_short_circuits_llm_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-policy")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("ESCALATE", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Customer purchased item A for $200.\n"
                    "Policy: Refunds allowed within 14 days.\n"
                    "Customer requested refund on day 5.\n"
                    "Return final action label.\n"
                    "Labels: REFUND_APPROVE, REFUND_DENY, ESCALATE"
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 0
    assert response.get("byte_reason") == "deterministic_reasoning"
    assert response["choices"][0]["message"]["content"] == "REFUND_APPROVE"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_capital_queries_reuse_reasoning_memory_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-capital")
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("Paris", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm

    first = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        cache_obj=cache_obj,
    )
    second = adapter_module.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "Which city is the capital of France?"}],
        cache_obj=cache_obj,
    )

    assert first["choices"][0]["message"]["content"] == "Paris"
    assert call_count[0] == 1
    assert second.get("byte_reason") == "reasoning_memory_reuse"
    assert second["choices"][0]["message"]["content"] == "Paris"


@pytest.mark.parametrize("adapter_module,provider,model", ADAPTER_CASES)
def test_wrong_profit_margin_answer_is_repaired_after_live_call_for_all_adapters(
    tmp_path, adapter_module, provider, model
) -> object:
    cache_obj = _init_reasoning_cache(tmp_path, f"{provider}-margin-repair", reasoning_reuse=False)
    call_count = [0]

    def fake_llm(**kwargs) -> object:
        call_count[0] += 1
        return _make_response("The margin is 30%.", provider, kwargs["model"])

    adapter_module.ChatCompletion.llm = fake_llm
    response = adapter_module.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "A company sells a product for $120.\n"
                    "Production cost = $60\n"
                    "Marketing cost = $20\n"
                    "Shipping cost = $10\n"
                    "Calculate profit margin percentage."
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert call_count[0] == 1
    assert response["choices"][0]["message"]["content"] == "25%"
