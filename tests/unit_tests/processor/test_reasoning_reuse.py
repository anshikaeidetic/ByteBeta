from byte import Config
from byte.processor.reasoning_reuse import (
    ReasoningMemoryStore,
    assess_reasoning_answer,
    capital_query_key,
    derive_reasoning_memory_record,
    resolve_reasoning_shortcut,
)


def test_reasoning_shortcut_solves_profit_margin_without_model() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
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
            ]
        }
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "deterministic_reasoning"
    assert shortcut.answer == "25%"


def test_reasoning_shortcut_executes_exact_contract_without_model() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Unique benchmark request 001. Reply exactly UNIQUE_001 and nothing else.",
                }
            ]
        }
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "contract_shortcut"
    assert shortcut.answer == "UNIQUE_001"


def test_reasoning_memory_store_tracks_compression_and_related_lookup() -> None:
    store = ReasoningMemoryStore(max_entries=8, codec_name="qjl", bits=8)
    store.remember(
        kind="knowledge_fact",
        key=capital_query_key("What is the capital of France?"),
        answer="Paris",
        verified=True,
    )

    exact = store.lookup(
        key=capital_query_key("What is the capital of France?"),
        kind="knowledge_fact",
        verified_only=True,
    )
    related = store.lookup_related(
        query_text="capital city of france",
        kind="knowledge_fact",
        verified_only=True,
        min_score=0.1,
    )

    assert exact is not None
    assert exact["compression"]["codec"] == "qjl"
    assert related
    assert related[0]["answer"] == "Paris"
    assert store.stats()["compressed_entries"] >= 1


def test_reasoning_shortcut_executes_exact_contract_even_with_attached_aux_context() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Review the shared support and repo context, then reply exactly SCX0101 and nothing else.",
                }
            ]
        },
        context_hints={
            "_byte_raw_aux_context": {
                "byte_repo_summary": {"repo": "sample-repo"},
                "byte_support_articles": [
                    {
                        "title": "Support article",
                        "snippet": "Duplicate charges are refunded in 5 business days.",
                    }
                ],
            }
        },
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "contract_shortcut"
    assert shortcut.answer == "SCX0101"


def test_reasoning_shortcut_extracts_grounded_invoice_identifier_from_aux_context() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Return exactly the invoice identifier from the retrieval context and nothing else.",
                }
            ]
        },
        context_hints={
            "_byte_raw_aux_context": {
                "byte_retrieval_context": [
                    {
                        "title": "Invoice note",
                        "snippet": "Invoice INV-4101 belongs to owner FINANCE and the open amount is $4,237.",
                    },
                    {
                        "title": "Schedule note",
                        "snippet": "The follow-up date for INV-4101 is 2026-06-11.",
                    },
                ]
            }
        },
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "grounded_context_shortcut"
    assert shortcut.answer == "INV-4101"


def test_reasoning_shortcut_extracts_grounded_cause_label_from_aux_context() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Return exactly the incident root-cause label from the retrieval context and nothing else.",
                }
            ]
        },
        context_hints={
            "_byte_raw_aux_context": {
                "byte_retrieval_context": [
                    {
                        "title": "Incident note",
                        "snippet": "Incident root cause label for case 01 is WEBHOOK_BACKLOG.",
                    },
                ]
            }
        },
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "grounded_context_shortcut"
    assert shortcut.answer == "WEBHOOK_BACKLOG"


def test_reasoning_shortcut_extracts_primary_service_from_repo_context() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Return exactly the validated primary service identifier and nothing else.",
                }
            ]
        },
        context_hints={
            "_byte_raw_aux_context": {
                "byte_repo_summary": {
                    "repo": "deepseek-architecture-01",
                    "services": ["svc-01", "fallback-01"],
                    "queue": "queue-01",
                    "policy_label": "ARCH_01",
                }
            }
        },
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "grounded_context_shortcut"
    assert shortcut.answer == "svc-01"


def test_reasoning_shortcut_extracts_action_label_from_support_context() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Use the workflow policy context to decide the final action. Return exactly the prescribed action label and nothing else.",
                }
            ]
        },
        context_hints={
            "_byte_raw_aux_context": {
                "byte_support_articles": [
                    {
                        "title": "Workflow policy 01A",
                        "snippet": "When the correct owner is billing, the final action label should be REFUND_APPROVE.",
                    },
                ],
                "byte_document_context": [
                    {
                        "title": "Workflow packet 01",
                        "snippet": "Escalation owner billing, prescribed action REFUND_APPROVE.",
                    },
                ],
            }
        },
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "grounded_context_shortcut"
    assert shortcut.answer == "REFUND_APPROVE"


def test_reasoning_shortcut_infers_bug_label_from_code_prompt() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
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
            ]
        },
        config=Config(coding_reasoning_shortcut=True),
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "coding_analysis_shortcut"
    assert shortcut.answer == "SYNTAX_ERROR"


def test_reasoning_shortcut_infers_complexity_label_from_code_prompt() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
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
            ]
        },
        config=Config(coding_reasoning_shortcut=True),
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "coding_analysis_shortcut"
    assert shortcut.answer == "O_N_SQUARED"


def test_reasoning_shortcut_infers_framework_label_from_code_prompt() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
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
            ]
        },
        config=Config(coding_reasoning_shortcut=True),
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "coding_analysis_shortcut"
    assert shortcut.answer == "PYTEST"


def test_reasoning_shortcut_uses_coding_exact_contract_for_synthetic_tokens() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
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
            ]
        },
        config=Config(coding_reasoning_shortcut=True),
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "coding_contract_shortcut"
    assert shortcut.answer == "BYTE_CODE_REFACTOR_B"


def test_reasoning_shortcut_reuses_capital_answer_from_memory() -> None:
    store = ReasoningMemoryStore()
    key = capital_query_key("What is the capital of France?")
    store.remember(
        kind="capital_city",
        key=key,
        answer="Paris",
        verified=True,
        metadata={"reason": "seed"},
    )

    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
                {"role": "user", "content": "Which city is the capital of France?"},
            ]
        },
        store=store,
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "reasoning_memory_reuse"
    assert shortcut.answer == "Paris"


def test_reasoning_shortcut_uses_curated_brief_capital_reference() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Japan? Return only the city name.",
                },
            ]
        },
        store=ReasoningMemoryStore(),
        config=Config(),
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "deterministic_reasoning"
    assert shortcut.answer == "Tokyo"


def test_reasoning_shortcut_uses_grounded_refund_policy_from_aux_context() -> None:
    shortcut = resolve_reasoning_shortcut(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Use the policy corpus and return only the final refund action label from {REFUND_APPROVE, REFUND_DENY, ESCALATE}.",
                },
            ]
        },
        store=ReasoningMemoryStore(),
        config=Config(),
        context_hints={
            "_byte_raw_aux_context": {
                "byte_support_articles": [
                    {
                        "title": "Refund Edge Case",
                        "snippet": "Refunds allowed within 14 days. Request on day 13. Labels: REFUND_APPROVE, REFUND_DENY, ESCALATE.",
                    }
                ]
            }
        },
    )

    assert shortcut is not None
    assert shortcut.byte_reason == "grounded_context_shortcut"
    assert shortcut.answer == "REFUND_APPROVE"


def test_parameter_sensitive_reuse_requires_prompt_diversity_promotion() -> None:
    store = ReasoningMemoryStore(max_entries=8, codec_name="disabled", bits=8)
    request = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Price = 120. Production cost = 60. Marketing cost = 20. "
                    "Shipping cost = 10. Return only the profit margin percentage."
                ),
            }
        ]
    }
    key = derive_reasoning_memory_record(request, "25%")["key"]
    store.remember(
        kind="profit_margin",
        key=key,
        answer="25%",
        verified=True,
        metadata={
            "prompt_signature": "sig-a",
            "promotion_required": True,
            "promotion_group": key,
            "promotion_state": "near_threshold_shadow",
        },
    )

    shortcut = resolve_reasoning_shortcut(request, store=store, config=Config())

    assert shortcut is not None
    assert shortcut.byte_reason == "deterministic_reasoning"

    store.remember(
        kind="profit_margin",
        key=key,
        answer="25%",
        verified=True,
        metadata={
            "prompt_signature": "sig-b",
            "promotion_required": True,
            "promotion_group": key,
            "promotion_state": "near_threshold_shadow",
        },
    )

    promoted = resolve_reasoning_shortcut(request, store=store, config=Config())

    assert promoted is not None
    assert promoted.byte_reason == "reasoning_memory_reuse"
    assert promoted.promotion_state == "dynamic_verified"


def test_assess_reasoning_answer_repairs_refund_policy_rule() -> None:
    assessment = assess_reasoning_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Customer purchased item A for $200.\n"
                        "Policy: Refunds allowed within 14 days.\n"
                        "Customer requested refund on day 20.\n"
                        "Return final action label.\n"
                        "Labels: REFUND_APPROVE, REFUND_DENY, ESCALATE"
                    ),
                }
            ]
        },
        "REFUND_APPROVE",
    )

    assert assessment is not None
    assert assessment["repaired_answer"] == "REFUND_DENY"
    assert assessment["constraint"] == "label_set"


def test_derive_reasoning_memory_record_keeps_short_capital_answer() -> None:
    record = derive_reasoning_memory_record(
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ]
        },
        "Paris",
    )

    assert record is not None
    assert record["kind"] == "capital_city"
    assert record["answer"] == "Paris"
