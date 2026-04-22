"""Pre-processing and context-compilation tests for prompt normalization flows."""

from types import SimpleNamespace

from byte.adapter.pipeline.context import (
    _compile_context_if_needed as adapter_compile_context_if_needed,
)
from byte.benchmarking.systems import _base_cache_config
from byte.config import Config
from byte.processor.optimization_memory import stable_digest
from byte.processor.pre import (
    all_content,
    canonicalize_text,
    compile_request_context,
    concat_all_queries,
    get_openai_moderation_input,
    get_prompt,
    last_content,
    last_content_without_prompt,
    nop,
    normalize_text,
    normalized_get_prompt,
    normalized_last_content,
)


def test_last_content() -> None:
    content = last_content({"messages": [{"content": "foo1"}, {"content": "foo2"}]})

    assert content == "foo2"


def test_last_content_serializes_multimodal_parts_stably() -> None:
    content = last_content(
        {
            "messages": [
                {"content": "foo1"},
                {
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,aGVsbG8=",
                                "name": "diagram.png",
                            },
                        },
                    ]
                },
            ]
        }
    )

    assert "text::Describe this image" in content
    assert "image::image/png::diagram.png::sha256:" in content


def test_normalized_last_content() -> None:
    content = normalized_last_content(
        {"messages": [{"content": "foo1"}, {"content": "  Hello,   WORLD!!  "}]}
    )
    assert content == "hello world"


def test_last_content_without_prompt() -> None:
    content = last_content_without_prompt({"messages": [{"content": "foo1"}, {"content": "foo2"}]})
    assert content == "foo2"

    content = last_content_without_prompt(
        {"messages": [{"content": "foo1"}, {"content": "foo2"}]}, prompts=None
    )
    assert content == "foo2"

    content = last_content_without_prompt(
        {"messages": [{"content": "foo1"}, {"content": "foo2"}]}, prompts=["foo"]
    )
    assert content == "2"


def test_all_content() -> None:
    content = all_content({"messages": [{"content": "foo1"}, {"content": "foo2"}]})

    assert content == "foo1\nfoo2"


def test_nop() -> None:
    content = nop({"str": "hello"})
    assert content == {"str": "hello"}


def test_get_prompt() -> None:
    content = get_prompt({"prompt": "foo"})
    assert content == "foo"


def test_normalized_get_prompt() -> None:
    content = normalized_get_prompt({"prompt": "  Cache-Key,   Please!! "})
    assert content == "cache key please"


def test_normalize_text() -> None:
    assert normalize_text("  Hi,\nTHERE!!  ") == "hi there"


def test_canonicalize_text_extracts_exact_answer_template() -> None:
    content = canonicalize_text("Reply with exactly TOKYO and nothing else.")
    assert content == "exact_answer::tokyo"


def test_canonicalize_text_handles_reordered_instruction_template() -> None:
    content = canonicalize_text(
        "Keep the answer to TOKYO. Byte benchmark request. Reply with exactly TOKYO and nothing else."
    )
    assert content == "exact_answer::tokyo"


def test_canonicalize_text_extracts_labeled_classification_template() -> None:
    content = canonicalize_text(
        'Classify the sentiment.\nLabels: POSITIVE, NEGATIVE, NEUTRAL\nReview: "I absolutely loved this movie."'
    )
    assert content.startswith("classify::review::negative|neutral|positive::")


def test_canonicalize_text_reuses_support_classification_variants() -> None:
    first = canonicalize_text(
        "You are triaging a support inbox.\n"
        "Labels: BILLING, TECHNICAL, SHIPPING\n"
        'Ticket: "My order has shown in transit for eight days and still has not reached me."\n'
        "Reply with exactly one label."
    )
    second = canonicalize_text(
        'Ticket: "My order has shown in transit for eight days and still has not reached me."\n'
        "Support categories: BILLING, TECHNICAL, SHIPPING\n"
        "Classify this request and answer with exactly one label."
    )
    assert first == second
    assert first.startswith("classify::ticket::billing|shipping|technical::")


def test_canonicalize_text_extracts_translation_template() -> None:
    content = canonicalize_text('Translate to Spanish: "The cache is warm."')
    assert content.startswith("translate::spanish::")


def test_canonicalize_text_does_not_treat_output_shape_as_exact_answer() -> None:
    content = canonicalize_text(
        'Classify the sentiment.\nLabels: POSITIVE, NEGATIVE, NEUTRAL\nReview: "I absolutely loved this movie."\nAnswer with exactly one label.'
    )
    assert content.startswith("classify::review::negative|neutral|positive::")


def test_canonicalize_text_extracts_summarization_template() -> None:
    content = canonicalize_text(
        'Summarize the following article in one sentence. Article: "Byte Cache reduces repeated LLM calls for support teams."'
    )
    assert content.startswith("summarize::one_sentence::")


def test_canonicalize_text_reuses_capital_city_question_variants() -> None:
    first = canonicalize_text("What is the capital of France?")
    second = canonicalize_text("Which city is the capital of France?")

    assert first == second
    assert first == "qa_fact::capital::france"


def test_canonicalize_text_extracts_structured_extraction_template() -> None:
    content = canonicalize_text(
        'Extract the fields. Fields: name, city\nText: "Name: Alice. City: Paris."\nReturn JSON only.'
    )
    assert content.startswith("extract::json::city|name::")


def test_canonicalize_text_reuses_extraction_fields_in_any_order() -> None:
    first = canonicalize_text(
        'Extract the fields. Fields: name, city\nText: "Name: Alice. City: Paris."\nReturn JSON only.'
    )
    second = canonicalize_text(
        'Return JSON with keys: city, name\nText: "Name: Alice. City: Paris."'
    )
    assert first == second


def test_canonicalize_text_reuses_code_fix_templates_across_paths_and_lines() -> None:
    first = canonicalize_text(
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
    second = canonicalize_text(
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
    assert first == second
    assert first.startswith("code_fix::python::mutable_default::")


def test_canonicalize_text_reuses_code_test_templates() -> None:
    first = canonicalize_text(
        "Write pytest tests for this helper.\n"
        "File: src/helpers.py\n"
        "```python\n"
        "def slugify(value):\n"
        "    return value.strip().lower().replace(' ', '-')\n"
        "```\n"
    )
    second = canonicalize_text(
        "Add unit tests using pytest for the selected function.\n"
        "Path: app/helpers.py:32\n"
        "```python\n"
        "def slugify(value):\n"
        "    return value.strip().lower().replace(' ', '-')\n"
        "```\n"
    )
    assert first == second
    assert first.startswith("code_tests::python::pytest::")


def test_canonicalize_text_extracts_code_explanation_template() -> None:
    content = canonicalize_text(
        "Explain this function in one sentence.\n"
        "File: src/math_utils.py\n"
        "```python\n"
        "def total(values):\n"
        "    result = 0\n"
        "    for value in values:\n"
        "        result += value\n"
        "    return result\n"
        "```\n"
    )
    assert content.startswith("code_explain::python::one_sentence::")


def test_canonicalize_text_reuses_complexity_explain_templates() -> None:
    first = canonicalize_text(
        "Explain this function in one sentence.\n"
        "File: src/math_utils.py\n"
        "```python\n"
        "def total(values):\n"
        "    result = 0\n"
        "    for value in values:\n"
        "        result += value\n"
        "    return result\n"
        "```\n"
        "Return exactly one complexity label from {O_1, O_N, O_N_SQUARED}."
    )
    second = canonicalize_text(
        "Walk me through what this selected code does.\n"
        "Path: app/math_utils.py\n"
        "Line 11\n"
        "```python\n"
        "def total(values):\n"
        "    result = 0\n"
        "    for value in values:\n"
        "        result += value\n"
        "    return result\n"
        "```\n"
        "Answer with exactly one complexity label from {O_1, O_N, O_N_SQUARED}."
    )
    assert first == second
    assert first.startswith("code_explain::python::complexity::")


def test_canonicalize_text_keeps_docstring_requests_distinct_from_cleanup_paths() -> None:
    content = canonicalize_text(
        "Add a docstring for this function.\n"
        "File: src/cleanup.py\n"
        "```python\n"
        "def normalize_name(value):\n"
        "    return value.strip().title()\n"
        "```\n"
        "Return exactly DOCSTRING_READY and nothing else."
    )

    assert content.startswith("code_doc::python::")


def test_compile_request_context_prioritizes_relevant_retrieval_items() -> None:
    compiled, stats = compile_request_context(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Answer the refund question using the support docs.",
                }
            ],
            "byte_retrieval_context": [
                {"title": "Password Reset", "snippet": "Reset passwords from the settings page."},
                {
                    "title": "Refund Policy",
                    "snippet": "Refund duplicate subscription charges in 5 business days.",
                },
                {"title": "Shipping FAQ", "snippet": "Standard shipping takes 5-7 business days."},
            ],
        },
        max_chars=1200,
        relevance_top_k=1,
    )

    last_message = compiled["messages"][-1]["content"].lower()

    assert "refund policy" in last_message
    assert "password reset" not in last_message
    assert stats["relevance_pruned_items"] == 2


def test_compile_request_context_prunes_negative_context_and_adds_sketch() -> None:
    refund_item = {
        "title": "Refund Policy",
        "snippet": "Refund duplicate subscription charges in 5 business days.",
    }
    password_item = {
        "title": "Password Reset",
        "snippet": "Reset passwords from the settings page.",
    }
    compiled, stats = compile_request_context(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Answer the refund question using the support docs.",
                }
            ],
            "byte_retrieval_context": [password_item, refund_item],
        },
        max_chars=1200,
        relevance_top_k=2,
        negative_context_digests={"retrieval_context": [stable_digest(password_item)]},
        context_sketches=True,
    )

    last_message = compiled["messages"][-1]["content"].lower()

    assert "retrieval context sketch" in last_message
    assert "refund policy" in last_message
    assert "password reset" not in last_message
    assert stats["negative_pruned_items"] == 1


def test_compile_request_context_can_prefix_aux_context_for_native_prompt_caching() -> None:
    compiled, stats = compile_request_context(
        {
            "messages": [
                {"role": "system", "content": "You are a grounded assistant."},
                {
                    "role": "user",
                    "content": "Return exactly the invoice identifier from the retrieval context and nothing else.",
                },
            ],
            "byte_retrieval_context": [
                {
                    "title": "Invoice note",
                    "snippet": "Invoice INV-4101 belongs to owner FINANCE and the open amount is $4,237.",
                },
                {
                    "title": "Schedule note",
                    "snippet": "The follow-up date for INV-4101 is 2026-06-11.",
                },
            ],
        },
        max_chars=1200,
        relevance_top_k=1,
        prefix_messages=True,
    )

    assert stats["compiled_aux_contexts"] >= 1
    assert compiled["messages"][1]["role"] == "system"
    assert "byte compiled context" in compiled["messages"][1]["content"].lower()
    assert "invoice inv-4101" in compiled["messages"][1]["content"].lower()
    assert (
        compiled["messages"][-1]["content"]
        == "Return exactly the invoice identifier from the retrieval context and nothing else."
    )


def test_compile_request_context_enforces_total_aux_budget_and_cross_note_dedupe() -> None:
    refund_item = {
        "title": "Refund Policy",
        "snippet": "Duplicate subscription charges are refunded in 5 business days. Reference ID REF-22.",
    }
    compiled, stats = compile_request_context(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Answer the refund question using the available support and document context.",
                }
            ],
            "byte_retrieval_context": [
                refund_item,
                {"title": "Password Reset", "snippet": "Reset passwords from the settings page."},
            ],
            "byte_support_articles": [
                refund_item,
                {"title": "Shipping FAQ", "snippet": "Standard shipping takes 5-7 business days."},
            ],
            "byte_document_context": [
                {
                    "title": "Long Handbook",
                    "snippet": (
                        "Company handbook section on workspace badges. " * 20
                        + "This is unrelated to refunds."
                    ),
                }
            ],
        },
        max_chars=520,
        relevance_top_k=2,
        focus_distillation=True,
        total_aux_budget_ratio=0.32,
        cross_note_dedupe=True,
    )

    last_message = compiled["messages"][-1]["content"].lower()

    assert "refund policy" in last_message
    assert "5 business days" in last_message
    assert "password reset" not in last_message
    assert stats["focused_distillation_hits"] >= 1
    assert stats["aux_budget_pruned_notes"] >= 1 or stats["aux_budget_trimmed_chars"] > 0


def test_compile_request_context_cross_note_dedupes_duplicate_aux_blocks() -> None:
    refund_item = {
        "title": "Refund Policy",
        "snippet": "Duplicate subscription charges are refunded in 5 business days. Reference ID REF-22.",
    }
    compiled, stats = compile_request_context(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Answer the refund question using the available support and retrieval context.",
                }
            ],
            "byte_retrieval_context": [refund_item],
            "byte_support_articles": [refund_item],
        },
        max_chars=1200,
        focus_distillation=True,
        cross_note_dedupe=True,
    )

    last_message = compiled["messages"][-1]["content"].lower()

    assert "refund policy" in last_message
    assert stats["cross_note_deduped_notes"] >= 1


def test_compile_request_context_applies_prompt_distillation_before_aux_compilation() -> None:
    compiled, stats = compile_request_context(
        {
            "messages": [
                {"role": "system", "content": "You are Byte."},
                {
                    "role": "user",
                    "content": "Return exactly the invoice identifier from the retrieval context and nothing else.",
                },
            ],
            "byte_retrieval_context": [
                {
                    "title": "Invoice note",
                    "snippet": (
                        "Invoice INV-9100 belongs to queue-BILLING-04 and due_date is 2026-09-14. "
                        + " ".join(
                            f"noise block {index} repeats redundant warehouse replication metadata."
                            for index in range(120)
                        )
                    ),
                }
            ],
        },
        max_chars=1400,
        relevance_top_k=1,
        prompt_distillation_mode="guarded",
        prompt_distillation_backend="hybrid_local",
        prompt_distillation_budget_ratio=0.45,
        prompt_distillation_min_chars=600,
        prompt_distillation_retrieval_mode="hybrid",
    )

    prompt_distillation = stats["prompt_distillation"]

    assert prompt_distillation["applied"] is True
    assert prompt_distillation["compression_ratio"] > 0.3
    assert prompt_distillation["retrieval_compression_ratio"] > 0.3
    assert "invoice inv-9100" in compiled["messages"][-1]["content"].lower()


def test_adapter_context_compilation_is_idempotent_across_cache_tiers() -> None:
    config = _base_cache_config("test-idempotent")
    chat_cache = SimpleNamespace(
        config=config,
        prompt_piece_store=None,
        prompt_module_registry=None,
        artifact_memory_store=None,
        session_delta_store=None,
        memory_scope="",
        failure_memory_hint=lambda *args, **kwargs: {},
    )
    context = {}
    request = {
        "messages": [
            {"role": "system", "content": "You are Byte."},
            {
                "role": "user",
                "content": "Return exactly the invoice identifier from the retrieval context and nothing else.",
            },
        ],
        "byte_retrieval_context": [
            {
                "title": "Invoice note",
                "snippet": (
                    "Invoice INV-9100 belongs to queue-BILLING-04 and due_date is 2026-09-14. "
                    + " ".join(
                        f"noise block {index} repeats redundant warehouse replication metadata."
                        for index in range(120)
                    )
                ),
            }
        ],
    }

    first = adapter_compile_context_if_needed(chat_cache, dict(request), context)
    first_meta = dict(context["_byte_prompt_distillation"])

    second = adapter_compile_context_if_needed(chat_cache, dict(first), context)
    second_meta = dict(context["_byte_prompt_distillation"])

    assert first == second
    assert first_meta == second_meta


def test_get_openai_moderation_input() -> None:
    content = get_openai_moderation_input({"input": ["hello", "world"]})
    assert content == "['hello', 'world']"


def test_get_messages_last_content() -> None:
    content = last_content({"messages": [{"content": "foo1"}, {"content": "foo2"}]})
    assert content == "foo2"


def test_concat_all_queries() -> None:
    config = Config()
    config.context_len = 2
    content = concat_all_queries(
        {
            "messages": [
                {"role": "system", "content": "foo1"},
                {"role": "user", "content": "foo2"},
                {"role": "assistant", "content": "foo3"},
                {"role": "user", "content": "foo4"},
                {"role": "assistant", "content": "foo5"},
                {"role": "user", "content": "foo6"},
            ]
        },
        cache_config=config,
    )
    assert content == "USER: foo4\nUSER: foo6"


if __name__ == "__main__":
    test_concat_all_queries()
