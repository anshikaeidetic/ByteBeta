from byte.processor.quality import QualityScorer


def test_quality_scorer_accepts_csv_extraction_when_all_fields_present() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Extract the fields.\n"
                        "Fields: name, city\n"
                        'Text: "Name: Alice. City: Paris."\n'
                        "Return CSV only."
                    ),
                }
            ]
        },
        "name,city\nAlice,Paris\n",
    )

    assert assessment.accepted is True
    assert assessment.constraint == "csv"
    assert assessment.score >= 0.9


def test_quality_scorer_rejects_structured_answer_missing_requested_fields() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Extract the fields.\n"
                        "Fields: name, city\n"
                        'Text: "Name: Alice. City: Paris."\n'
                        "Return JSON only."
                    ),
                }
            ]
        },
        '{"name":"Alice"}',
    )

    assert assessment.accepted is False
    assert assessment.constraint == "json"
    assert assessment.reason == "json_missing_fields"


def test_quality_scorer_penalizes_one_sentence_summary_that_is_not_one_sentence() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        'Summarize the following article in one sentence. Article: "Byte reduces repeated LLM calls."'
                    ),
                }
            ]
        },
        "Byte reduces repeated LLM calls. It also lowers latency for repeated workflows.",
    )

    assert assessment.accepted is True
    assert assessment.constraint == "summary_style"
    assert assessment.score < 0.78


def test_quality_scorer_uses_slot_aware_label_matching() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        'Ticket: "The app crashes whenever I click export."\n'
                        "Classify this request and answer with exactly one label from {BILLING, TECHNICAL, SHIPPING}."
                    ),
                }
            ]
        },
        "TECHNICAL",
    )

    assert assessment.accepted is True
    assert assessment.constraint == "label_set"
    assert assessment.repaired_answer == "TECHNICAL"


def test_quality_scorer_maps_common_complexity_aliases_to_canonical_labels() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
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
                    ),
                }
            ]
        },
        "This runs in O(n) time.",
    )

    assert assessment.accepted is True
    assert assessment.constraint == "label_set"
    assert assessment.repaired_answer == "O_N"


def test_quality_scorer_repairs_docstring_contract_from_docstring_output() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
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
            ]
        },
        '"""Normalize the provided customer name."""',
    )

    assert assessment.accepted is True
    assert assessment.constraint == "exact_token"
    assert assessment.repaired_answer == "DOCSTRING_READY"


def test_quality_scorer_repairs_nested_loop_complexity_label_from_request_semantics() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
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
        "O_N",
    )

    assert assessment.accepted is True
    assert assessment.constraint == "label_set"
    assert assessment.repaired_answer == "O_N_SQUARED"
    assert assessment.reason == "label_semantics_repaired"


def test_quality_scorer_repairs_framework_label_from_request_semantics() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
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
        "UNITTEST",
    )

    assert assessment.accepted is True
    assert assessment.constraint == "label_set"
    assert assessment.repaired_answer == "PYTEST"
    assert assessment.reason == "label_semantics_repaired"


def test_quality_scorer_repairs_bug_label_from_request_semantics() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
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
            ]
        },
        "NONE",
    )

    assert assessment.accepted is True
    assert assessment.constraint == "label_set"
    assert assessment.repaired_answer == "BROAD_EXCEPTION"
    assert assessment.reason == "label_semantics_repaired"


def test_quality_scorer_rejects_grounded_extraction_with_unsupported_values() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": ("Extract the fields.\nFields: name, city\nReturn JSON only."),
                }
            ]
        },
        '{"name":"Alice","city":"Berlin"}',
        context_hints={
            "_byte_raw_aux_context": {
                "byte_document_context": [
                    {
                        "text": "Customer record: Name: Alice. City: Paris.",
                    }
                ]
            }
        },
    )

    assert assessment.accepted is False
    assert assessment.constraint == "json"
    assert assessment.reason == "json_unsupported_values"


def test_quality_scorer_repairs_profit_margin_to_deterministic_answer() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
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
        },
        "30%",
    )

    assert assessment.accepted is True
    assert assessment.constraint == "numeric_answer"
    assert assessment.repaired_answer == "25%"
    assert assessment.reason == "deterministic_reasoning_repaired"


def test_quality_scorer_repairs_refund_policy_label_from_explicit_rule() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
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

    assert assessment.accepted is True
    assert assessment.constraint == "label_set"
    assert assessment.repaired_answer == "REFUND_DENY"
    assert assessment.reason == "deterministic_reasoning_repaired"


def test_quality_scorer_repairs_curated_policy_label_family() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Signal score 9. Radius external. Manual override no. "
                        "Labels: ALLOW, REVIEW, BLOCK. Apply the rule exactly and return only the label."
                    ),
                }
            ]
        },
        "ALLOW",
    )

    assert assessment.accepted is True
    assert assessment.constraint == "label_set"
    assert assessment.repaired_answer == "BLOCK"


def test_quality_scorer_rejects_grounded_freeform_answer_without_support() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "According to the support article, what is the refund window?",
                }
            ]
        },
        "30 days",
        context_hints={
            "_byte_raw_aux_context": {
                "byte_support_articles": [
                    {
                        "title": "Refund Policy",
                        "snippet": "Duplicate subscription refunds are processed in 5 business days.",
                    }
                ]
            }
        },
    )

    assert assessment.accepted is False
    assert assessment.constraint == "grounded_answer"
    assert assessment.reason == "grounded_answer_unsupported"


def test_quality_scorer_accepts_supported_summary_with_context() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize the following article in one sentence.",
                }
            ]
        },
        "Byte reduces repeated LLM calls for support teams.",
        context_hints={
            "_byte_raw_aux_context": {
                "byte_document_context": [
                    "Byte reduces repeated LLM calls for support teams and lowers repeated latency."
                ]
            }
        },
    )

    assert assessment.accepted is True
    assert assessment.constraint == "summary_style"
    assert assessment.score >= 0.28


def test_quality_scorer_does_not_treat_generic_city_name_phrase_as_exact_token() -> None:
    scorer = QualityScorer()
    assessment = scorer.assess_request_answer(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France? Return only the city name.",
                }
            ]
        },
        "Paris",
    )

    assert assessment.accepted is True
    assert assessment.reason != "exact_token_missing"
