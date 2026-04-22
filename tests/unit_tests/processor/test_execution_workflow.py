from byte.processor.execution import ExecutionMemoryStore, FailureMemoryStore, PatchPatternStore
from byte.processor.optimization_memory import stable_digest
from byte.processor.pre import compile_request_context
from byte.processor.workflow import detect_ambiguity, plan_request_workflow


def test_compile_request_context_dedupes_and_summarizes_messages() -> None:
    request = {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Fix this.\n```python\nprint('hi')\n```"},
            {"role": "user", "content": "Fix this.\n```python\nprint('hi')\n```"},
            {"role": "assistant", "content": "I can help with that."},
            {"role": "user", "content": "Here is the updated file.\n```python\nprint('hi')\n```"},
            {"role": "user", "content": "Final ask.\n```python\nprint('bye')\n```"},
        ]
    }

    compiled, stats = compile_request_context(request, keep_last_messages=2, max_chars=400)

    assert stats["applied"] is True
    assert stats["deduped_messages"] >= 1
    assert stats["summarized_messages"] >= 1
    assert len(compiled["messages"]) <= 4
    assert "Previous conversation summary" in compiled["messages"][1]["content"]


def test_detect_ambiguity_for_code_request_without_context() -> None:
    assessment = detect_ambiguity(
        {
            "messages": [
                {"role": "user", "content": "Fix this bug for me."},
            ]
        }
    )

    assert assessment.ambiguous is True
    assert assessment.reason in {"missing_reference", "missing_code_context", "too_short"}
    assert "file" in assessment.question.lower() or "code" in assessment.question.lower()


def test_detect_ambiguity_does_not_flag_simple_exact_or_translation_prompts() -> None:
    exact = detect_ambiguity(
        {
            "messages": [
                {"role": "user", "content": "Reply with exactly BYTE_ROUTE_OK and nothing else."},
            ]
        }
    )
    translation = detect_ambiguity(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Translate to Spanish and answer with the translation only: Good morning team",
                },
            ]
        }
    )

    assert exact.ambiguous is False
    assert translation.ambiguous is False


def test_detect_ambiguity_does_not_clarify_exact_prompt_that_mentions_notes_below() -> None:
    assessment = detect_ambiguity(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Draft a release note headline based on the notes below, then reply with exactly RELEASE_NOTE_READY and nothing else.\n"
                        "Drafting notes:\n"
                        "- Added CSV export for analytics dashboards\n"
                        "- Reduced report generation latency by 35 percent\n"
                        "- Fixed duplicate billing edge case for seat upgrades"
                    ),
                }
            ]
        }
    )

    assert assessment.ambiguous is False


def test_detect_ambiguity_does_not_clarify_exact_prompt_with_code_topic_word() -> None:
    assessment = detect_ambiguity(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Single-token benchmark request 05.\n"
                        "Topic: code review follow-up 05.\n"
                        "Reply exactly STRESS_UNIQUE_05 and nothing else."
                    ),
                }
            ]
        }
    )

    assert assessment.ambiguous is False


def test_detect_ambiguity_does_not_treat_generic_test_word_as_code_context() -> None:
    assessment = detect_ambiguity(
        {
            "messages": [
                {"role": "user", "content": "test calculate 1+3"},
            ]
        }
    )

    assert assessment.ambiguous is False


def test_detect_ambiguity_allows_classification_with_brace_label_set() -> None:
    assessment = detect_ambiguity(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Clause text:\n"
                        "This agreement renews automatically for successive one-year terms unless either party "
                        "provides written notice at least 30 days before the renewal date.\n"
                        "Return exactly one label from {AUTO_RENEWAL, TERMINATION_FOR_CAUSE, DATA_PROCESSING}."
                    ),
                },
            ]
        }
    )

    assert assessment.ambiguous is False


def test_detect_ambiguity_flags_missing_source_for_document_summary() -> None:
    assessment = detect_ambiguity(
        {
            "messages": [
                {"role": "user", "content": "Summarize the following article in one sentence."},
            ]
        }
    )

    assert assessment.ambiguous is True
    assert assessment.reason == "missing_source_context"
    assert "summarize" in assessment.question.lower() or "article" in assessment.question.lower()


def test_detect_ambiguity_allows_summary_with_inline_source_text() -> None:
    assessment = detect_ambiguity(
        {
            "messages": [
                {
                    "role": "user",
                    "content": 'Summarize the following article in one sentence. Article: "Byte reduces repeated LLM calls for support teams and improves latency."',
                },
            ]
        }
    )

    assert assessment.ambiguous is False


def test_detect_ambiguity_uses_repo_summary_and_changed_files_as_source_context() -> None:
    assessment = detect_ambiguity(
        {
            "messages": [
                {"role": "user", "content": "Fix the selected bug in the nearby file."},
            ]
        },
        context_hints={
            "_byte_raw_aux_context": {
                "byte_repo_summary": "The repository contains a cart service and a checkout workflow.",
                "byte_changed_files": ["src/cart.py"],
            }
        },
    )

    assert assessment.ambiguous is False


def test_execution_memory_lookup_requires_verified_entry_when_requested() -> None:
    store = ExecutionMemoryStore()
    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Fix the bug.\n```python\ndef add_item(item, items=[]):\n    items.append(item)\n    return items\n```",
            }
        ],
    }
    store.remember(
        request,
        answer="patch-a",
        verification={"verified": False},
        repo_fingerprint="repo-1",
        model="gpt-4o-mini",
    )
    store.remember(
        request,
        answer="patch-b",
        verification={"verified": True},
        repo_fingerprint="repo-1",
        model="gpt-4o-mini",
    )

    assert (
        store.lookup(
            request,
            answer="patch-a",
            repo_fingerprint="repo-1",
            model="gpt-4o-mini",
            verified_only=True,
        )
        is None
    )
    verified = store.lookup(
        request,
        answer="patch-b",
        repo_fingerprint="repo-1",
        model="gpt-4o-mini",
        verified_only=True,
    )
    assert verified is not None
    assert verified["verified"] is True


def test_patch_pattern_store_generates_suggested_diff() -> None:
    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Fix the bug.\n"
                    "```python\n"
                    "def add_item(item, items=[]):\n"
                    "    items.append(item)\n"
                    "    return items\n"
                    "```"
                ),
            }
        ],
    }
    patch = (
        "--- original\n"
        "+++ fixed\n"
        "@@\n"
        "-def add_item(item, items=[]):\n"
        "+def add_item(item, items=None):\n"
        "+    if items is None:\n"
        "+        items = []\n"
    )
    store = PatchPatternStore()
    store.remember(
        request,
        patch=patch,
        repo_fingerprint="repo-1",
        verified=True,
        model="gpt-4o-mini",
    )

    suggestion = store.suggest(request, repo_fingerprint="repo-1")

    assert suggestion is not None
    assert "items=None" in suggestion["patched_code"]
    assert "byte_suggested" in suggestion["patch_text"]


def test_workflow_planner_prefers_verified_patch_reuse() -> None:
    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Apply the same fix.\n```python\nprint('x')\n```"}
        ],
        "byte_allow_patch_reuse": True,
    }
    decision = plan_request_workflow(
        request,
        type(
            "Cfg",
            (),
            {
                "ambiguity_min_chars": 24,
                "ambiguity_detection": True,
                "delta_generation": True,
                "planner_allow_verified_short_circut": True,
                "planner_allow_verified_short_circuit": True,
                "budget_strategy": "balanced",
            },
        )(),
        ambiguity=detect_ambiguity(request),
        patch_candidate={"patch_text": "--- original\n+++ byte_suggested"},
    )

    assert decision.action == "reuse_verified_patch"


def test_failure_memory_hint_returns_negative_context_digests() -> None:
    store = FailureMemoryStore()
    refund_item = {
        "title": "Refund Policy",
        "snippet": "Refund duplicate subscription charges in 5 business days.",
    }
    request = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Answer the refund question."}],
    }
    store.record(
        request,
        reason="verification_failed",
        metadata={
            "negative_context_digests": {
                "retrieval_context": [stable_digest(refund_item)],
            },
            "negative_context_summaries": {
                stable_digest(refund_item): "retrieval context: Refund Policy",
            },
        },
    )

    hint = store.hint(request)

    assert hint["prefer_expensive"] is True
    assert stable_digest(refund_item) in hint["negative_context_digests"]["retrieval_context"]
