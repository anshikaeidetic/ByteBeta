from byte import Config
from byte.processor.reuse_policy import detect_reuse_policy


def test_unique_exact_token_uses_context_only_policy() -> None:
    policy = detect_reuse_policy(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Single-token benchmark request. Reply exactly STRESS_UNIQUE_0001 and nothing else.",
                }
            ]
        },
        config=Config(enable_token_counter=False),
    )

    assert policy.mode == "context_only"
    assert policy.unique_output is True
    assert policy.exact_token == "STRESS_UNIQUE_0001"


def test_normal_exact_token_stays_full_reuse() -> None:
    policy = detect_reuse_policy(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly TOKYO and nothing else.",
                }
            ]
        },
        config=Config(enable_token_counter=False),
    )

    assert policy.mode == "full_reuse"
    assert policy.unique_output is False


def test_explicit_override_can_force_direct_only() -> None:
    policy = detect_reuse_policy(
        {
            "messages": [{"role": "user", "content": "Summarize this."}],
            "byte_reuse_policy": "direct_only",
        },
        config=Config(enable_token_counter=False),
    )

    assert policy.mode == "direct_only"
    assert policy.explicit is True


def test_grounded_aux_context_uses_context_only_policy() -> None:
    policy = detect_reuse_policy(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the handle_request function in this repository.",
                }
            ]
        },
        config=Config(enable_token_counter=False),
        context={
            "_byte_raw_aux_context": {
                "byte_repo_summary": {"modules": ["auth", "request validation", "handle_request"]},
                "byte_changed_files": ["app/server.py"],
            }
        },
    )

    assert policy.mode == "context_only"
    assert policy.reason == "grounded_context_request"


def test_inline_source_text_without_byte_aux_context_stays_full_reuse() -> None:
    policy = detect_reuse_policy(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Extract invoice_id and amount from the document.\n\n"
                        "Document: Invoice INV-4101 total amount $4,237 due on 2026-06-11."
                    ),
                }
            ]
        },
        config=Config(enable_token_counter=False),
    )

    assert policy.mode == "full_reuse"


def test_retrieval_context_without_session_scope_stays_full_reuse() -> None:
    policy = detect_reuse_policy(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize this repository health status in one sentence.",
                }
            ]
        },
        config=Config(enable_token_counter=False),
        context={
            "_byte_raw_aux_context": {
                "byte_retrieval_context": {"docs": ["repository health answer-1"]}
            }
        },
    )

    assert policy.mode == "full_reuse"
