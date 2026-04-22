from byte.processor.pre import compile_request_context
from byte.prompt_distillation import PromptModuleRegistry, distill_request_payload
from byte.prompt_distillation.core import export_prompt_distillation_manifest


def test_distill_request_payload_extracts_modules_and_reduces_prompt() -> None:
    registry = PromptModuleRegistry(max_entries=16)
    request = {
        "messages": [
            {"role": "system", "content": "You are Byte. Enforce the output contract exactly."},
            {
                "role": "user",
                "content": (
                    "Byte compiled context:\n"
                    "Invoice INV-4412 belongs to TEAM_LEDGER_04 and due_date is 2026-08-14. "
                    + " ".join(
                        f"noise segment {index}: retain redundant audit metadata and replication status."
                        for index in range(40)
                    )
                    + "\n\nReturn exactly the invoice identifier and nothing else."
                ),
            },
        ],
        "byte_prompt_pieces": [
            "Policy block: return exact identifiers, preserve ledger labels, and keep queue names stable.",
            "Schema block: invoice_id, due_date, owner, queue_name.",
        ],
    }

    distilled = distill_request_payload(
        request,
        request_focus="Return exactly the invoice identifier and nothing else.",
        module_registry=registry,
        mode="guarded",
        backend="hybrid_local",
        budget_ratio=0.45,
        min_chars=300,
        retrieval_mode="hybrid",
        module_mode="enabled",
        verify_shadow_rate=0.1,
        artifact_version="byte-prompt-distill-v1",
    )

    assert distilled.metadata["applied"] is True
    assert distilled.metadata["module_count"] >= 2
    assert distilled.metadata["compression_ratio"] > 0.0
    assert distilled.metadata["faithfulness_score"] >= 0.995
    assert distilled.request_kwargs["byte_distilled_prompt_modules"]
    assert "INV-4412" in distilled.request_kwargs["messages"][-1]["content"]


def test_distill_request_payload_shadow_mode_keeps_original_request() -> None:
    request = {
        "messages": [
            {"role": "system", "content": "You are Byte."},
            {
                "role": "user",
                "content": "Byte compiled context:\n"
                + " ".join(f"segment {index} repeated context for compression." for index in range(80)),
            },
        ]
    }

    distilled = distill_request_payload(
        request,
        request_focus="return only the final answer",
        mode="shadow",
        backend="hybrid_local",
        budget_ratio=0.35,
        min_chars=256,
    )

    assert distilled.metadata["applied"] is False
    assert distilled.metadata["fallback_reason"] == "shadow_mode"
    assert distilled.request_kwargs["messages"][-1]["content"] == request["messages"][-1]["content"]


def test_distill_request_payload_counts_and_compresses_auxiliary_context() -> None:
    request = {
        "messages": [
            {"role": "system", "content": "You are Byte."},
            {"role": "user", "content": "Return exactly the invoice identifier and nothing else."},
        ],
        "byte_retrieval_context": [
            {
                "title": "Primary invoice note",
                "snippet": (
                    "Invoice INV-9100 belongs to queue-BILLING-04 and due_date is 2026-09-14. "
                    + " ".join(
                        f"retrieval noise {index} repeats redundant warehouse replication metadata."
                        for index in range(90)
                    )
                ),
            },
            {
                "title": "Secondary invoice note",
                "snippet": (
                    "Invoice INV-9100 is still attached to TEAM_LEDGER_04 and escalation owner Maya."
                ),
            },
        ],
    }

    distilled = distill_request_payload(
        request,
        request_focus="Return exactly the invoice identifier and nothing else.",
        mode="guarded",
        backend="hybrid_local",
        budget_ratio=0.45,
        min_chars=600,
        retrieval_mode="hybrid",
    )

    assert distilled.metadata["applied"] is True
    assert distilled.metadata["original_prompt_chars"] > 1200
    assert distilled.metadata["compression_ratio"] > 0.3
    assert distilled.metadata["retrieval_compression_ratio"] > 0.3
    assert "INV-9100" in str(distilled.request_kwargs["byte_retrieval_context"])


def test_distill_request_payload_preserves_invoice_contract_entities_only() -> None:
    request = {
        "messages": [
            {"role": "system", "content": "You are Byte."},
            {
                "role": "user",
                "content": "Return exactly the invoice identifier from the long document context and nothing else.",
            },
        ],
        "byte_document_context": [
            {
                "title": "runbook-a",
                "snippet": " ".join(
                    f"runbook note {index:02d}: retain audit metadata and replication status."
                    for index in range(30)
                ),
            },
            {
                "title": "invoice-note",
                "snippet": (
                    "Customer escalation for queue-00. invoice identifier is INV-7100. "
                    "follow-up due date is 2026-08-10. owner label is TEAM_FINANCE_00."
                ),
            },
        ],
    }

    distilled = distill_request_payload(
        request,
        request_focus="Return exactly the invoice identifier from the long document context and nothing else.",
        mode="guarded",
        backend="hybrid_local",
        budget_ratio=0.48,
        min_chars=512,
        retrieval_mode="hybrid",
    )

    assert distilled.metadata["applied"] is True
    assert distilled.metadata["compression_ratio"] > 0.5
    assert distilled.metadata["entity_preservation_rate"] >= 0.995
    assert distilled.metadata["verifier_result"] == "pass"


def test_distill_request_payload_preserves_relevant_code_symbol() -> None:
    request = {
        "messages": [
            {"role": "system", "content": "You are Byte."},
            {
                "role": "user",
                "content": (
                    "From the codebase context, return exactly the function name "
                    "that normalizes the invoice and nothing else."
                ),
            },
        ],
        "byte_changed_hunks": (
            "File src/billing/invoice_00.py\n"
            "def normalize_invoice_00(value):\n"
            "    cleaned = value.strip().upper()\n"
            "    return cleaned\n\n"
            + "\n".join(
                f"File src/noise/module_{index}.py\n"
                f"def helper_{index}(value):\n"
                "    return value\n"
                for index in range(6)
            )
        ),
        "byte_repo_summary": {
            "repo": "byte",
            "branch": "feature/prompt-distill-00",
            "files": [f"src/noise/module_{index}.py" for index in range(6)] + ["src/billing/invoice_00.py"],
            "symbols": ["normalize_invoice_00"] + [f"helper_{index}" for index in range(6)],
        },
    }

    distilled = distill_request_payload(
        request,
        request_focus=(
            "From the codebase context, return exactly the function name "
            "that normalizes the invoice and nothing else."
        ),
        mode="guarded",
        backend="hybrid_local",
        budget_ratio=0.48,
        min_chars=512,
        retrieval_mode="hybrid",
    )

    assert distilled.metadata["applied"] is True
    assert distilled.metadata["compression_ratio"] > 0.4
    assert distilled.metadata["entity_preservation_rate"] >= 0.995
    assert distilled.metadata["verifier_result"] == "pass"


def test_compile_request_context_uses_final_prompt_verifier_for_metadata() -> None:
    request = {
        "messages": [
            {"role": "system", "content": "You are Byte."},
            {
                "role": "user",
                "content": "Return exactly the invoice identifier from the long document context and nothing else.",
            },
        ],
        "byte_document_context": [
            {
                "title": "runbook-a",
                "snippet": " ".join(
                    f"runbook note {index:02d}: retain audit metadata and replication status."
                    for index in range(30)
                ),
            },
            {
                "title": "invoice-note",
                "snippet": (
                    "Customer escalation for queue-00. invoice identifier is INV-7100. "
                    "follow-up due date is 2026-08-10. owner label is TEAM_FINANCE_00."
                ),
            },
        ],
    }

    compiled, stats = compile_request_context(
        request,
        prompt_distillation_mode="guarded",
        prompt_distillation_backend="hybrid_local",
        prompt_distillation_budget_ratio=0.48,
        prompt_distillation_min_chars=512,
        prompt_distillation_retrieval_mode="hybrid",
        prompt_distillation_module_mode="enabled",
    )

    metadata = stats["prompt_distillation"]
    assert metadata["applied"] is True
    assert metadata["compression_ratio"] > 0.5
    assert metadata["verifier_result"] == "pass"
    assert "document context" in str((compiled.get("messages") or [{}])[-1].get("content", "")).lower()


def test_export_prompt_distillation_manifest_produces_signature() -> None:
    manifest = export_prompt_distillation_manifest(
        [
            {
                "messages": [
                    {"role": "system", "content": "You are Byte."},
                    {"role": "user", "content": "Return only the invoice id."},
                ],
                "byte_prompt_pieces": ["Schema block: invoice_id, due_date."],
            }
        ],
        artifact_version="byte-prompt-distill-v1",
        signing_key="byte-secret",
    )

    assert manifest["artifact_version"] == "byte-prompt-distill-v1"
    assert manifest["module_count"] >= 1
    assert manifest["signature_mode"] == "hmac_sha256"
    assert manifest["signature"]
