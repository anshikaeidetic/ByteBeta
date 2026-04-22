"""Prompt-distillation merge helpers for request context compilation."""

from __future__ import annotations

from typing import Any


def _merge_prompt_distillation_metadata(
    pre_metadata: dict[str, Any],
    post_metadata: dict[str, Any],
    *,
    final_prompt_chars: int,
    final_prompt_tokens: int,
    final_verifier: dict[str, Any],
) -> dict[str, Any]:
    pre_metadata = dict(pre_metadata or {})
    post_metadata = dict(post_metadata or {})
    final_verifier = dict(final_verifier or {})
    original_chars = max(
        int(pre_metadata.get("original_prompt_chars", 0) or 0),
        int(post_metadata.get("original_prompt_chars", 0) or 0),
        int(final_prompt_chars or 0),
    )
    original_tokens = max(
        int(pre_metadata.get("original_prompt_tokens", 0) or 0),
        int(post_metadata.get("original_prompt_tokens", 0) or 0),
        int(final_prompt_tokens or 0),
    )
    final_chars = int(final_prompt_chars or original_chars)
    final_tokens = int(final_prompt_tokens or original_tokens)
    applied = bool(
        pre_metadata.get("applied")
        or post_metadata.get("applied")
        or final_chars < original_chars
        or final_tokens < original_tokens
    )
    merged = dict(pre_metadata)
    backend_path = "disabled"
    for candidate in (post_metadata.get("backend_path"), pre_metadata.get("backend_path")):
        normalized = str(candidate or "").strip()
        if normalized and normalized != "disabled":
            backend_path = normalized
            break
    merged.update(
        {
            "applied": applied,
            "backend_path": backend_path,
            "original_prompt_chars": original_chars,
            "distilled_prompt_chars": final_chars,
            "original_prompt_tokens": original_tokens,
            "distilled_prompt_tokens": final_tokens,
            "token_reduction_ratio": round(
                max(0.0, (original_tokens - final_tokens) / original_tokens),
                4,
            )
            if original_tokens > 0
            else 0.0,
            "compression_ratio": round(
                max(0.0, (original_chars - final_chars) / original_chars),
                4,
            )
            if original_chars > 0
            else 0.0,
            "retrieval_compression_ratio": max(
                float(pre_metadata.get("retrieval_compression_ratio", 0.0) or 0.0),
                float(post_metadata.get("retrieval_compression_ratio", 0.0) or 0.0),
            ),
            "module_hits": max(
                int(pre_metadata.get("module_hits", 0) or 0),
                int(post_metadata.get("module_hits", 0) or 0),
            ),
            "module_count": max(
                int(pre_metadata.get("module_count", 0) or 0),
                int(post_metadata.get("module_count", 0) or 0),
            ),
            "faithfulness_score": float(final_verifier.get("faithfulness_score", 1.0) or 1.0),
            "entity_preservation_rate": float(
                final_verifier.get("entity_preservation_rate", 1.0) or 1.0
            ),
            "schema_preservation_rate": float(
                final_verifier.get("schema_preservation_rate", 1.0) or 1.0
            ),
            "verifier_result": str(final_verifier.get("verifier_result", "pass") or "pass"),
        }
    )
    if applied and str(final_verifier.get("verifier_result", "pass") or "pass") == "pass":
        merged["fallback_reason"] = ""
    else:
        merged["fallback_reason"] = str(
            post_metadata.get("fallback_reason")
            or pre_metadata.get("fallback_reason")
            or "faithfulness_verifier_failed"
        )
    return merged
