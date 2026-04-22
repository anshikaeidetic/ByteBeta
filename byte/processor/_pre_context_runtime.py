"""Orchestration entrypoint for request context compilation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from byte.processor._optimization_summary import extract_prompt_pieces
from byte.processor._pre_context_aux import _compile_aux_context
from byte.processor._pre_context_budget import (
    _append_compiled_context_notes,
    _compiled_primary_chars,
    _fit_compiled_context_notes,
    _trim_text_middle,
)
from byte.processor._pre_context_distillation import _merge_prompt_distillation_metadata
from byte.processor._pre_context_messages import _dedupe_messages, _summarize_messages
from byte.processor._pre_selection import _request_focus_text
from byte.prompt_distillation import (
    PromptDistillationResult,
    distill_request_payload,
    measure_request_prompt,
    verify_request_faithfulness,
)
from byte.utils.multimodal import content_signature


def compile_request_context(
    request_kwargs: dict[str, Any],
    *,
    keep_last_messages: int = 6,
    max_chars: int = 6000,
    prompt_piece_store: Any | None = None, artifact_memory_store: Any | None = None,
    session_delta_store: Any | None = None, session_key: str = "",
    memory_scope: str = "",
    relevance_top_k: int = 4,
    related_memory: bool = True, related_min_score: float = 0.18,
    negative_context_digests: dict[str, list[str]] | None = None,
    context_sketches: bool = True, focus_distillation: bool = True,
    total_aux_budget_ratio: float = 0.65,
    cross_note_dedupe: bool = True, prefix_messages: bool = False, stable_prefix: bool = False,
    prompt_module_registry: Any | None = None,
    prompt_distillation_mode: str = "disabled", prompt_distillation_backend: str = "hybrid_local",
    prompt_distillation_budget_ratio: float = 0.55,
    prompt_distillation_min_chars: int = 512,
    prompt_distillation_retrieval_mode: str = "hybrid", prompt_distillation_module_mode: str = "enabled",
    prompt_distillation_verify_shadow_rate: float = 0.1, prompt_distillation_artifact_version: str = "byte-prompt-distill-v1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shrink request payloads deterministically before provider dispatch."""
    raw_request = deepcopy(request_kwargs or {})
    compiled = deepcopy(request_kwargs or {})
    stats = {
        "applied": False,
        "original_chars": 0,
        "compiled_chars": 0,
        "saved_chars": 0,
        "deduped_messages": 0,
        "summarized_messages": 0,
        "collapsed_code_blocks": 0,
        "compiled_aux_contexts": 0,
        "prompt_piece_entries": 0,
        "artifact_entries": 0,
        "artifact_reuse_hits": 0,
        "related_artifact_hits": 0,
        "session_delta_hits": 0,
        "artifact_savings_chars": 0,
        "relevance_pruned_items": 0,
        "negative_pruned_items": 0,
        "sketches_used": 0,
        "focused_distillation_hits": 0,
        "cross_note_deduped_notes": 0,
        "aux_budget_pruned_notes": 0,
        "aux_budget_trimmed_chars": 0,
        "prompt_distillation_applied": 0,
        "prompt_distillation_saved_chars": 0,
        "prompt_distillation_module_hits": 0,
        "prompt_distillation_fallbacks": 0,
        "prompt_distillation": {},
    }
    request_focus = _request_focus_text(compiled)

    messages = compiled.get("messages") or []
    if messages:
        stats["original_chars"] = sum(
            len(
                content_signature(msg.get("content", ""))
                if isinstance(msg.get("content", ""), list)
                else str(msg.get("content", ""))
            )
            for msg in messages
        )
        preserved_system = [deepcopy(msg) for msg in messages if msg.get("role") == "system"]
        non_system = [deepcopy(msg) for msg in messages if msg.get("role") != "system"]
        deduped, deduped_count, collapsed_code = _dedupe_messages(non_system)
        stats["deduped_messages"] = deduped_count
        stats["collapsed_code_blocks"] = collapsed_code
        if len(deduped) > keep_last_messages:
            dropped = deduped[:-keep_last_messages]
            kept = deduped[-keep_last_messages:]
            summary_message = {
                "role": "system",
                "content": _summarize_messages(dropped),
            }
            stats["summarized_messages"] = len(dropped)
            compiled["messages"] = preserved_system + [summary_message] + kept
        else:
            compiled["messages"] = preserved_system + deduped
        stats["compiled_chars"] = sum(
            len(
                content_signature(msg.get("content", ""))
                if isinstance(msg.get("content", ""), list)
                else str(msg.get("content", ""))
            )
            for msg in compiled["messages"]
        )
    else:
        text_key = (
            "prompt"
            if compiled.get("prompt") is not None
            else "input"
            if compiled.get("input") is not None
            else ""
        )
        if text_key:
            raw_text = str(compiled.get(text_key) or "")
            stats["original_chars"] = len(raw_text)
            compiled[text_key] = _trim_text_middle(raw_text, max_chars=max_chars)
            stats["compiled_chars"] = len(str(compiled[text_key]))

    prompt_pieces = extract_prompt_pieces(compiled)
    if prompt_pieces and prompt_piece_store is not None:
        remembered_pieces = prompt_piece_store.remember_many(
            prompt_pieces,
            scope=memory_scope or "",
            source="request",
        )
        stats["prompt_piece_entries"] = len(remembered_pieces)

    pre_distillation = distill_request_payload(
        compiled,
        request_focus=request_focus,
        prompt_piece_store=prompt_piece_store,
        module_registry=prompt_module_registry,
        mode=str(prompt_distillation_mode or "disabled"),
        backend=str(prompt_distillation_backend or "hybrid_local"),
        budget_ratio=float(prompt_distillation_budget_ratio or 0.55),
        min_chars=int(prompt_distillation_min_chars or 512),
        retrieval_mode=str(prompt_distillation_retrieval_mode or "hybrid"),
        module_mode=str(prompt_distillation_module_mode or "enabled"),
        verify_shadow_rate=float(prompt_distillation_verify_shadow_rate or 0.1),
        artifact_version=str(
            prompt_distillation_artifact_version or "byte-prompt-distill-v1"
        ),
    )
    compiled = pre_distillation.request_kwargs

    aux_context_keys = (
        "byte_changed_hunks",
        "byte_changed_files",
        "byte_repo_snapshot",
        "byte_repo_summary",
        "byte_retrieval_context",
        "byte_document_context",
        "byte_support_articles",
        "byte_tool_result_context",
        "byte_prompt_pieces",
    )
    base_chars = _compiled_primary_chars(compiled)
    compiled_notes: list[dict[str, Any]] = []
    seen_compiled_digests = set()
    for ctx_key in aux_context_keys:
        value = compiled.pop(ctx_key, None)
        if value in (None, "", [], {}):
            continue
        ctx_note, ctx_stats = _compile_aux_context(
            ctx_key,
            value,
            max_chars=max_chars,
            artifact_memory_store=artifact_memory_store,
            session_delta_store=session_delta_store,
            session_key=session_key,
            memory_scope=memory_scope,
            seen_digests=seen_compiled_digests,
            request_focus=request_focus,
            relevance_top_k=relevance_top_k,
            related_memory=related_memory,
            related_min_score=related_min_score,
            negative_context_digests=negative_context_digests or {},
            context_sketches=context_sketches,
            focus_distillation=focus_distillation,
            stable_prefix=stable_prefix,
        )
        if not ctx_note or not ctx_note.get("note"):
            continue
        ctx_note["order"] = len(compiled_notes)
        compiled_notes.append(ctx_note)
        stats["compiled_aux_contexts"] += 1
        stats["artifact_entries"] += int(ctx_stats.get("artifact_entries", 0) or 0)
        stats["artifact_reuse_hits"] += int(ctx_stats.get("artifact_reuse_hits", 0) or 0)
        stats["related_artifact_hits"] += int(ctx_stats.get("related_artifact_hits", 0) or 0)
        stats["session_delta_hits"] += int(ctx_stats.get("session_delta_hits", 0) or 0)
        stats["artifact_savings_chars"] += int(ctx_stats.get("saved_chars", 0) or 0)
        stats["relevance_pruned_items"] += int(ctx_stats.get("relevance_pruned_items", 0) or 0)
        stats["negative_pruned_items"] += int(ctx_stats.get("negative_pruned_items", 0) or 0)
        stats["sketches_used"] += int(ctx_stats.get("sketches_used", 0) or 0)
        stats["focused_distillation_hits"] += int(
            ctx_stats.get("focused_distillation_hits", 0) or 0
        )

    if compiled_notes:
        selected_notes, fit_stats = _fit_compiled_context_notes(
            compiled_notes,
            max_chars=max_chars,
            base_chars=base_chars,
            total_aux_budget_ratio=total_aux_budget_ratio,
            cross_note_dedupe=cross_note_dedupe,
        )
        stats["cross_note_deduped_notes"] = int(fit_stats.get("cross_note_deduped_notes", 0) or 0)
        stats["aux_budget_pruned_notes"] = int(fit_stats.get("aux_budget_pruned_notes", 0) or 0)
        stats["aux_budget_trimmed_chars"] = int(fit_stats.get("aux_budget_trimmed_chars", 0) or 0)
        _append_compiled_context_notes(compiled, selected_notes, prefix_messages=prefix_messages)

    current_prompt = measure_request_prompt(compiled)
    post_target_chars = max(
        256,
        int(
            max(
                int(pre_distillation.metadata.get("original_prompt_chars", 0) or 0),
                int(current_prompt.get("chars", 0) or 0),
            )
            * float(prompt_distillation_budget_ratio or 0.55)
        ),
    )
    if int(current_prompt.get("chars", 0) or 0) <= post_target_chars:
        post_distillation = PromptDistillationResult(
            compiled,
            {
                "applied": False,
                "mode": str(prompt_distillation_mode or "disabled"),
                "backend": str(prompt_distillation_backend or "hybrid_local"),
                "backend_path": "disabled",
                "artifact_version": str(
                    prompt_distillation_artifact_version or "byte-prompt-distill-v1"
                ),
                "original_prompt_chars": int(current_prompt.get("chars", 0) or 0),
                "distilled_prompt_chars": int(current_prompt.get("chars", 0) or 0),
                "original_prompt_tokens": int(current_prompt.get("tokens", 0) or 0),
                "distilled_prompt_tokens": int(current_prompt.get("tokens", 0) or 0),
                "token_reduction_ratio": 0.0,
                "compression_ratio": 0.0,
                "module_hits": 0,
                "module_count": 0,
                "retrieval_compression_ratio": 0.0,
                "faithfulness_score": 1.0,
                "entity_preservation_rate": 1.0,
                "schema_preservation_rate": 1.0,
                "verifier_result": "pass",
                "fallback_reason": "already_within_budget",
                "shadow_verified": False,
                "verify_shadow_rate": float(prompt_distillation_verify_shadow_rate or 0.1),
            },
        )
    else:
        post_distillation = distill_request_payload(
            compiled,
            request_focus=request_focus,
            prompt_piece_store=prompt_piece_store,
            module_registry=prompt_module_registry,
            mode=str(prompt_distillation_mode or "disabled"),
            backend=str(prompt_distillation_backend or "hybrid_local"),
            budget_ratio=float(prompt_distillation_budget_ratio or 0.55),
            min_chars=int(prompt_distillation_min_chars or 512),
            retrieval_mode=str(prompt_distillation_retrieval_mode or "hybrid"),
            module_mode=str(prompt_distillation_module_mode or "enabled"),
            verify_shadow_rate=float(prompt_distillation_verify_shadow_rate or 0.1),
            artifact_version=str(
                prompt_distillation_artifact_version or "byte-prompt-distill-v1"
            ),
        )
    compiled = post_distillation.request_kwargs
    final_prompt = measure_request_prompt(compiled)
    final_verifier = verify_request_faithfulness(raw_request, compiled)
    prompt_distillation = _merge_prompt_distillation_metadata(
        dict(pre_distillation.metadata or {}),
        dict(post_distillation.metadata or {}),
        final_prompt_chars=int(final_prompt.get("chars", 0) or 0),
        final_prompt_tokens=int(final_prompt.get("tokens", 0) or 0),
        final_verifier=final_verifier,
    )
    stats["prompt_distillation"] = prompt_distillation
    stats["prompt_distillation_applied"] = 1 if prompt_distillation.get("applied") else 0
    stats["prompt_distillation_saved_chars"] = max(
        int(prompt_distillation.get("original_prompt_chars", 0) or 0)
        - int(prompt_distillation.get("distilled_prompt_chars", 0) or 0),
        0,
    )
    stats["prompt_distillation_module_hits"] = int(
        prompt_distillation.get("module_hits", 0) or 0
    )
    stats["prompt_distillation_fallbacks"] = (
        0 if not str(prompt_distillation.get("fallback_reason", "") or "") else 1
    )

    if stats["compiled_chars"] == 0:
        stats["compiled_chars"] = stats["original_chars"]
    if compiled.get("messages"):
        stats["compiled_chars"] = sum(
            len(
                content_signature(msg.get("content", ""))
                if isinstance(msg.get("content", ""), list)
                else str(msg.get("content", ""))
            )
            for msg in compiled["messages"]
        )
    elif compiled.get("prompt") is not None:
        stats["compiled_chars"] = len(str(compiled.get("prompt") or ""))
    elif compiled.get("input") is not None:
        stats["compiled_chars"] = len(str(compiled.get("input") or ""))
    stats["saved_chars"] = max(stats["original_chars"] - stats["compiled_chars"], 0)
    stats["applied"] = any(
        stats[key] > 0
        for key in (
            "saved_chars",
            "deduped_messages",
            "summarized_messages",
            "collapsed_code_blocks",
            "compiled_aux_contexts",
            "artifact_savings_chars",
            "artifact_reuse_hits",
            "related_artifact_hits",
            "session_delta_hits",
            "relevance_pruned_items",
            "negative_pruned_items",
            "sketches_used",
            "focused_distillation_hits",
            "cross_note_deduped_notes",
            "aux_budget_pruned_notes",
            "aux_budget_trimmed_chars",
        )
    )
    return compiled, stats
