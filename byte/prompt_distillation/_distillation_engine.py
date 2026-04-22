"""Prompt distillation orchestration and artifact trimming helpers."""

from __future__ import annotations

# ruff: noqa: F401
import hashlib
import hmac
import json
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Optional

from byte.processor.optimization_memory import (
    compact_text,
    distill_artifact_for_focus,
    estimate_tokens,
    extract_prompt_pieces,
    summarize_artifact_payload,
)
from byte.prompt_distillation._distillation_common import (
    _DISTILLABLE_FIELD_TYPES,
    _INJECTION_PATTERN,
    _MODULE_TYPES,
    _SPLIT_PATTERN,
    PromptDistillationResult,
    PromptModuleRegistry,
    _module_id,
    _ratio_delta,
    normalize_text,
)

try:
    from byte.prompt_distillation._distillation_faithfulness import (
        _preserve_segment,
        _segment_score,
        _verify_faithfulness,
    )
except ImportError:
    import re as _re
    _PRESERVE_RE = _re.compile(r"\b-?\d+(?:\.\d+)?%?\b|\"[A-Za-z0-9_:-]+\"\s*:|[A-Z]:\\|[A-Z][A-Z0-9_:-]{2,}\b")
    def _preserve_segment(segment: str) -> bool:  # type: ignore[misc]
        return bool(_PRESERVE_RE.search(str(segment or "")))
    def _segment_score(segment: str, **_kw) -> float:  # type: ignore[misc]
        return 0.5
    def _verify_faithfulness(*_a, **_kw):  # type: ignore[misc]
        return _a[0] if _a else {}
from byte.prompt_distillation._distillation_measure import (
    _primary_request_chars,
    _request_chars,
    _request_text,
    _retrieval_context_chars,
    _stringify_value,
)


def distill_request_payload(
    request_kwargs: dict[str, Any],
    *,
    request_focus: str = "",
    prompt_piece_store: Any | None = None,
    module_registry: PromptModuleRegistry | None = None,
    mode: str = "disabled",
    backend: str = "hybrid_local",
    budget_ratio: float = 0.55,
    min_chars: int = 512,
    retrieval_mode: str = "hybrid",
    module_mode: str = "enabled",
    verify_shadow_rate: float = 0.1,
    artifact_version: str = "byte-prompt-distill-v1",
) -> PromptDistillationResult:
    compiled = deepcopy(request_kwargs or {})
    normalized_mode = str(
        compiled.get("byte_prompt_distillation_mode", mode) or mode or "disabled"
    ).strip().lower()
    normalized_backend = str(
        compiled.get("byte_prompt_distillation_backend", backend) or backend or "hybrid_local"
    ).strip().lower()
    normalized_module_mode = "disabled" if compiled.get("byte_prompt_distillation_disable_modules") else str(
        module_mode or "enabled"
    ).strip().lower()
    normalized_retrieval_mode = str(retrieval_mode or "hybrid").strip().lower()
    effective_budget_ratio = float(
        compiled.get("byte_prompt_distillation_budget_ratio", budget_ratio) or budget_ratio or 0.55
    )
    effective_budget_ratio = max(0.2, min(effective_budget_ratio, 0.9))
    effective_min_chars = max(128, int(min_chars or 512))
    original_text = _request_text(compiled)
    original_chars = len(original_text)
    original_tokens = estimate_tokens(original_text)
    modules: list[dict[str, Any]] = []
    prompt_pieces = extract_prompt_pieces(compiled)
    if normalized_module_mode == "enabled":
        modules = _build_prompt_modules(
            prompt_pieces,
            prompt_piece_store=prompt_piece_store,
            module_registry=module_registry,
        )
        if modules:
            compiled["byte_distilled_prompt_modules"] = [item["module_id"] for item in modules]

    metadata = {
        "applied": False,
        "mode": normalized_mode,
        "backend": normalized_backend,
        "backend_path": "disabled",
        "artifact_version": str(artifact_version or "byte-prompt-distill-v1").strip(),
        "original_prompt_chars": original_chars,
        "distilled_prompt_chars": original_chars,
        "original_prompt_tokens": original_tokens,
        "distilled_prompt_tokens": original_tokens,
        "token_reduction_ratio": 0.0,
        "compression_ratio": 0.0,
        "module_hits": sum(1 for item in modules if int(item.get("hits", 0) or 0) > 0),
        "module_count": len(modules),
        "retrieval_compression_ratio": 0.0,
        "faithfulness_score": 1.0,
        "entity_preservation_rate": 1.0,
        "schema_preservation_rate": 1.0,
        "verifier_result": "skipped",
        "fallback_reason": "",
        "shadow_verified": False,
        "verify_shadow_rate": round(float(verify_shadow_rate or 0.0), 4),
    }
    if normalized_mode == "disabled":
        metadata["fallback_reason"] = "disabled"
        return PromptDistillationResult(compiled, metadata)
    if original_chars < effective_min_chars:
        metadata["fallback_reason"] = "below_min_chars"
        return PromptDistillationResult(compiled, metadata)

    target_chars = max(256, int(original_chars * effective_budget_ratio))
    distilled = deepcopy(compiled)
    backend_path = "heuristic"
    distilled, retrieval_ratio = _distill_auxiliary_fields(
        distilled,
        request_focus=request_focus,
        target_chars=target_chars,
        retrieval_mode=normalized_retrieval_mode,
        aggressive=False,
    )
    if distilled.get("messages"):
        messages = _distill_messages(
            list(distilled.get("messages") or []),
            request_focus=request_focus,
            target_chars=target_chars,
            retrieval_mode=normalized_retrieval_mode,
        )
        retrieval_ratio = max(
            retrieval_ratio,
            _retrieval_ratio(list(distilled.get("messages") or []), messages),
        )
        distilled["messages"] = messages
    elif distilled.get("prompt") is not None:
        distilled["prompt"] = _compress_text_block(
            str(distilled.get("prompt") or ""),
            request_focus=request_focus,
            max_chars=target_chars,
            retrieval_mode=normalized_retrieval_mode,
        )
    elif distilled.get("input") is not None:
        distilled["input"] = _compress_text_block(
            str(distilled.get("input") or ""),
            request_focus=request_focus,
            max_chars=target_chars,
            retrieval_mode=normalized_retrieval_mode,
        )

    distilled_chars = _request_chars(distilled)
    if distilled_chars > target_chars and normalized_backend in {"hybrid_local", "local_model"}:
        distilled, retrieval_ratio = _distill_auxiliary_fields(
            distilled,
            request_focus=request_focus,
            target_chars=target_chars,
            retrieval_mode=normalized_retrieval_mode,
            aggressive=True,
        )
        distilled = _compress_with_local_backend(
            distilled,
            request_focus=request_focus,
            max_chars=target_chars,
        )
        distilled_chars = _request_chars(distilled)
        backend_path = "local_model" if normalized_backend == "local_model" else "hybrid_local"

    verifier = _verify_faithfulness(compiled, distilled)
    distilled_text = _request_text(distilled)
    distilled_tokens = estimate_tokens(distilled_text)
    metadata.update(
        {
            "backend_path": backend_path,
            "distilled_prompt_chars": distilled_chars,
            "distilled_prompt_tokens": distilled_tokens,
            "token_reduction_ratio": _ratio_delta(original_tokens, distilled_tokens),
            "compression_ratio": _ratio_delta(original_chars, distilled_chars),
            "retrieval_compression_ratio": round(float(retrieval_ratio), 4),
            "faithfulness_score": round(float(verifier["faithfulness_score"]), 4),
            "entity_preservation_rate": round(float(verifier["entity_preservation_rate"]), 4),
            "schema_preservation_rate": round(float(verifier["schema_preservation_rate"]), 4),
            "verifier_result": str(verifier["verifier_result"]),
        }
    )
    if normalized_mode == "shadow":
        metadata["shadow_verified"] = verifier["verifier_result"] == "pass"
        metadata["fallback_reason"] = "shadow_mode"
        return PromptDistillationResult(compiled, metadata)
    if verifier["verifier_result"] != "pass":
        metadata["fallback_reason"] = "faithfulness_verifier_failed"
        return PromptDistillationResult(compiled, metadata)

    metadata["applied"] = True
    try:
        from byte.telemetry import (
            bump_research_counter as _bump,  # pylint: disable=import-outside-toplevel
        )
        _bump("prompt_distillation_calls")
        tokens_saved = max(0, int(original_tokens) - int(distilled_tokens))
        if tokens_saved > 0:
            _bump("prompt_distillation_tokens_saved", tokens_saved)
    except Exception:  # pragma: no cover - defensive
        pass
    return PromptDistillationResult(distilled, metadata)


def export_prompt_distillation_manifest(
    requests: list[dict[str, Any]],
    *,
    artifact_version: str = "byte-prompt-distill-v1",
    signing_key: str = "",
) -> dict[str, Any]:
    modules: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for request in requests or []:
        for piece in extract_prompt_pieces(request):
            piece_type = str(piece.get("type") or piece.get("piece_type") or "").strip().lower()
            if piece_type not in _MODULE_TYPES:
                continue
            module_id = _module_id(piece_type, piece.get("content"))
            entry = modules.setdefault(
                module_id,
                {
                    "module_id": module_id,
                    "module_type": piece_type,
                    "preview": compact_text(piece.get("content"), max_chars=160),
                    "normalized_preview": normalize_text(compact_text(piece.get("content"), max_chars=320)),
                    "occurrences": 0,
                    "artifact_version": str(artifact_version or "byte-prompt-distill-v1").strip(),
                },
            )
            entry["occurrences"] = int(entry.get("occurrences", 0) or 0) + 1
    manifest = {
        "artifact_version": str(artifact_version or "byte-prompt-distill-v1").strip(),
        "generated_at": int(time.time()),
        "module_count": len(modules),
        "modules": list(modules.values()),
    }
    payload = json.dumps(manifest["modules"], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if signing_key:
        signature = hmac.new(
            signing_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signature_mode = "hmac_sha256"
    else:
        signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        signature_mode = "sha256_digest"
    manifest["signature"] = signature
    manifest["signature_mode"] = signature_mode
    return manifest

def _build_prompt_modules(
    prompt_pieces: list[dict[str, Any]],
    *,
    prompt_piece_store: Any | None,
    module_registry: PromptModuleRegistry | None,
) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    registry = module_registry
    for piece in prompt_pieces or []:
        piece_type = str(piece.get("type") or piece.get("piece_type") or "").strip().lower()
        if piece_type not in _MODULE_TYPES:
            continue
        content = piece.get("content")
        module = {
            "module_id": _module_id(piece_type, content),
            "module_type": piece_type,
            "preview": compact_text(content, max_chars=120),
            "hits": 0,
        }
        if prompt_piece_store is not None:
            try:
                remembered = prompt_piece_store.remember(
                    piece_type,
                    content,
                    scope="prompt_distillation",
                    source="moduleization",
                    metadata={"module_id": module["module_id"]},
                )
                module["piece_key"] = remembered.get("key", "")
            except Exception:  # pylint: disable=W0703
                pass
        if registry is not None:
            try:
                remembered_module = registry.remember(
                    piece_type,
                    content,
                    scope="prompt_distillation",
                    source="moduleization",
                )
                module["hits"] = int(remembered_module.get("hits", 0) or 0)
            except Exception:  # pylint: disable=W0703
                pass
        modules.append(module)
    return modules


def _distill_messages(
    messages: list[dict[str, Any]],
    *,
    request_focus: str,
    target_chars: int,
    retrieval_mode: str,
) -> list[dict[str, Any]]:
    if not messages:
        return messages
    result = deepcopy(messages)
    current_chars = _request_chars({"messages": result})
    if current_chars <= target_chars:
        return result
    overflow = current_chars - target_chars
    for index in range(len(result) - 1, -1, -1):
        message = dict(result[index] or {})
        role = str(message.get("role", "") or "").strip().lower()
        if role == "system" and "byte compiled context" not in str(message.get("content", "") or "").lower():
            continue
        raw = message.get("content", "")
        if isinstance(raw, list):
            continue
        original = str(raw or "")
        if not original.strip():
            continue
        max_chars = max(96, len(original) - overflow)
        compressed = _compress_text_block(
            original,
            request_focus=request_focus,
            max_chars=max_chars,
            retrieval_mode=retrieval_mode,
        )
        if compressed == original:
            continue
        result[index]["content"] = compressed
        overflow -= max(0, len(original) - len(compressed))
        if overflow <= 0:
            break
    return result


def _compress_with_local_backend(
    request_kwargs: dict[str, Any],
    *,
    request_focus: str,
    max_chars: int,
) -> dict[str, Any]:
    result = deepcopy(request_kwargs)
    if result.get("messages"):
        for message in result["messages"]:
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            if len(content) <= max_chars:
                continue
            message["content"] = _compress_text_block(
                content,
                request_focus=request_focus,
                max_chars=max_chars,
                retrieval_mode="hybrid",
                aggressive=True,
            )
    elif result.get("prompt") is not None:
        result["prompt"] = _compress_text_block(
            str(result.get("prompt") or ""),
            request_focus=request_focus,
            max_chars=max_chars,
            retrieval_mode="hybrid",
            aggressive=True,
        )
    elif result.get("input") is not None:
        result["input"] = _compress_text_block(
            str(result.get("input") or ""),
            request_focus=request_focus,
            max_chars=max_chars,
            retrieval_mode="hybrid",
            aggressive=True,
        )
    return result


def _compress_text_block(
    text: str,
    *,
    request_focus: str,
    max_chars: int,
    retrieval_mode: str,
    aggressive: bool = False,
) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    if "byte compiled context:" in text.lower():
        prefix, _, tail = text.partition("Byte compiled context:")
        distilled_tail = _compress_context_lines(
            tail,
            request_focus=request_focus,
            max_chars=max_chars - len(prefix) - 24,
            retrieval_mode=retrieval_mode,
            aggressive=aggressive,
        )
        return f"{prefix}Byte compiled context:\n{distilled_tail}".strip()
    return _compress_context_lines(
        text,
        request_focus=request_focus,
        max_chars=max_chars,
        retrieval_mode=retrieval_mode,
        aggressive=aggressive,
    )


def _compress_context_lines(
    text: str,
    *,
    request_focus: str,
    max_chars: int,
    retrieval_mode: str,
    aggressive: bool,
) -> str:
    segments = [segment.strip() for segment in _SPLIT_PATTERN.split(str(text or "")) if segment.strip()]
    if not segments:
        return compact_text(text, max_chars=max_chars)
    query_tokens = set(normalize_text(request_focus).split())
    ranked: list[tuple[float, int, str]] = []
    for index, segment in enumerate(segments):
        score = _segment_score(segment, query_tokens=query_tokens)
        if retrieval_mode == "extractive" and score < 0.15 and not _preserve_segment(segment):
            continue
        if retrieval_mode == "hybrid" and aggressive and score < 0.12 and not _preserve_segment(segment):
            continue
        ranked.append((score, index, segment))
    if not ranked:
        return compact_text(text, max_chars=max_chars)
    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected: list[tuple[int, str]] = []
    remaining = max_chars
    for score, index, segment in ranked:
        candidate = segment
        if aggressive and score < 0.25 and not _preserve_segment(segment):
            candidate = compact_text(segment, max_chars=max(48, min(96, remaining)))
        needed = len(candidate) + (1 if selected else 0)
        if needed > remaining and not _preserve_segment(candidate):
            continue
        selected.append((index, candidate))
        remaining -= needed
        if remaining <= 48:
            break
    if not selected:
        return compact_text(text, max_chars=max_chars)
    selected.sort(key=lambda item: item[0])
    return compact_text(" ".join(segment for _, segment in selected), max_chars=max_chars)

def _retrieval_ratio(original_messages: list[dict[str, Any]], distilled_messages: list[dict[str, Any]]) -> float:
    original = "\n".join(
        str(message.get("content", "") or "")
        for message in original_messages or []
        if "byte compiled context" in str(message.get("content", "") or "").lower()
    )
    distilled = "\n".join(
        str(message.get("content", "") or "")
        for message in distilled_messages or []
        if "byte compiled context" in str(message.get("content", "") or "").lower()
    )
    if not original:
        return 0.0
    return _ratio_delta(len(original), len(distilled))


def _sanitize_auxiliary_text(text: Any) -> str:
    segments = [
        segment.strip()
        for segment in _SPLIT_PATTERN.split(str(text or ""))
        if segment and segment.strip()
    ]
    safe_segments = [segment for segment in segments if not _INJECTION_PATTERN.search(segment)]
    if not safe_segments:
        return ""
    return " ".join(safe_segments)


def _distill_auxiliary_fields(
    request_kwargs: dict[str, Any],
    *,
    request_focus: str,
    target_chars: int,
    retrieval_mode: str,
    aggressive: bool,
) -> tuple[dict[str, Any], float]:
    result = deepcopy(request_kwargs)
    original_total = _request_chars(result)
    if original_total <= target_chars:
        return result, 0.0

    primary_chars = _primary_request_chars(result)
    current_aux_chars = max(0, original_total - primary_chars)
    if current_aux_chars <= 0:
        return result, 0.0

    target_aux_chars = max(160, target_chars - min(primary_chars, max(192, int(target_chars * 0.55))))
    if current_aux_chars <= target_aux_chars:
        return result, 0.0

    weighted_total = 0.0
    field_entries: list[tuple[str, str, float, int]] = []
    for field, artifact_type, weight in _DISTILLABLE_FIELD_TYPES:
        value = result.get(field)
        if value in (None, "", [], {}):
            continue
        field_chars = len(_stringify_value(value))
        if field_chars <= 0:
            continue
        weighted_total += field_chars * weight
        field_entries.append((field, artifact_type, weight, field_chars))
    if weighted_total <= 0:
        return result, 0.0

    retrieval_before = _retrieval_context_chars(result)
    remaining_budget = target_aux_chars
    for index, (field, artifact_type, weight, field_chars) in enumerate(field_entries):
        proportional = (field_chars * weight) / weighted_total
        if index == len(field_entries) - 1:
            field_budget = max(72, remaining_budget)
        else:
            field_budget = max(72, int(round(target_aux_chars * proportional)))
            field_budget = min(field_budget, remaining_budget)
        distilled_value = _distill_auxiliary_value(
            artifact_type,
            result.get(field),
            request_focus=request_focus,
            max_chars=field_budget,
            retrieval_mode=retrieval_mode,
            aggressive=aggressive,
        )
        if distilled_value in (None, "", [], {}):
            result.pop(field, None)
        else:
            result[field] = distilled_value
        remaining_budget = max(72, remaining_budget - min(field_budget, remaining_budget))

    retrieval_after = _retrieval_context_chars(result)
    retrieval_ratio = _ratio_delta(retrieval_before, retrieval_after) if retrieval_before > 0 else 0.0
    return result, retrieval_ratio


def _distill_auxiliary_value(
    artifact_type: str,
    value: Any,
    *,
    request_focus: str,
    max_chars: int,
    retrieval_mode: str,
    aggressive: bool,
) -> Any:
    if value in (None, "", [], {}):
        return value
    if retrieval_mode == "disabled" and artifact_type in {"retrieval_context", "document_context", "support_articles"}:
        return ""
    if artifact_type in {"retrieval_context", "document_context", "support_articles", "tool_result_context"}:
        value = _sanitize_auxiliary_artifact(value)
        if value in (None, "", [], {}):
            return ""
    if isinstance(value, str):
        return _compress_text_block(
            value,
            request_focus=request_focus,
            max_chars=max_chars,
            retrieval_mode=retrieval_mode,
            aggressive=aggressive,
        )

    top_k = 3 if artifact_type in {"retrieval_context", "document_context", "support_articles"} else 2
    distilled = distill_artifact_for_focus(
        artifact_type,
        value,
        query_text=request_focus,
        max_chars=max_chars,
        top_k=top_k,
    )
    if not distilled or len(distilled) >= len(_stringify_value(value)):
        distilled = summarize_artifact_payload(artifact_type, value, max_chars=max_chars)
    if aggressive and len(distilled) > max_chars:
        distilled = compact_text(distilled, max_chars=max_chars)
    return distilled


def _sanitize_auxiliary_artifact(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_auxiliary_text(value)
    if isinstance(value, list):
        sanitized_items = []
        for item in value:
            sanitized = _sanitize_auxiliary_artifact(item)
            if sanitized not in (None, "", [], {}):
                sanitized_items.append(sanitized)
        return sanitized_items
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key in {"snippet", "text", "content", "summary", "body", "chunk"}:
                cleaned = _sanitize_auxiliary_text(item)
                if cleaned in (None, "", [], {}):
                    continue
                sanitized[key] = cleaned
            else:
                sanitized[key] = item
        return sanitized
    return value


__all__ = ["distill_request_payload", "export_prompt_distillation_manifest"]
