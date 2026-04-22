import hashlib
import json
from typing import Any

from byte.adapter.runtime_state import (
    get_fingerprinter,
)
from byte.processor.fingerprint import selective_payload_fingerprint
from byte.processor.intent import extract_request_intent
from byte.processor.model_router import (
    route_request_model,
)
from byte.processor.optimization_memory import stable_digest, summarize_artifact_payload
from byte.processor.policy import policy_hint, record_policy_event
from byte.processor.pre import compile_request_context
from byte.processor.quality import extract_output_contract
from byte.processor.task_policy import resolve_task_policy
from byte.processor.tool_calls import request_tool_signature
from byte.processor.uncertainty import estimate_request_uncertainty
from byte.processor.workflow import detect_ambiguity, plan_request_workflow
from byte.similarity_evaluation.exact_match import ExactMatchEvaluation


def _resolve_memory_context(kwargs, context) -> Any:
    memory_context = kwargs.pop("byte_memory", None) or {}
    if not isinstance(memory_context, dict):
        memory_context = {"value": memory_context}
    for field in (
        "byte_repo_fingerprint",
        "byte_workspace_fingerprint",
        "byte_codebase_fingerprint",
        "byte_patch",
        "byte_test_command",
        "byte_test_result",
        "byte_lint_result",
        "byte_schema_validation",
        "byte_tool_checks",
        "byte_provider",
    ):
        if field in kwargs and field not in memory_context:
            memory_context[field] = kwargs.get(field)
    context["_byte_memory"] = memory_context
    return memory_context


def _capture_aux_context(kwargs, context) -> Any:
    aux = {}
    for field in (
        "byte_changed_hunks",
        "byte_changed_files",
        "byte_repo_snapshot",
        "byte_repo_summary",
        "byte_retrieval_context",
        "byte_document_context",
        "byte_support_articles",
        "byte_tool_result_context",
        "byte_prompt_pieces",
    ):
        value = kwargs.get(field)
        if value not in (None, "", [], {}):
            aux[field] = value
    if aux:
        context["_byte_raw_aux_context"] = aux
    return aux


def _failure_hint_for_request(chat_cache, request_kwargs, context) -> dict[str, Any]:
    memory_context = context.get("_byte_memory", {}) or {}
    provider = str(memory_context.get("provider") or memory_context.get("byte_provider") or "")
    model_name = str(request_kwargs.get("model", "") or "")
    try:
        return dict(
            chat_cache.failure_memory_hint(
                request_kwargs,
                provider=provider,
                model=model_name,
            )
            or {}
        )
    except Exception:  # pylint: disable=W0703
        return {}


def _compile_context_if_needed(chat_cache, kwargs, context, *, session=None) -> Any:
    if not getattr(chat_cache.config, "context_compiler", True):
        return kwargs
    if bool(context.get("_byte_context_compiled", False)):
        return kwargs
    _capture_aux_context(kwargs, context)
    failure_hint = _failure_hint_for_request(chat_cache, kwargs, context)
    if failure_hint:
        context["_byte_failure_hint"] = failure_hint
    session_key = _effective_session_key(kwargs, session)
    task_policy = resolve_task_policy(kwargs, chat_cache.config)
    context["_byte_task_policy"] = task_policy
    uncertainty = estimate_request_uncertainty(
        kwargs,
        chat_cache.config,
        failure_hint=failure_hint,
    )
    context["_byte_uncertainty"] = uncertainty.to_dict()
    base_max_chars = int(
        task_policy.get("context_max_chars")
        or getattr(chat_cache.config, "context_compiler_max_chars", 6000)
    )
    max_chars = base_max_chars
    if getattr(chat_cache.config, "dynamic_context_budget", True):
        if uncertainty.band == "low":
            max_chars = min(base_max_chars, uncertainty.recommended_context_chars)
        elif uncertainty.band == "high":
            max_chars = max(base_max_chars, uncertainty.recommended_context_chars)
        else:
            max_chars = uncertainty.recommended_context_chars
    compiled, stats = compile_request_context(
        kwargs,
        keep_last_messages=getattr(chat_cache.config, "context_compiler_keep_last_messages", 6),
        max_chars=max_chars,
        prompt_piece_store=getattr(chat_cache, "prompt_piece_store", None),
        prompt_module_registry=getattr(chat_cache, "prompt_module_registry", None),
        artifact_memory_store=getattr(chat_cache, "artifact_memory_store", None),
        session_delta_store=getattr(chat_cache, "session_delta_store", None),
        session_key=session_key,
        memory_scope=getattr(chat_cache, "memory_scope", "") or "",
        relevance_top_k=int(getattr(chat_cache.config, "context_compiler_relevance_top_k", 4) or 4),
        related_memory=bool(getattr(chat_cache.config, "context_compiler_related_memory", True)),
        related_min_score=float(
            getattr(chat_cache.config, "context_compiler_related_min_score", 0.18) or 0.18
        ),
        negative_context_digests=(failure_hint.get("negative_context_digests", {}) or {}),
        context_sketches=bool(getattr(chat_cache.config, "context_compiler_sketches", True)),
        focus_distillation=bool(
            getattr(chat_cache.config, "context_compiler_focus_distillation", True)
        ),
        total_aux_budget_ratio=float(
            getattr(chat_cache.config, "context_compiler_total_aux_budget_ratio", 0.65) or 0.65
        ),
        cross_note_dedupe=bool(
            getattr(chat_cache.config, "context_compiler_cross_note_dedupe", True)
        ),
        prefix_messages=bool(getattr(chat_cache.config, "native_prompt_caching", False)),
        stable_prefix=bool(getattr(chat_cache.config, "native_prompt_caching", False)),
        prompt_distillation_mode=(
            str(getattr(chat_cache.config, "prompt_distillation_mode", "disabled") or "disabled")
            if bool(getattr(chat_cache.config, "prompt_distillation", False))
            else "disabled"
        ),
        prompt_distillation_backend=str(
            getattr(chat_cache.config, "prompt_distillation_backend", "hybrid_local")
            or "hybrid_local"
        ),
        prompt_distillation_budget_ratio=float(
            getattr(chat_cache.config, "prompt_distillation_budget_ratio", 0.55) or 0.55
        ),
        prompt_distillation_min_chars=int(
            getattr(chat_cache.config, "prompt_distillation_min_chars", 512) or 512
        ),
        prompt_distillation_retrieval_mode=str(
            getattr(chat_cache.config, "prompt_distillation_retrieval_mode", "hybrid")
            or "hybrid"
        ),
        prompt_distillation_module_mode=str(
            getattr(chat_cache.config, "prompt_distillation_module_mode", "enabled") or "enabled"
        ),
        prompt_distillation_verify_shadow_rate=float(
            getattr(chat_cache.config, "prompt_distillation_verify_shadow_rate", 0.1) or 0.1
        ),
        prompt_distillation_artifact_version=str(
            getattr(chat_cache.config, "prompt_distillation_artifact_version", "byte-prompt-distill-v1")
            or "byte-prompt-distill-v1"
        ),
    )
    context["_byte_context_compile"] = stats
    context["_byte_prompt_distillation"] = dict(stats.get("prompt_distillation", {}) or {})
    context["_byte_context_compiled"] = True
    if session_key:
        context["_byte_session_key"] = session_key
    artifact_fingerprint = _artifact_fingerprint_from_request(kwargs)
    if artifact_fingerprint:
        context["_byte_artifact_fingerprint"] = artifact_fingerprint
    if stats.get("applied"):
        intent = extract_request_intent(compiled)
        record_policy_event(
            intent.route_key,
            category=intent.category,
            event="context_compiled",
            context_savings_chars=int(stats.get("saved_chars", 0) or 0),
        )
    return compiled


def _apply_request_namespaces(pre_embedding_data, request_kwargs, chat_cache, context=None) -> Any:
    """Apply optional request partitions before embedding/search."""
    if not isinstance(pre_embedding_data, str):
        return pre_embedding_data

    if chat_cache.config.context_fingerprint:
        messages = request_kwargs.get("messages", [])
        if messages and len(messages) > 1:
            fp = get_fingerprinter(chat_cache, chat_cache.config.fingerprint_window)
            pre_embedding_data = fp.enrich_pre_embedding(pre_embedding_data, messages)

    if chat_cache.config.tool_namespace:
        tool_sig = request_tool_signature(request_kwargs)
        if tool_sig:
            pre_embedding_data = f"{pre_embedding_data}||tools:{tool_sig}"

    if chat_cache.config.retrieval_namespace_fields:
        retrieval_sig = selective_payload_fingerprint(
            request_kwargs,
            chat_cache.config.retrieval_namespace_fields,
        )
        if retrieval_sig:
            pre_embedding_data = f"{pre_embedding_data}||retr:{retrieval_sig}"

    implicit_namespace_fields = {}
    for field in (
        "byte_retrieval_context",
        "byte_repo_fingerprint",
        "byte_workspace_fingerprint",
        "byte_codebase_fingerprint",
        "byte_tool_result_context",
    ):
        value = request_kwargs.pop(field, None)
        if value not in (None, "", [], {}):
            implicit_namespace_fields[field] = value
    memory_context = (context or {}).get("_byte_memory", {}) or {}
    raw_aux_context = (context or {}).get("_byte_raw_aux_context", {}) or {}
    for field in (
        "byte_retrieval_context",
        "byte_repo_fingerprint",
        "byte_workspace_fingerprint",
        "byte_codebase_fingerprint",
        "byte_tool_result_context",
    ):
        if field in implicit_namespace_fields:
            continue
        value = memory_context.get(field)
        if value not in (None, "", [], {}):
            implicit_namespace_fields[field] = value
    for field in (
        "byte_retrieval_context",
        "byte_document_context",
        "byte_support_articles",
        "byte_tool_result_context",
        "byte_repo_summary",
        "byte_repo_snapshot",
        "byte_changed_files",
        "byte_changed_hunks",
    ):
        if field in implicit_namespace_fields:
            continue
        value = raw_aux_context.get(field)
        if value not in (None, "", [], {}):
            implicit_namespace_fields[field] = value
    if implicit_namespace_fields:
        implicit_sig = selective_payload_fingerprint(
            implicit_namespace_fields,
            list(implicit_namespace_fields.keys()),
        )
        if implicit_sig:
            pre_embedding_data = f"{pre_embedding_data}||bytectx:{implicit_sig}"

    model_for_ns = request_kwargs.get("model", "")
    if (chat_cache.config.model_namespace or chat_cache.config.model_routing) and model_for_ns:
        pre_embedding_data = f"{model_for_ns}::{pre_embedding_data}"

    return pre_embedding_data


def _provider_safe_kwargs(request_kwargs) -> Any:
    return {
        key: value
        for key, value in request_kwargs.items()
        if not str(key).startswith("byte_") and not str(key).startswith("_byte_")
    }


def _output_contract_payload(context) -> Any:
    contract = context.get("_byte_output_contract", {}) or {}
    if hasattr(contract, "to_dict"):
        return contract.to_dict()
    if isinstance(contract, dict):
        return contract
    return {}


def _build_output_contract_instruction(contract) -> str:
    exact_token = str(contract.get("exact_token", "") or "").strip()
    labels = [str(item).strip() for item in (contract.get("labels") or []) if str(item).strip()]
    structured_format = str(contract.get("structured_format", "") or "").strip().lower()
    if exact_token:
        return (
            "Byte output contract: the final answer must be exactly "
            f"{exact_token} and no other text. Do not include explanations, code fences, or extra words."
        )
    if labels:
        rendered_labels = ", ".join(labels)
        return (
            "Byte output contract: the final answer must be exactly one label from "
            f"{{{rendered_labels}}} and no other text. Do not include explanations or additional formatting."
        )
    if structured_format in {"json", "yaml", "csv"}:
        return (
            "Byte output contract: return valid "
            f"{structured_format.upper()} only. Do not include markdown fences or commentary."
        )
    return ""


def _provider_request_kwargs(chat_cache, request_kwargs, context) -> Any:
    provider_kwargs = dict(request_kwargs)
    contract = _output_contract_payload(context)
    if not contract:
        contract = extract_output_contract(request_kwargs).to_dict()
    if bool(getattr(chat_cache.config, "output_contract_enforcement", True)) and bool(
        contract.get("strict")
    ):
        instruction = _build_output_contract_instruction(contract)
        messages = list(provider_kwargs.get("messages") or [])
        if instruction and messages:
            contract_message = {"role": "system", "content": instruction}
            existing = [
                str(message.get("content", "") or "")
                for message in messages[:2]
                if isinstance(message, dict) and str(message.get("role", "") or "") == "system"
            ]
            if instruction not in existing:
                if messages and str(messages[0].get("role", "") or "") == "system":
                    messages = [messages[0], contract_message, *messages[1:]]
                else:
                    messages = [contract_message, *messages]
                provider_kwargs["messages"] = messages
    return _provider_safe_kwargs(provider_kwargs)


def _repo_fingerprint_from_context(context) -> str:
    memory_context = context.get("_byte_memory", {}) or {}
    for field in (
        "byte_repo_fingerprint",
        "byte_workspace_fingerprint",
        "byte_codebase_fingerprint",
    ):
        value = memory_context.get(field)
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def _artifact_fingerprint_from_request(request_kwargs) -> Any:
    payload = {}
    for field in (
        "byte_retrieval_context",
        "byte_document_context",
        "byte_support_articles",
        "byte_repo_snapshot",
        "byte_repo_summary",
        "byte_changed_files",
        "byte_changed_hunks",
        "byte_tool_result_context",
        "byte_prompt_pieces",
    ):
        value = request_kwargs.get(field)
        if value not in (None, "", [], {}):
            payload[field] = value
    if not payload:
        return ""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]


def _effective_session_key(request_kwargs, session) -> Any:
    explicit = str(request_kwargs.get("byte_session_id", "") or "")
    if explicit:
        return explicit
    if session is not None and getattr(session, "name", None):
        return str(session.name)
    return ""


def _plan_workflow(chat_cache, request_kwargs, context) -> Any | None:
    if not getattr(chat_cache.config, "planner_enabled", True):
        return None
    if not request_kwargs.get("messages") and not request_kwargs.get(
        "byte_force_workflow_planner", False
    ):
        return None
    memory_context = context.get("_byte_memory", {}) or {}
    str(memory_context.get("provider") or memory_context.get("byte_provider") or "")
    str(request_kwargs.get("model", "") or "")
    failure_hint = dict(context.get("_byte_failure_hint", {}) or {})
    if not failure_hint:
        failure_hint = _failure_hint_for_request(chat_cache, request_kwargs, context)
    route_key = extract_request_intent(request_kwargs).route_key
    global_hint = (
        policy_hint(route_key) if getattr(chat_cache.config, "tenant_policy_learning", True) else {}
    )
    workflow_hint = {}
    try:
        workflow_hint = chat_cache.workflow_plan_hint(
            request_kwargs,
            repo_fingerprint=_repo_fingerprint_from_context(context),
            artifact_fingerprint=str(context.get("_byte_artifact_fingerprint", "") or ""),
        )
    except Exception:  # pylint: disable=W0703
        workflow_hint = {}
    if workflow_hint:
        merged_hint = dict(global_hint)
        merged_hint.update({k: v for k, v in workflow_hint.items() if v not in (None, "", [], {})})
        global_hint = merged_hint
    patch_candidate = None
    if getattr(chat_cache.config, "delta_generation", True):
        try:
            patch_candidate = chat_cache.suggest_patch_pattern(
                request_kwargs,
                repo_fingerprint=_repo_fingerprint_from_context(context),
            )
        except Exception:  # pylint: disable=W0703
            patch_candidate = None
    ambiguity = detect_ambiguity(
        request_kwargs,
        min_chars=getattr(chat_cache.config, "ambiguity_min_chars", 24),
        context_hints=context,
    )
    decision = plan_request_workflow(
        request_kwargs,
        chat_cache.config,
        ambiguity=ambiguity,
        failure_hint=failure_hint,
        global_hint=global_hint,
        patch_candidate=patch_candidate,
    )
    context["_byte_workflow_plan"] = decision
    context["_byte_ambiguity"] = ambiguity
    if decision.route_preference:
        request_kwargs["byte_route_preference"] = decision.route_preference
    return decision


def _maybe_route_request(chat_cache, request_kwargs, context) -> Any:
    decision = route_request_model(request_kwargs, chat_cache.config)
    if decision is not None:
        context["_byte_model_route"] = decision
    return decision


def _semantic_cache_allowed(chat_cache, request_kwargs, context) -> bool:
    allowed_categories = getattr(chat_cache.config, "semantic_allowed_categories", []) or []
    if not allowed_categories:
        return True
    # The allow-list is only meant to guard broad semantic reuse. Exact-match
    # layers (including normalized exact caches) should still be able to hit/save.
    if isinstance(getattr(chat_cache, "similarity_evaluation", None), ExactMatchEvaluation):
        return True
    cached_intent = context.get("_byte_intent")
    if cached_intent is None:
        cached_intent = extract_request_intent(request_kwargs)
        context["_byte_intent"] = cached_intent
    return cached_intent.category in set(allowed_categories)


def _negative_context_metadata(context) -> dict[str, Any]:
    raw_aux = context.get("_byte_raw_aux_context", {}) or {}
    negative_digests = {}
    negative_summaries = {}
    for field, value in raw_aux.items():
        artifact_type = str(field).replace("byte_", "", 1)
        if artifact_type not in {
            "retrieval_context",
            "document_context",
            "support_articles",
            "tool_result_context",
            "changed_hunks",
            "changed_files",
        }:
            continue
        if isinstance(value, list):
            items = value[:6]
        else:
            items = [value]
        digests = []
        for item in items:
            digest = stable_digest(item)
            digests.append(digest)
            negative_summaries[digest] = summarize_artifact_payload(
                artifact_type, item, max_chars=140
            )
        if digests:
            negative_digests[artifact_type] = digests
    if not negative_digests:
        return {}
    return {
        "negative_context_digests": negative_digests,
        "negative_context_summaries": negative_summaries,
    }


__all__ = [name for name in globals() if not name.startswith("__")]
