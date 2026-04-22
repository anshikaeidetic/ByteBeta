"""Provider/system helpers shared by benchmark executable systems."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from byte import Cache, Config
from byte.adapter.api import init_cache
from byte.adapter.prompt_cache_bridge import apply_native_prompt_cache
from byte.benchmarking._optional_runtime import load_chat_backend
from byte.benchmarking.contracts import BenchmarkItem
from byte.processor.pre import last_content, normalized_last_content

_REUSE_ONLY_REASONS = {"reasoning_memory_reuse"}
_SHORTCUT_REASONS = {
    "deterministic_reasoning",
    "contract_shortcut",
    "coding_contract_shortcut",
    "coding_analysis_shortcut",
    "grounded_context_shortcut",
    "clarification_required",
    "verified_patch_reuse",
}


@dataclass
class SystemSpec:
    """Static configuration for one provider/system benchmark lane."""

    system: str
    baseline_type: str
    provider: str
    model: str
    label: str


def default_system_specs(providers: list[str]) -> list[SystemSpec]:
    """Return the default benchmark systems for each requested provider."""
    specs: list[SystemSpec] = []
    for provider in providers:
        model = provider_default_model(provider)
        specs.extend(
            [
                SystemSpec("direct", "direct", provider, model, f"{provider}_direct"),
                SystemSpec(
                    "native_cache",
                    "provider_native_cache",
                    provider,
                    model,
                    f"{provider}_native_cache",
                ),
                SystemSpec("byte", "byte", provider, model, f"{provider}_byte"),
            ]
        )
        if provider == "openai":
            specs.extend(
                [
                    SystemSpec(
                        "langchain_redis",
                        "langchain_redis",
                        provider,
                        model,
                        "openai_langchain_redis",
                    ),
                    SystemSpec(
                        "embedding_similarity",
                        "embedding_similarity",
                        provider,
                        model,
                        "openai_embedding_similarity",
                    ),
                ]
            )
    return specs


def provider_default_model(provider: str) -> str:
    """Return the canonical benchmark model for a provider track."""
    normalized = str(provider or "").strip().lower()
    if normalized == "openai":
        return "gpt-4o-mini"
    if normalized == "anthropic":
        return "claude-3-5-sonnet-latest"
    if normalized == "deepseek":
        return "deepseek-chat"
    raise ValueError(f"Unsupported provider: {provider}")


def provider_key_env(provider: str) -> str:
    """Return the environment variable that carries provider credentials."""
    normalized = str(provider or "").strip().lower()
    if normalized == "openai":
        return "OPENAI_API_KEY"
    if normalized == "anthropic":
        return "ANTHROPIC_API_KEY"
    if normalized == "deepseek":
        return "DEEPSEEK_API_KEY"
    raise ValueError(f"Unsupported provider: {provider}")


def chat_backend(provider: str) -> Any:
    """Resolve the provider benchmark backend."""
    return load_chat_backend(provider)


def direct_request(provider: str, request_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Execute a direct provider request with no Byte caching layer."""
    backend = chat_backend(provider)
    return dict(backend._llm_handler(**request_kwargs) or {})


def native_cache_request(provider: str, request_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Execute a provider-native prompt-cache request for baseline comparison."""
    cfg = Config(native_prompt_caching=True, native_prompt_cache_min_chars=0)
    cached_kwargs = apply_native_prompt_cache(provider, request_kwargs, cfg)
    backend = chat_backend(provider)
    response = dict(backend._llm_handler(**cached_kwargs) or {})
    response.setdefault("byte_native_cache_mode", native_cache_mode(provider))
    return response


def byte_request(provider: str, request_kwargs: dict[str, Any], *, cache_obj: Cache | None) -> dict[str, Any]:
    """Execute a request through the Byte runtime cache path."""
    backend = chat_backend(provider)
    return dict(backend.create(cache_obj=cache_obj, **request_kwargs) or {})


def request_kwargs(
    item: BenchmarkItem,
    *,
    provider: str,
    model: str,
    include_visible_context: bool,
) -> dict[str, Any]:
    """Render one workload item into backend request kwargs."""
    payload = copy.deepcopy(item.input_payload)
    payload["model"] = model
    payload["temperature"] = float(payload.get("temperature", 0.0) or 0.0)
    payload["max_tokens"] = int(payload.get("max_tokens", 32) or 32)
    messages = list(payload.get("messages") or [])
    context_payload = dict(payload.pop("context_payload", {}) or {})
    if context_payload:
        payload["messages"] = (
            visible_messages(messages, context_payload) if include_visible_context else messages
        )
        payload.update(context_payload)
    else:
        payload["messages"] = messages
    payload["byte_memory"] = {
        "provider": provider,
        "metadata": {
            "family": item.family,
            "scenario": item.scenario,
            "seed_id": item.seed_id,
            "variant_id": item.variant_id,
            "benchmark_item_id": item.item_id,
        },
    }
    return payload


def visible_messages(messages: list[dict[str, Any]], context_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize user-visible context for direct/native baselines."""
    if not messages:
        messages = [{"role": "user", "content": ""}]
    visible = copy.deepcopy(messages)
    rendered = render_context_payload(context_payload)
    if rendered:
        last = dict(visible[-1])
        last["content"] = f"{str(last.get('content', '')).strip()}\n\nContext:\n{rendered}".strip()
        visible[-1] = last
    return visible


def render_context_payload(context_payload: dict[str, Any]) -> str:
    """Render context payload fields as readable inline context."""
    pieces = []
    for key, value in context_payload.items():
        if value in (None, "", [], {}):
            continue
        pieces.append(f"{key}: {value}")
    return "\n".join(pieces)


def prompt_signature(request_kwargs: dict[str, Any]) -> str:
    """Create a stable prompt signature for benchmark baselines."""
    messages = request_kwargs.get("messages") or []
    if messages:
        return "\n".join(str(message.get("content", "") or "") for message in messages)
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    return ""


def extract_text(response: dict[str, Any]) -> str:
    """Extract plain assistant text from an OpenAI-style response payload."""
    choices = list(response.get("choices") or [])
    if not choices:
        return str(response.get("text", "") or "")
    message = dict(choices[0].get("message", {}) or {})
    content = message.get("content")
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "") or part.get("content", "") or "")
            if isinstance(part, dict)
            else str(part or "")
            for part in content
        ).strip()
    return str(content or "").strip()


def usage_fields(usage: Any) -> dict[str, int]:
    """Normalize provider usage payloads into benchmark token fields."""
    if not usage:
        return {
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "uncached_prompt_tokens": 0,
            "completion_tokens": 0,
        }
    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details", {}) or {}
        cached_prompt_tokens = int(details.get("cached_tokens", 0) or 0)
        hit_tokens = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        miss_tokens = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
    else:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached_prompt_tokens = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        hit_tokens = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        miss_tokens = int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0)
    if hit_tokens or miss_tokens:
        cached_prompt_tokens = min(hit_tokens, prompt_tokens)
        uncached_prompt_tokens = min(miss_tokens, max(prompt_tokens - cached_prompt_tokens, 0))
    else:
        cached_prompt_tokens = min(cached_prompt_tokens, prompt_tokens)
        uncached_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    return {
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def estimate_cost(provider: str, model: str, usage: dict[str, int]) -> float:
    """Estimate benchmark cost from normalized usage fields and static pricing."""
    pricing = pricing_entry(provider, model)
    if not pricing:
        return 0.0
    return round(
        (usage["uncached_prompt_tokens"] / 1_000_000) * float(pricing["input"])
        + (usage["cached_prompt_tokens"] / 1_000_000) * float(pricing.get("cached_input", pricing["input"]))
        + (usage["completion_tokens"] / 1_000_000) * float(pricing["output"]),
        8,
    )


def pricing_entry(provider: str, model: str) -> dict[str, float] | None:
    """Return the benchmark pricing card for a provider/model pair."""
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip().lower()
    if normalized_provider == "deepseek":
        return {"input": 0.28, "cached_input": 0.028, "output": 0.42}
    if normalized_provider == "openai":
        if normalized_model.startswith("text-embedding-3-small"):
            return {"input": 0.02, "cached_input": 0.02, "output": 0.0}
        return {"input": 0.15, "cached_input": 0.075, "output": 0.60}
    if normalized_provider == "anthropic":
        return {"input": 3.00, "cached_input": 3.00, "output": 15.00}
    return None


def served_via(response: dict[str, Any]) -> str:
    """Normalize response metadata into the benchmark served-via taxonomy."""
    if not bool(response.get("byte", False)):
        return "upstream"
    reason = str(response.get("byte_reason", "") or "")
    if reason in _REUSE_ONLY_REASONS:
        return "reuse"
    if reason in _SHORTCUT_REASONS:
        return "local_compute"
    return "reuse"


def reuse_confidence(response: dict[str, Any], served_via_name: str) -> float:
    """Return calibrated reuse confidence when present, else benchmark defaults."""
    trust = dict(response.get("byte_trust", {}) or {})
    if "calibrated_confidence" in trust:
        return round(float(trust.get("calibrated_confidence", 0.0) or 0.0), 4)
    reasoning = dict(response.get("byte_reasoning", {}) or {})
    if "confidence" in reasoning:
        return round(float(reasoning.get("confidence", 0.0) or 0.0), 4)
    if served_via_name == "reuse":
        return 0.93
    if served_via_name == "local_compute":
        return 0.20
    return 0.05


def native_cache_mode(provider: str) -> str:
    """Describe how a provider exposes native prompt caching."""
    if str(provider or "").strip().lower() == "deepseek":
        return "provider_automatic"
    return "request_hint"


def prompt_distillation_tokens(payload: dict[str, Any], *, key: str, fallback: int) -> int:
    """Extract prompt-distillation token counts with benchmark-safe fallback."""
    value = payload.get(key)
    if value in (None, ""):
        return int(fallback or 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return int(fallback or 0)


def ratio_delta(baseline: int, observed: int) -> float:
    """Return a normalized reduction ratio for prompt/cost deltas."""
    if int(baseline or 0) <= 0:
        return 0.0
    return round((float(baseline) - float(observed)) / float(baseline), 4)


def reuse_decision_correct(
    actual_reuse: bool,
    fallback_taken: bool,
    reuse_safe: bool,
    must_fallback: bool,
) -> bool:
    """Judge whether the system made the correct reuse/fallback decision."""
    if actual_reuse:
        return reuse_safe
    if fallback_taken:
        return must_fallback or not reuse_safe
    return True


def base_cache_config(scope: str) -> Config:
    """Return the standard Byte benchmark cache configuration."""
    return Config(
        enable_token_counter=False,
        embedding_cache_size=30000,
        tiered_cache=True,
        tier1_max_size=4096,
        tier1_promote_on_write=True,
        async_write_back=True,
        memory_scope=scope,
        intent_memory=True,
        execution_memory=True,
        model_namespace=True,
        tool_namespace=True,
        context_fingerprint=True,
        verified_reuse_for_coding=True,
        verified_reuse_for_all=True,
        unique_output_guard=True,
        context_only_unique_prompts=True,
        grounded_context_only=True,
        coding_reasoning_shortcut=True,
        output_contract_enforcement=True,
        delta_generation=True,
        context_compiler=True,
        context_compiler_keep_last_messages=6,
        context_compiler_max_chars=7600,
        context_compiler_related_memory=True,
        context_compiler_related_min_score=0.18,
        context_compiler_sketches=True,
        context_compiler_focus_distillation=True,
        context_compiler_total_aux_budget_ratio=0.65,
        context_compiler_cross_note_dedupe=True,
        dynamic_context_budget=True,
        negative_context_memory=True,
        ambiguity_detection=True,
        planner_enabled=True,
        failure_memory=True,
        tenant_policy_learning=True,
        native_prompt_caching=True,
        native_prompt_cache_min_chars=1200,
        evidence_verification=True,
        adaptive_threshold=True,
        cache_admission_min_score=0.1,
        cache_latency_guard=True,
        cache_latency_probe_samples=32,
        cache_latency_min_hits=4,
        cache_latency_force_miss_samples=64,
        budget_quality_floor=0.82,
        budget_latency_target_ms=900.0,
        trust_mode="guarded",
        query_risk_mode="hybrid",
        confidence_mode="calibrated",
        confidence_backend="hybrid",
        reuse_band_policy="adaptive",
        conformal_mode="guarded",
        conformal_target_coverage=0.95,
        deterministic_execution=True,
        deterministic_contract_mode="enforced",
        semantic_cache_verifier_mode="hybrid",
        semantic_cache_promotion_mode="verified",
        novelty_threshold=0.62,
        prompt_module_mode="enabled",
        prompt_distillation=True,
        prompt_distillation_mode="guarded",
        prompt_distillation_backend="hybrid_local",
        prompt_distillation_budget_ratio=0.48,
        prompt_distillation_min_chars=512,
        prompt_distillation_retrieval_mode="hybrid",
        prompt_distillation_module_mode="enabled",
        prompt_distillation_verify_shadow_rate=0.1,
        prompt_distillation_artifact_version="byte-prompt-distill-v1",
        calibration_artifact_version="byte-trust-v2",
        benchmark_contract_version="byte-benchmark-v2",
        semantic_min_token_overlap=0.18,
        semantic_max_length_ratio=1.4,
        semantic_enforce_canonical_match=True,
        semantic_allowed_categories=[
            "instruction",
            "support_classification",
            "document_classification",
            "document_extraction",
            "question_answer",
            "code_explanation",
            "code_fix",
            "test_generation",
        ],
        task_policies={"*": {"native_prompt_cache": True}},
    )


def begin_byte_phase(cache_obj: Cache, *, cache_dir: str, scope: str) -> None:
    """Initialize a temporary benchmark cache for Byte system execution."""
    config = base_cache_config(scope)
    init_cache(
        mode="hybrid",
        data_dir=cache_dir,
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=config,
        exact_config=config,
        normalized_config=config,
        semantic_config=config,
    )


__all__ = [
    "SystemSpec",
    "base_cache_config",
    "begin_byte_phase",
    "byte_request",
    "chat_backend",
    "default_system_specs",
    "direct_request",
    "estimate_cost",
    "extract_text",
    "native_cache_mode",
    "native_cache_request",
    "pricing_entry",
    "prompt_distillation_tokens",
    "prompt_signature",
    "provider_default_model",
    "provider_key_env",
    "ratio_delta",
    "request_kwargs",
    "reuse_confidence",
    "reuse_decision_correct",
    "served_via",
    "usage_fields",
]
