"""Typed section dataclasses and environment parsing helpers for Byte configuration."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Union, get_args, get_origin

from byte.utils.error import CacheError


@dataclass
class ObservabilityConfig:
    log_time_func: Callable[[str, float], None] | None = None
    enable_token_counter: bool = True
    disable_report: bool = False
    telemetry_enabled: bool = False
    telemetry_tracer_name: str = "byteai.cache"
    telemetry_meter_name: str = "byteai.cache"
    telemetry_attributes: dict[str, Any] | None = None


@dataclass
class CacheConfig:
    similarity_threshold: float = 0.8
    prompts: list[str] | None = None
    template: str | None = None
    auto_flush: int = 20
    input_summary_len: int | None = None
    context_len: int | None = None
    skip_list: list[str] | None = None
    data_check: bool = False
    ttl: int | float | None = None
    model_namespace: bool = False
    embedding_cache_size: int = 10000
    adaptive_threshold: bool = False
    target_hit_rate: float = 0.5
    dedup_threshold: float = 0.0
    context_fingerprint: bool = False
    fingerprint_window: int = 3
    tool_namespace: bool = False
    retrieval_namespace_fields: list[str] | None = None
    semantic_min_token_overlap: float = 0.0
    semantic_max_length_ratio: float | None = None
    semantic_enforce_canonical_match: bool = False
    tiered_cache: bool = False
    tier1_max_size: int = 1000
    tier1_promotion_threshold: int = 2
    tier1_promotion_window_s: float = 300.0
    tier1_promote_on_write: bool = True
    async_write_back: bool = False
    async_write_back_queue_size: int = 10000
    cache_admission_min_score: float = 0.0
    cache_latency_guard: bool = True
    cache_latency_min_samples: int = 8
    cache_latency_probe_samples: int = 32
    cache_latency_p95_multiplier: float = 1.15
    cache_latency_min_hits: int = 4
    cache_latency_force_miss_samples: int = 64
    cache_latency_min_hit_rate: float = 0.1
    native_prompt_caching: bool = True
    native_prompt_cache_min_chars: int = 1200
    native_prompt_cache_ttl: str | None = None
    semantic_allowed_categories: list[str] | None = None
    vcache_enabled: bool = False               # vCache per-prompt learned threshold 
    vcache_delta: float = 0.05                 # max tolerated error rate; default 5%
    vcache_min_observations: int = 10          # cold-start: use global threshold below this count
    vcache_cold_fallback_threshold: float = 0.80
    dual_threshold_reference_mode: bool = False  # Stage 2 reference lane 
    # Cost-aware eviction . Choices: "LRU" (default) | "COST_AWARE".
    eviction_policy: str = "LRU"
    # LSH prefilter  — near-duplicate MinHash-LSH gate before vector search.
    lsh_prefilter_enabled: bool = False
    lsh_num_perm: int = 128
    lsh_threshold: float = 0.6
    lsh_shingle_k: int = 5


@dataclass
class RoutingConfig:
    model_routing: bool = False
    routing_cheap_model: str | None = None
    routing_expensive_model: str | None = None
    routing_tool_model: str | None = None
    routing_default_model: str | None = None
    routing_coder_model: str | None = None
    routing_reasoning_model: str | None = None
    routing_verifier_model: str | None = None
    routing_long_prompt_chars: int = 1200
    routing_multi_turn_threshold: int = 6
    routing_model_aliases: dict[str, Any] | None = None
    routing_fallbacks: dict[str, list[str]] | None = None
    routing_strategy: str = "priority"
    routing_retry_attempts: int = 0
    routing_retry_backoff_ms: float = 0.0
    routing_cooldown_seconds: float = 15.0
    routing_provider_keys: dict[str, Any] | None = None
    routing_max_cheap_labels: int = 6
    routing_max_cheap_fields: int = 6
    cheap_consensus_enabled: bool = False
    cheap_consensus_models: list[str] | None = None
    cheap_consensus_min_score: float = 0.5
    routing_verifier_enabled: bool = False
    routing_verify_cheap_responses: bool = True
    routing_verify_min_score: float = 0.75
    routing_grey_zone_min_score: float = 0.55
    routing_adaptive: bool = False
    routing_adaptive_min_samples: int = 6
    routing_adaptive_quality_floor: float = 0.75
    speculative_routing: bool = False
    speculative_max_parallel: int = 2
    # Byte Smart Router — multi-signal complexity analysis for model tier selection
    route_llm_enabled: bool = True
    route_llm_threshold: float = 0.5
    route_llm_seed_path: str = ""   # optional path to labelled JSON seed examples
    # Byte Cascade — confidence-gated escalation from cheap to strong tier
    cascade_escalation_enabled: bool = True
    cascade_confidence_threshold: float = 0.55


@dataclass
class QualityConfig:
    evidence_verification: bool = True
    evidence_min_support: float = 0.35
    evidence_structured_min_support: float = 0.78
    evidence_summary_min_support: float = 0.28
    response_repair: bool = True
    unique_output_guard: bool = True
    context_only_unique_prompts: bool = True
    reasoning_reuse: bool = True
    coding_reasoning_shortcut: bool = False
    reasoning_repair: bool = True
    grounded_context_only: bool = True
    grounded_context_categories: list[str] | None = None
    output_contract_enforcement: bool = True
    ambiguity_detection: bool = True
    ambiguity_min_chars: int = 24
    llm_equivalence_enabled: bool = False                        # LLM-based ambiguity resolution 
    llm_equivalence_ambiguity_band_low: float = 0.70             # lower bound of ambiguity band
    llm_equivalence_ambiguity_band_high: float = 0.85            # upper bound of ambiguity band
    llm_equivalence_model: str = ""                              # defaults to routing_cheap_model at runtime


@dataclass
class ContextCompilerConfig:
    context_compiler: bool = True
    context_compiler_keep_last_messages: int = 6
    context_compiler_max_chars: int = 6000
    context_compiler_relevance_top_k: int = 4
    context_compiler_related_memory: bool = True
    context_compiler_related_min_score: float = 0.18
    context_compiler_sketches: bool = True
    context_compiler_focus_distillation: bool = True
    context_compiler_total_aux_budget_ratio: float = 0.65
    context_compiler_cross_note_dedupe: bool = True
    dynamic_context_budget: bool = True
    context_budget_low_risk_chars: int = 2200
    context_budget_medium_risk_chars: int = 4800
    context_budget_high_risk_chars: int = 7600
    negative_context_memory: bool = True
    intent_context_filtering_enabled: bool = False  # intent-driven token pruning 
    intent_context_budget_ratio: float = 0.6        # keep top 60% of context tokens by intent relevance
    intent_cache_intent_labels: bool = True         # store intent label in context alongside answer


@dataclass
class PromptDistillationConfig:
    prompt_distillation: bool = True
    prompt_distillation_mode: str = "enabled"
    prompt_distillation_backend: str = "hybrid_local"
    prompt_distillation_budget_ratio: float = 0.55
    prompt_distillation_min_chars: int = 512
    prompt_distillation_retrieval_mode: str = "hybrid"
    prompt_distillation_module_mode: str = "enabled"
    prompt_distillation_verify_shadow_rate: float = 0.1
    prompt_distillation_artifact_version: str = "byte-prompt-distill-v1"


@dataclass
class MemoryConfig:
    intent_memory: bool = True
    memory_scope: str | None = None
    tool_result_ttl: int | float | None = None
    execution_memory: bool = True
    verified_reuse_for_coding: bool = False
    verified_reuse_for_all: bool = False
    reasoning_memory: bool = True
    delta_generation: bool = True
    planner_enabled: bool = True
    planner_allow_verified_short_circuit: bool = True
    failure_memory: bool = True
    tenant_policy_learning: bool = True
    task_policies: dict[str, Any] | None = None
    memory_max_entries: int = 2000
    memory_embedding_preview_dims: int = 32


@dataclass
class BudgetConfig:
    budget_strategy: str = "balanced"
    budget_quality_floor: float = 0.75
    budget_latency_target_ms: float = 1200.0


@dataclass
class SecurityConfig:
    compliance_profile: str | None = None
    security_mode: bool = False
    security_encryption_key: str | None = None
    security_redact_logs: bool = True
    security_redact_reports: bool = False
    security_redact_memory: bool = False
    security_require_admin_auth: bool = False
    security_admin_token: str | None = None
    security_audit_log_path: str | None = None
    security_export_root: str | None = None
    security_require_https: bool = False
    security_trust_proxy_headers: bool = False
    security_disable_cache_file_endpoint: bool = False
    security_encrypt_artifacts: bool = False
    security_allow_provider_host_override: bool = False
    security_allowed_egress_hosts: list[str] | None = None
    security_max_request_bytes: int = 1_048_576
    security_max_upload_bytes: int = 16_777_216
    security_max_archive_bytes: int = 33_554_432
    security_max_archive_members: int = 256
    security_rate_limit_public_per_minute: int = 0
    security_rate_limit_admin_per_minute: int = 0
    security_max_inflight_public: int = 0
    security_max_inflight_admin: int = 0


@dataclass
class CompressionConfig:
    kv_codec: str = "disabled"
    kv_bits: int = 8
    kv_hot_window_ratio: float = 0.25
    vector_codec: str = "disabled"
    vector_bits: int = 8
    compression_mode: str = "shadow"
    compression_backend_policy: str = "auto"
    compression_verify_shadow_rate: float = 0.1


@dataclass
class TrustConfig:
    trust_mode: str = "guarded"
    query_risk_mode: str = "hybrid"
    confidence_mode: str = "calibrated"
    confidence_backend: str = "hybrid"
    conformal_target_coverage: float = 0.95
    reuse_band_policy: str = "adaptive"  # Deprecated: use vcache_enabled / vcache_delta (vCache, )
    conformal_mode: str = "guarded"
    deterministic_execution: bool = True
    deterministic_contract_mode: str = "enforced"
    semantic_cache_verifier_mode: str = "hybrid"
    semantic_cache_promotion_mode: str = "shadow"
    novelty_threshold: float = 0.62  # Deprecated: use vcache_delta (vCache, )
    prompt_module_mode: str = "enabled"
    calibration_artifact_version: str = "byte-trust-v2"
    benchmark_contract_version: str = "byte-benchmark-v2"


@dataclass
class IntegrationConfig:
    mcp_timeout_s: float = 30.0
    h2o_enabled: bool = False
    h2o_heavy_ratio: float = 0.1
    h2o_recent_ratio: float = 0.1


_DEFAULT_GROUNDED_CONTEXT_CATEGORIES = [
    "classification",
    "summarization",
    "extraction",
    "comparison",
    "question_answer",
    "instruction",
    "code_fix",
    "code_refactor",
    "test_generation",
    "code_explanation",
    "documentation",
]

_SECTION_ORDER = (
    "observability",
    "cache",
    "routing",
    "quality",
    "context_compiler",
    "prompt_distillation",
    "memory",
    "budget",
    "security",
    "compression",
    "trust",
    "integrations",
)

_SECTION_TYPES = {
    "observability": ObservabilityConfig,
    "cache": CacheConfig,
    "routing": RoutingConfig,
    "quality": QualityConfig,
    "context_compiler": ContextCompilerConfig,
    "prompt_distillation": PromptDistillationConfig,
    "memory": MemoryConfig,
    "budget": BudgetConfig,
    "security": SecurityConfig,
    "compression": CompressionConfig,
    "trust": TrustConfig,
    "integrations": IntegrationConfig,
}
_SECTION_ATTRS = {
    "observability": "_observability",
    "cache": "_cache",
    "routing": "_routing",
    "quality": "_quality",
    "context_compiler": "_context_compiler_config",
    "prompt_distillation": "_prompt_distillation_config",
    "memory": "_memory",
    "budget": "_budget",
    "security": "_security",
    "compression": "_compression",
    "trust": "_trust",
    "integrations": "_integrations",
}

_FIELD_TO_SECTION: dict[str, str] = {}
_FIELD_META: dict[str, Any] = {}
for _section_name, _section_type in _SECTION_TYPES.items():
    for _field in fields(_section_type):
        _FIELD_TO_SECTION[_field.name] = _section_name
        _FIELD_META[_field.name] = _field

_LEGACY_FIELD_ORDER = [
    "log_time_func",
    "similarity_threshold",
    "prompts",
    "template",
    "auto_flush",
    "enable_token_counter",
    "input_summary_len",
    "context_len",
    "skip_list",
    "data_check",
    "disable_report",
    "ttl",
    "model_namespace",
    "embedding_cache_size",
    "adaptive_threshold",
    "target_hit_rate",
    "dedup_threshold",
    "context_fingerprint",
    "fingerprint_window",
    "tool_namespace",
    "retrieval_namespace_fields",
    "semantic_min_token_overlap",
    "semantic_max_length_ratio",
    "semantic_enforce_canonical_match",
    "tiered_cache",
    "tier1_max_size",
    "tier1_promotion_threshold",
    "tier1_promotion_window_s",
    "tier1_promote_on_write",
    "async_write_back",
    "async_write_back_queue_size",
    "intent_memory",
    "memory_scope",
    "tool_result_ttl",
    "model_routing",
    "routing_cheap_model",
    "routing_expensive_model",
    "routing_tool_model",
    "routing_default_model",
    "routing_coder_model",
    "routing_reasoning_model",
    "routing_verifier_model",
    "routing_long_prompt_chars",
    "routing_multi_turn_threshold",
    "routing_model_aliases",
    "routing_fallbacks",
    "routing_strategy",
    "routing_retry_attempts",
    "routing_retry_backoff_ms",
    "routing_cooldown_seconds",
    "routing_provider_keys",
    "routing_max_cheap_labels",
    "routing_max_cheap_fields",
    "cheap_consensus_enabled",
    "cheap_consensus_models",
    "cheap_consensus_min_score",
    "routing_verifier_enabled",
    "evidence_verification",
    "evidence_min_support",
    "evidence_structured_min_support",
    "evidence_summary_min_support",
    "routing_verify_cheap_responses",
    "routing_verify_min_score",
    "routing_grey_zone_min_score",
    "routing_adaptive",
    "routing_adaptive_min_samples",
    "routing_adaptive_quality_floor",
    "response_repair",
    "cache_admission_min_score",
    "execution_memory",
    "verified_reuse_for_coding",
    "verified_reuse_for_all",
    "unique_output_guard",
    "context_only_unique_prompts",
    "reasoning_reuse",
    "coding_reasoning_shortcut",
    "reasoning_memory",
    "reasoning_repair",
    "grounded_context_only",
    "grounded_context_categories",
    "output_contract_enforcement",
    "delta_generation",
    "ambiguity_detection",
    "ambiguity_min_chars",
    "context_compiler",
    "context_compiler_keep_last_messages",
    "context_compiler_max_chars",
    "context_compiler_relevance_top_k",
    "context_compiler_related_memory",
    "context_compiler_related_min_score",
    "context_compiler_sketches",
    "context_compiler_focus_distillation",
    "context_compiler_total_aux_budget_ratio",
    "context_compiler_cross_note_dedupe",
    "dynamic_context_budget",
    "context_budget_low_risk_chars",
    "context_budget_medium_risk_chars",
    "context_budget_high_risk_chars",
    "negative_context_memory",
    "prompt_distillation",
    "prompt_distillation_mode",
    "prompt_distillation_backend",
    "prompt_distillation_budget_ratio",
    "prompt_distillation_min_chars",
    "prompt_distillation_retrieval_mode",
    "prompt_distillation_module_mode",
    "prompt_distillation_verify_shadow_rate",
    "prompt_distillation_artifact_version",
    "planner_enabled",
    "planner_allow_verified_short_circuit",
    "failure_memory",
    "tenant_policy_learning",
    "task_policies",
    "budget_strategy",
    "budget_quality_floor",
    "budget_latency_target_ms",
    "cache_latency_guard",
    "cache_latency_min_samples",
    "cache_latency_probe_samples",
    "cache_latency_p95_multiplier",
    "cache_latency_min_hits",
    "cache_latency_force_miss_samples",
    "cache_latency_min_hit_rate",
    "native_prompt_caching",
    "native_prompt_cache_min_chars",
    "native_prompt_cache_ttl",
    "speculative_routing",
    "speculative_max_parallel",
    "semantic_allowed_categories",
    "memory_max_entries",
    "memory_embedding_preview_dims",
    "compliance_profile",
    "security_mode",
    "security_encryption_key",
    "security_redact_logs",
    "security_redact_reports",
    "security_redact_memory",
    "security_require_admin_auth",
    "security_admin_token",
    "security_audit_log_path",
    "security_export_root",
    "security_require_https",
    "security_trust_proxy_headers",
    "security_disable_cache_file_endpoint",
    "security_encrypt_artifacts",
    "security_allow_provider_host_override",
    "security_allowed_egress_hosts",
    "security_max_request_bytes",
    "security_max_upload_bytes",
    "security_max_archive_bytes",
    "security_max_archive_members",
    "security_rate_limit_public_per_minute",
    "security_rate_limit_admin_per_minute",
    "security_max_inflight_public",
    "security_max_inflight_admin",
    "kv_codec",
    "kv_bits",
    "kv_hot_window_ratio",
    "vector_codec",
    "vector_bits",
    "compression_mode",
    "compression_backend_policy",
    "compression_verify_shadow_rate",
    "telemetry_enabled",
    "telemetry_tracer_name",
    "telemetry_meter_name",
    "telemetry_attributes",
    "h2o_enabled",
    "h2o_heavy_ratio",
    "h2o_recent_ratio",
    "vcache_enabled",
    "vcache_delta",
    "vcache_min_observations",
    "vcache_cold_fallback_threshold",
    "dual_threshold_reference_mode",
    "llm_equivalence_enabled",
    "llm_equivalence_ambiguity_band_low",
    "llm_equivalence_ambiguity_band_high",
    "llm_equivalence_model",
    "intent_context_filtering_enabled",
    "intent_context_budget_ratio",
    "intent_cache_intent_labels",
]


def _split_legacy_args(args: tuple[Any, ...]) -> dict[str, Any]:
    if len(args) > len(_LEGACY_FIELD_ORDER):
        raise TypeError(
            f"Config() takes at most {len(_LEGACY_FIELD_ORDER)} positional arguments "
            f"but {len(args)} were given"
        )
    return dict(zip(_LEGACY_FIELD_ORDER[: len(args)], args, strict=True))


def _normalize_field_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        non_none = [item for item in get_args(annotation) if item is not type(None)]
        if len(non_none) == 1:
            return _normalize_field_annotation(non_none[0])
        return annotation
    return annotation


def _parse_bool(raw: str) -> bool:
    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise CacheError(f"Invalid boolean env override: {raw}")


def _parse_number(raw: str) -> int | float:
    text = str(raw).strip()
    if any(token in text.lower() for token in (".", "e")):
        return float(text)
    return int(text)


def _coerce_env_value(field_name: str, raw: str, current: Any) -> Any:
    annotation = _normalize_field_annotation(_FIELD_META[field_name].type)
    text = str(raw).strip()
    if text.lower() in {"none", "null"}:
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is bool or isinstance(current, bool):
        return _parse_bool(text)
    if annotation is int or (isinstance(current, int) and not isinstance(current, bool)):
        return int(text)
    if annotation is float or isinstance(current, float):
        return float(text)
    if annotation is str or isinstance(current, str):
        return text
    if origin is list:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in text.split(",") if item.strip()]
    if origin is dict:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise CacheError(f"Expected JSON object env override for {field_name}")
        return parsed
    if origin in {set, frozenset}:
        return {item.strip() for item in text.split(",") if item.strip()}
    if annotation is Callable:
        return current
    if args and set(args).issubset({int, float}):
        return _parse_number(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _normalize_string_list(values: list[Any] | None, *, lower: bool = False) -> list[str]:
    result: list[str] = []
    for item in values or []:
        text = str(item).strip()
        if not text:
            continue
        result.append(text.lower() if lower else text)
    return result
