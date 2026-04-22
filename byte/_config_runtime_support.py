"""Construction, normalization, and validation helpers for runtime Config."""

from __future__ import annotations

import os
import warnings
from copy import deepcopy
from dataclasses import fields
from typing import Any

from byte._config_sections import (
    _DEFAULT_GROUNDED_CONTEXT_CATEGORIES,
    _FIELD_TO_SECTION,
    _SECTION_ATTRS,
    _SECTION_ORDER,
    _SECTION_TYPES,
    BudgetConfig,
    CacheConfig,
    CompressionConfig,
    ContextCompilerConfig,
    IntegrationConfig,
    MemoryConfig,
    ObservabilityConfig,
    PromptDistillationConfig,
    QualityConfig,
    RoutingConfig,
    SecurityConfig,
    TrustConfig,
    _coerce_env_value,
    _normalize_string_list,
    _split_legacy_args,
)
from byte.utils.error import CacheError

_SECTION_DEFAULT_FACTORIES = {
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

_PUBLIC_CONFIG_DIR_ENTRIES = {
    "observability",
    "cache",
    "routing",
    "quality",
    "memory",
    "budget",
    "security",
    "compression",
    "trust",
    "integrations",
    "context_compiler_config",
    "prompt_distillation_config",
}


def prepare_config_values(args: tuple[Any, ...], overrides: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy positional args plus explicit overrides into flat field values."""
    values = _split_legacy_args(args)
    overlap = set(values).intersection(overrides)
    if overlap:
        repeated = ", ".join(sorted(overlap))
        raise TypeError(f"Config() got multiple values for argument(s): {repeated}")
    values.update(overrides)
    return values


def extract_section_objects(values: dict[str, Any]) -> dict[str, Any]:
    """Pop section objects from flat values and validate that remaining keys are known."""
    section_objects: dict[str, Any] = {}
    for section_name, section_type in _SECTION_TYPES.items():
        candidate = values.get(section_name)
        if isinstance(candidate, section_type):
            section_objects[section_name] = values.pop(section_name)

    unknown = [name for name in values if name not in _FIELD_TO_SECTION]
    if unknown:
        repeated = ", ".join(sorted(unknown))
        raise TypeError(f"Config() got unexpected keyword argument(s): {repeated}")
    return section_objects


def initialize_sections(
    config: Any,
    *,
    section_objects: dict[str, Any],
    explicit_sections: dict[str, Any | None],
) -> None:
    """Populate grouped config sections with deep-copied explicit or default values."""
    for section_name, section_attr in _SECTION_ATTRS.items():
        selected = section_objects.get(section_name, explicit_sections.get(section_name))
        factory = _SECTION_DEFAULT_FACTORIES[section_name]
        object.__setattr__(
            config,
            section_attr,
            deepcopy(selected) if selected is not None else factory(),
        )


def apply_env_overrides(config: Any, *, prefix: str) -> None:
    """Apply environment-based overrides to flat config fields."""
    env_prefix = f"{str(prefix or 'BYTE').upper()}_"
    for field_name in _FIELD_TO_SECTION:
        env_name = f"{env_prefix}{field_name.upper()}"
        if env_name not in os.environ:
            continue
        current = getattr(config, field_name)
        setattr(config, field_name, _coerce_env_value(field_name, os.environ[env_name], current))


def finalize_config(config: Any) -> None:
    """Normalize derived config state after section construction and overrides."""
    cfg: Any = config
    if cfg.skip_list is None:
        cfg.skip_list = ["system", "assistant"]
    cfg.retrieval_namespace_fields = _normalize_string_list(cfg.retrieval_namespace_fields)
    cfg.semantic_allowed_categories = _normalize_string_list(cfg.semantic_allowed_categories)
    cfg.cheap_consensus_models = _normalize_string_list(cfg.cheap_consensus_models)
    cfg.grounded_context_categories = _normalize_string_list(
        cfg.grounded_context_categories or _DEFAULT_GROUNDED_CONTEXT_CATEGORIES
    )
    cfg.security_allowed_egress_hosts = _normalize_string_list(
        cfg.security_allowed_egress_hosts, lower=True
    )
    cfg.kv_codec = str(cfg.kv_codec or "disabled").strip().lower()
    cfg.vector_codec = str(cfg.vector_codec or "disabled").strip().lower()
    cfg.compression_mode = str(cfg.compression_mode or "shadow").strip().lower()
    cfg.compression_backend_policy = str(cfg.compression_backend_policy or "auto").strip().lower()
    cfg.trust_mode = str(cfg.trust_mode or "guarded").strip().lower()
    cfg.query_risk_mode = str(cfg.query_risk_mode or "hybrid").strip().lower()
    cfg.confidence_mode = str(cfg.confidence_mode or "calibrated").strip().lower()
    cfg.confidence_backend = str(cfg.confidence_backend or "hybrid").strip().lower()
    cfg.reuse_band_policy = str(cfg.reuse_band_policy or "adaptive").strip().lower()
    cfg.conformal_mode = str(cfg.conformal_mode or "guarded").strip().lower()
    cfg.deterministic_contract_mode = str(cfg.deterministic_contract_mode or "enforced").strip().lower()
    cfg.semantic_cache_verifier_mode = str(cfg.semantic_cache_verifier_mode or "hybrid").strip().lower()
    cfg.semantic_cache_promotion_mode = str(cfg.semantic_cache_promotion_mode or "shadow").strip().lower()
    cfg.prompt_module_mode = str(cfg.prompt_module_mode or "enabled").strip().lower()
    cfg.prompt_distillation_mode = str(cfg.prompt_distillation_mode or "shadow").strip().lower()
    cfg.prompt_distillation_backend = str(
        cfg.prompt_distillation_backend or "hybrid_local"
    ).strip().lower()
    cfg.prompt_distillation_retrieval_mode = str(
        cfg.prompt_distillation_retrieval_mode or "hybrid"
    ).strip().lower()
    cfg.prompt_distillation_module_mode = str(
        cfg.prompt_distillation_module_mode or "enabled"
    ).strip().lower()
    cfg.prompt_distillation_artifact_version = str(
        cfg.prompt_distillation_artifact_version or "byte-prompt-distill-v1"
    ).strip()
    cfg.calibration_artifact_version = str(
        cfg.calibration_artifact_version or "byte-trust-v2"
    ).strip()
    cfg.benchmark_contract_version = str(
        cfg.benchmark_contract_version or "byte-benchmark-v2"
    ).strip()

    aliases: dict[str, list[str]] = {}
    for key, value in (cfg.routing_model_aliases or {}).items():
        normalized = [value] if isinstance(value, str) else [str(item) for item in (value or [])]
        aliases[str(key)] = [item for item in normalized if item]
    cfg.routing_model_aliases = aliases

    cfg.routing_fallbacks = {
        str(key): [str(item) for item in (value or []) if str(item).strip()]
        for key, value in (cfg.routing_fallbacks or {}).items()
    }
    cfg.routing_provider_keys = {
        str(key): value for key, value in (cfg.routing_provider_keys or {}).items()
    }
    cfg.task_policies = {
        str(key): dict(value or {})
        for key, value in (cfg.task_policies or {}).items()
        if isinstance(value, dict)
    }
    if isinstance(cfg.telemetry_attributes, dict):
        cfg.telemetry_attributes = {
            str(key): value for key, value in cfg.telemetry_attributes.items()
        }
    cfg.routing_verifier_enabled = bool(cfg.routing_verifier_enabled or cfg.routing_verifier_model)

    # Deprecation shim: map legacy TrustConfig fields to vCache equivalents (arXiv 2502.03771)
    _default_reuse_band = "adaptive"
    _default_novelty = 0.62
    if cfg.reuse_band_policy != _default_reuse_band:
        warnings.warn(
            "TrustConfig.reuse_band_policy is deprecated. Use CacheConfig.vcache_enabled instead "
            "(vCache, arXiv 2502.03771). The old field will be removed in a future release.",
            DeprecationWarning,
            stacklevel=4,
        )
        if cfg.reuse_band_policy == "fixed":
            cfg.vcache_enabled = True
    if cfg.novelty_threshold != _default_novelty:
        warnings.warn(
            "TrustConfig.novelty_threshold is deprecated. Use CacheConfig.vcache_delta instead "
            "(vCache, arXiv 2502.03771). The old field will be removed in a future release.",
            DeprecationWarning,
            stacklevel=4,
        )
        cfg.vcache_delta = max(0.0, min(1.0, 1.0 - cfg.novelty_threshold))

    compliance_profile = str(cfg.compliance_profile or "").strip().lower() or None
    cfg.compliance_profile = compliance_profile
    cfg.security_mode = bool(cfg.security_mode or compliance_profile is not None)
    if cfg.compliance_profile in {"hipaa", "soc2"}:
        cfg.security_mode = True
        cfg.security_redact_logs = True
        cfg.security_redact_reports = True
        cfg.security_redact_memory = True
        cfg.security_require_admin_auth = True
        cfg.security_disable_cache_file_endpoint = True
        cfg.security_encrypt_artifacts = True
        cfg.security_allow_provider_host_override = False


def validate_config(config: Any) -> None:
    """Validate grouped config fields after normalization."""
    _validate_cache_and_memory(config)
    _validate_routing(config)
    _validate_context_and_distillation(config)
    _validate_budget_security_and_telemetry(config)
    _validate_compression_and_trust(config)


def to_flat_dict(config: Any) -> dict[str, Any]:
    """Serialize grouped config state back into the flat compatibility payload."""
    payload: dict[str, Any] = {}
    for section_name in _SECTION_ORDER:
        section = object.__getattribute__(config, _SECTION_ATTRS[section_name])
        for section_field in fields(section):
            payload[section_field.name] = deepcopy(getattr(section, section_field.name))
    return payload


def public_dir_entries() -> set[str]:
    """Return extra attribute names exposed through Config.__dir__."""
    return set(_PUBLIC_CONFIG_DIR_ENTRIES)


def _validate_cache_and_memory(config: Any) -> None:
    if config.similarity_threshold < 0 or config.similarity_threshold > 1:
        raise CacheError("Invalid the similarity threshold param, reasonable range: 0-1")
    if config.semantic_min_token_overlap < 0 or config.semantic_min_token_overlap > 1:
        raise CacheError("Invalid semantic_min_token_overlap, reasonable range: 0-1")
    if config.semantic_max_length_ratio is not None and config.semantic_max_length_ratio < 1:
        raise CacheError("semantic_max_length_ratio must be >= 1 when provided")
    if config.tool_result_ttl is not None and config.tool_result_ttl < 0:
        raise CacheError("tool_result_ttl must be >= 0 when provided")
    if config.memory_max_entries < 1:
        raise CacheError("memory_max_entries must be >= 1")
    if config.memory_embedding_preview_dims < 0:
        raise CacheError("memory_embedding_preview_dims must be >= 0")
    if config.tier1_promotion_threshold < 1:
        raise CacheError("tier1_promotion_threshold must be >= 1")
    if config.tier1_promotion_window_s <= 0:
        raise CacheError("tier1_promotion_window_s must be > 0")
    if config.async_write_back_queue_size < 1:
        raise CacheError("async_write_back_queue_size must be >= 1")
    if config.cache_admission_min_score < 0 or config.cache_admission_min_score > 1:
        raise CacheError("cache_admission_min_score must be between 0 and 1")
    if config.cache_latency_min_samples < 1:
        raise CacheError("cache_latency_min_samples must be >= 1")
    if config.cache_latency_probe_samples < config.cache_latency_min_samples:
        raise CacheError("cache_latency_probe_samples must be >= cache_latency_min_samples")
    if config.cache_latency_p95_multiplier <= 0:
        raise CacheError("cache_latency_p95_multiplier must be > 0")
    if config.cache_latency_min_hits < 0:
        raise CacheError("cache_latency_min_hits must be >= 0")
    if config.cache_latency_force_miss_samples < config.cache_latency_probe_samples:
        raise CacheError(
            "cache_latency_force_miss_samples must be >= cache_latency_probe_samples"
        )
    if config.cache_latency_min_hit_rate < 0 or config.cache_latency_min_hit_rate > 1:
        raise CacheError("cache_latency_min_hit_rate must be between 0 and 1")
    if config.native_prompt_cache_min_chars < 0:
        raise CacheError("native_prompt_cache_min_chars must be >= 0")


def _validate_routing(config: Any) -> None:
    if config.routing_long_prompt_chars < 1:
        raise CacheError("routing_long_prompt_chars must be >= 1")
    if config.routing_multi_turn_threshold < 1:
        raise CacheError("routing_multi_turn_threshold must be >= 1")
    if config.routing_retry_attempts < 0:
        raise CacheError("routing_retry_attempts must be >= 0")
    if config.routing_retry_backoff_ms < 0:
        raise CacheError("routing_retry_backoff_ms must be >= 0")
    if config.routing_cooldown_seconds < 0:
        raise CacheError("routing_cooldown_seconds must be >= 0")
    if config.routing_max_cheap_labels < 1:
        raise CacheError("routing_max_cheap_labels must be >= 1")
    if config.routing_max_cheap_fields < 1:
        raise CacheError("routing_max_cheap_fields must be >= 1")
    if config.cheap_consensus_min_score < 0 or config.cheap_consensus_min_score > 1:
        raise CacheError("cheap_consensus_min_score must be between 0 and 1")
    if config.evidence_min_support < 0 or config.evidence_min_support > 1:
        raise CacheError("evidence_min_support must be between 0 and 1")
    if config.evidence_structured_min_support < 0 or config.evidence_structured_min_support > 1:
        raise CacheError("evidence_structured_min_support must be between 0 and 1")
    if config.evidence_summary_min_support < 0 or config.evidence_summary_min_support > 1:
        raise CacheError("evidence_summary_min_support must be between 0 and 1")
    if config.routing_strategy not in {
        "priority",
        "round_robin",
        "simple_shuffle",
        "latency",
        "cost",
        "health_weighted",
    }:
        raise CacheError(
            "routing_strategy must be one of priority, round_robin, simple_shuffle, latency, cost, health_weighted"
        )
    if config.routing_verify_min_score < 0 or config.routing_verify_min_score > 1:
        raise CacheError("routing_verify_min_score must be between 0 and 1")
    if config.routing_grey_zone_min_score < 0 or config.routing_grey_zone_min_score > 1:
        raise CacheError("routing_grey_zone_min_score must be between 0 and 1")
    if config.routing_grey_zone_min_score > config.routing_verify_min_score:
        raise CacheError("routing_grey_zone_min_score must be <= routing_verify_min_score")
    if config.routing_adaptive_min_samples < 1:
        raise CacheError("routing_adaptive_min_samples must be >= 1")
    if config.routing_adaptive_quality_floor < 0 or config.routing_adaptive_quality_floor > 1:
        raise CacheError("routing_adaptive_quality_floor must be between 0 and 1")
    if config.speculative_max_parallel < 1:
        raise CacheError("speculative_max_parallel must be >= 1")


def _validate_context_and_distillation(config: Any) -> None:
    if config.ambiguity_min_chars < 1:
        raise CacheError("ambiguity_min_chars must be >= 1")
    if config.context_compiler_keep_last_messages < 1:
        raise CacheError("context_compiler_keep_last_messages must be >= 1")
    if config.context_compiler_max_chars < 128:
        raise CacheError("context_compiler_max_chars must be >= 128")
    if config.context_compiler_relevance_top_k < 1:
        raise CacheError("context_compiler_relevance_top_k must be >= 1")
    if (
        config.context_compiler_related_min_score < 0
        or config.context_compiler_related_min_score > 1
    ):
        raise CacheError("context_compiler_related_min_score must be between 0 and 1")
    if (
        config.context_compiler_total_aux_budget_ratio <= 0
        or config.context_compiler_total_aux_budget_ratio > 1
    ):
        raise CacheError("context_compiler_total_aux_budget_ratio must be between 0 and 1")
    if config.context_budget_low_risk_chars < 128:
        raise CacheError("context_budget_low_risk_chars must be >= 128")
    if config.context_budget_medium_risk_chars < config.context_budget_low_risk_chars:
        raise CacheError(
            "context_budget_medium_risk_chars must be >= context_budget_low_risk_chars"
        )
    if config.context_budget_high_risk_chars < config.context_budget_medium_risk_chars:
        raise CacheError(
            "context_budget_high_risk_chars must be >= context_budget_medium_risk_chars"
        )
    if config.prompt_distillation_mode not in {"disabled", "shadow", "guarded", "enabled"}:
        raise CacheError(
            "prompt_distillation_mode must be one of disabled, shadow, guarded, enabled"
        )
    if config.prompt_distillation_backend not in {"hybrid_local", "heuristic", "local_model"}:
        raise CacheError(
            "prompt_distillation_backend must be one of hybrid_local, heuristic, local_model"
        )
    if config.prompt_distillation_budget_ratio <= 0 or config.prompt_distillation_budget_ratio >= 1:
        raise CacheError("prompt_distillation_budget_ratio must be between 0 and 1")
    if config.prompt_distillation_min_chars < 128:
        raise CacheError("prompt_distillation_min_chars must be >= 128")
    if config.prompt_distillation_retrieval_mode not in {"disabled", "extractive", "hybrid"}:
        raise CacheError(
            "prompt_distillation_retrieval_mode must be one of disabled, extractive, hybrid"
        )
    if config.prompt_distillation_module_mode not in {"disabled", "enabled"}:
        raise CacheError("prompt_distillation_module_mode must be one of disabled, enabled")
    if (
        config.prompt_distillation_verify_shadow_rate < 0
        or config.prompt_distillation_verify_shadow_rate > 1
    ):
        raise CacheError("prompt_distillation_verify_shadow_rate must be between 0 and 1")
    if not config.prompt_distillation_artifact_version:
        raise CacheError("prompt_distillation_artifact_version must be a non-empty string")
    if config.prompt_module_mode not in {"disabled", "enabled"}:
        raise CacheError("prompt_module_mode must be one of disabled, enabled")


def _validate_budget_security_and_telemetry(config: Any) -> None:
    if config.budget_quality_floor < 0 or config.budget_quality_floor > 1:
        raise CacheError("budget_quality_floor must be between 0 and 1")
    if config.budget_latency_target_ms <= 0:
        raise CacheError("budget_latency_target_ms must be > 0")
    if config.telemetry_enabled and not str(config.telemetry_tracer_name or "").strip():
        raise CacheError(
            "telemetry_tracer_name must be a non-empty string when telemetry is enabled"
        )
    if config.telemetry_enabled and not str(config.telemetry_meter_name or "").strip():
        raise CacheError(
            "telemetry_meter_name must be a non-empty string when telemetry is enabled"
        )
    if config.telemetry_attributes is not None and not isinstance(config.telemetry_attributes, dict):
        raise CacheError("telemetry_attributes must be a dictionary when provided")
    if config.budget_strategy not in {"balanced", "lowest_cost", "low_latency", "quality_first"}:
        raise CacheError(
            "budget_strategy must be one of balanced, lowest_cost, low_latency, quality_first"
        )
    if config.compliance_profile not in (None, "hipaa", "soc2"):
        raise CacheError("compliance_profile must be one of hipaa, soc2, or None")
    if config.security_max_request_bytes < 1024:
        raise CacheError("security_max_request_bytes must be >= 1024")
    if config.security_max_upload_bytes < config.security_max_request_bytes:
        raise CacheError("security_max_upload_bytes must be >= security_max_request_bytes")
    if config.security_max_archive_bytes < config.security_max_request_bytes:
        raise CacheError("security_max_archive_bytes must be >= security_max_request_bytes")
    if config.security_max_archive_members < 1:
        raise CacheError("security_max_archive_members must be >= 1")
    if config.mcp_timeout_s <= 0:
        raise CacheError("mcp_timeout_s must be > 0")


def _validate_compression_and_trust(config: Any) -> None:
    if config.h2o_heavy_ratio < 0 or config.h2o_heavy_ratio > 1:
        raise CacheError("h2o_heavy_ratio must be between 0 and 1")
    if config.h2o_recent_ratio < 0 or config.h2o_recent_ratio > 1:
        raise CacheError("h2o_recent_ratio must be between 0 and 1")
    if (config.h2o_heavy_ratio + config.h2o_recent_ratio) > 1:
        raise CacheError("h2o_heavy_ratio + h2o_recent_ratio must be <= 1")
    if config.kv_codec not in {"disabled", "h2o", "qjl", "polarquant", "turboquant", "hybrid"}:
        raise CacheError("kv_codec must be one of disabled, h2o, qjl, polarquant, turboquant, hybrid")
    if config.vector_codec not in {"disabled", "qjl", "turboquant"}:
        raise CacheError("vector_codec must be one of disabled, qjl, turboquant")
    if config.kv_bits < 1 or config.kv_bits > 16:
        raise CacheError("kv_bits must be between 1 and 16")
    if config.vector_bits < 1 or config.vector_bits > 16:
        raise CacheError("vector_bits must be between 1 and 16")
    if config.kv_hot_window_ratio < 0 or config.kv_hot_window_ratio > 1:
        raise CacheError("kv_hot_window_ratio must be between 0 and 1")
    if config.compression_mode not in {"shadow", "guarded", "enabled"}:
        raise CacheError("compression_mode must be one of shadow, guarded, enabled")
    if config.compression_backend_policy not in {"auto", "cuda", "triton", "torch"}:
        raise CacheError("compression_backend_policy must be one of auto, cuda, triton, torch")
    if config.compression_verify_shadow_rate < 0 or config.compression_verify_shadow_rate > 1:
        raise CacheError("compression_verify_shadow_rate must be between 0 and 1")
    if config.trust_mode not in {"disabled", "shadow", "guarded", "enforced"}:
        raise CacheError("trust_mode must be one of disabled, shadow, guarded, enforced")
    if config.query_risk_mode not in {"disabled", "heuristic", "hybrid"}:
        raise CacheError("query_risk_mode must be one of disabled, heuristic, hybrid")
    if config.confidence_mode not in {"disabled", "heuristic", "calibrated"}:
        raise CacheError("confidence_mode must be one of disabled, heuristic, calibrated")
    if config.confidence_backend not in {"local", "provider", "hybrid"}:
        raise CacheError("confidence_backend must be one of local, provider, hybrid")
    if config.reuse_band_policy not in {"fixed", "adaptive"}:
        raise CacheError("reuse_band_policy must be one of fixed, adaptive")
    if config.conformal_mode not in {"disabled", "shadow", "guarded", "enforced"}:
        raise CacheError("conformal_mode must be one of disabled, shadow, guarded, enforced")
    if config.conformal_target_coverage < 0 or config.conformal_target_coverage > 1:
        raise CacheError("conformal_target_coverage must be between 0 and 1")
    if config.deterministic_contract_mode not in {"disabled", "guarded", "enforced"}:
        raise CacheError(
            "deterministic_contract_mode must be one of disabled, guarded, enforced"
        )
    if config.semantic_cache_verifier_mode not in {"disabled", "local", "provider", "hybrid"}:
        raise CacheError(
            "semantic_cache_verifier_mode must be one of disabled, local, provider, hybrid"
        )
    if config.semantic_cache_promotion_mode not in {"disabled", "shadow", "verified"}:
        raise CacheError(
            "semantic_cache_promotion_mode must be one of disabled, shadow, verified"
        )
    if config.novelty_threshold < 0 or config.novelty_threshold > 1:
        raise CacheError("novelty_threshold must be between 0 and 1")
    if not config.calibration_artifact_version:
        raise CacheError("calibration_artifact_version must be a non-empty string")
    if not config.benchmark_contract_version:
        raise CacheError("benchmark_contract_version must be a non-empty string")
    if config.vcache_delta < 0 or config.vcache_delta > 1:
        raise CacheError("vcache_delta must be between 0 and 1")
    if config.vcache_min_observations < 1:
        raise CacheError("vcache_min_observations must be >= 1")
    if config.vcache_cold_fallback_threshold < 0 or config.vcache_cold_fallback_threshold > 1:
        raise CacheError("vcache_cold_fallback_threshold must be between 0 and 1")
    if config.llm_equivalence_ambiguity_band_low < 0 or config.llm_equivalence_ambiguity_band_low > 1:
        raise CacheError("llm_equivalence_ambiguity_band_low must be between 0 and 1")
    if config.llm_equivalence_ambiguity_band_high < 0 or config.llm_equivalence_ambiguity_band_high > 1:
        raise CacheError("llm_equivalence_ambiguity_band_high must be between 0 and 1")
    if config.llm_equivalence_ambiguity_band_low >= config.llm_equivalence_ambiguity_band_high:
        raise CacheError("llm_equivalence_ambiguity_band_low must be < llm_equivalence_ambiguity_band_high")
    if config.intent_context_budget_ratio <= 0 or config.intent_context_budget_ratio > 1:
        raise CacheError("intent_context_budget_ratio must be between 0 (exclusive) and 1")
