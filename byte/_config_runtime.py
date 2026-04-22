"""Grouped runtime configuration with flat backwards-compatible accessors."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from byte._config_runtime_support import (
    apply_env_overrides,
    extract_section_objects,
    finalize_config,
    initialize_sections,
    prepare_config_values,
    public_dir_entries,
    to_flat_dict,
    validate_config,
)
from byte._config_sections import (
    _FIELD_TO_SECTION,
    _SECTION_ATTRS,
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
)


class Config:
    """Grouped runtime configuration with flat backwards-compatible access."""

    def __init__(
        self,
        *args: Any,
        observability_config: ObservabilityConfig | None = None,
        cache_config: CacheConfig | None = None,
        routing_config: RoutingConfig | None = None,
        quality_config: QualityConfig | None = None,
        context_compiler_config: ContextCompilerConfig | None = None,
        prompt_distillation_config: PromptDistillationConfig | None = None,
        memory_config: MemoryConfig | None = None,
        budget_config: BudgetConfig | None = None,
        security_config: SecurityConfig | None = None,
        compression_config: CompressionConfig | None = None,
        trust_config: TrustConfig | None = None,
        integration_config: IntegrationConfig | None = None,
        load_env: bool = True,
        env_prefix: str = "BYTE",
        **overrides: Any,
    ) -> None:
        values = prepare_config_values(args, overrides)
        section_objects = extract_section_objects(values)
        initialize_sections(
            self,
            section_objects=section_objects,
            explicit_sections={
                "observability": observability_config,
                "cache": cache_config,
                "routing": routing_config,
                "quality": quality_config,
                "context_compiler": context_compiler_config,
                "prompt_distillation": prompt_distillation_config,
                "memory": memory_config,
                "budget": budget_config,
                "security": security_config,
                "compression": compression_config,
                "trust": trust_config,
                "integrations": integration_config,
            },
        )
        if load_env:
            apply_env_overrides(self, prefix=env_prefix)
        for field_name, value in values.items():
            setattr(self, field_name, value)
        finalize_config(self)
        validate_config(self)

    def _apply_env_overrides(self, *, prefix: str) -> None:
        apply_env_overrides(self, prefix=prefix)

    def _finalize(self) -> None:
        finalize_config(self)

    def _validate(self) -> None:
        validate_config(self)

    def to_flat_dict(self) -> dict[str, Any]:
        return to_flat_dict(self)

    def __getattr__(self, name: str) -> Any:
        section_name = _FIELD_TO_SECTION.get(name)
        if section_name is None:
            raise AttributeError(name)
        section = object.__getattribute__(self, _SECTION_ATTRS[section_name])
        return getattr(section, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _SECTION_TYPES and isinstance(value, _SECTION_TYPES[name]):
            object.__setattr__(self, _SECTION_ATTRS[name], value)
            return
        if name in _SECTION_ATTRS.values():
            object.__setattr__(self, name, value)
            return
        section_name = _FIELD_TO_SECTION.get(name)
        if section_name is None:
            object.__setattr__(self, name, value)
            return
        section = object.__getattribute__(self, _SECTION_ATTRS[section_name])
        setattr(section, name, value)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(_FIELD_TO_SECTION) | public_dir_entries())

    def __deepcopy__(self, memo: dict[int, Any]) -> Config:
        clone = type(self)(load_env=False, **deepcopy(self.to_flat_dict()))
        memo[id(self)] = clone
        return clone

    @classmethod
    def from_env(cls, *, prefix: str = "BYTE", **overrides: Any) -> Config:
        return cls(load_env=True, env_prefix=prefix, **overrides)

    @property
    def observability(self) -> ObservabilityConfig:
        return cast(ObservabilityConfig, object.__getattribute__(self, _SECTION_ATTRS["observability"]))

    @property
    def cache(self) -> CacheConfig:
        return cast(CacheConfig, object.__getattribute__(self, _SECTION_ATTRS["cache"]))

    @property
    def routing(self) -> RoutingConfig:
        return cast(RoutingConfig, object.__getattribute__(self, _SECTION_ATTRS["routing"]))

    @property
    def quality(self) -> QualityConfig:
        return cast(QualityConfig, object.__getattribute__(self, _SECTION_ATTRS["quality"]))

    @property
    def memory(self) -> MemoryConfig:
        return cast(MemoryConfig, object.__getattribute__(self, _SECTION_ATTRS["memory"]))

    @property
    def prompt_distillation_config(self) -> PromptDistillationConfig:
        return cast(
            PromptDistillationConfig,
            object.__getattribute__(self, _SECTION_ATTRS["prompt_distillation"]),
        )

    @property
    def budget(self) -> BudgetConfig:
        return cast(BudgetConfig, object.__getattribute__(self, _SECTION_ATTRS["budget"]))

    @property
    def security(self) -> SecurityConfig:
        return cast(SecurityConfig, object.__getattribute__(self, _SECTION_ATTRS["security"]))

    @property
    def compression(self) -> CompressionConfig:
        return cast(
            CompressionConfig, object.__getattribute__(self, _SECTION_ATTRS["compression"])
        )

    @property
    def trust(self) -> TrustConfig:
        return cast(TrustConfig, object.__getattribute__(self, _SECTION_ATTRS["trust"]))

    @property
    def integrations(self) -> IntegrationConfig:
        return cast(
            IntegrationConfig, object.__getattribute__(self, _SECTION_ATTRS["integrations"])
        )

    @property
    def context_compiler_config(self) -> ContextCompilerConfig:
        return cast(
            ContextCompilerConfig,
            object.__getattribute__(self, _SECTION_ATTRS["context_compiler"]),
        )
