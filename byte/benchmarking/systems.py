from __future__ import annotations

import copy
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from byte import Cache, Config
from byte.adapter.api import init_cache
from byte.adapter.prompt_cache_bridge import apply_native_prompt_cache
from byte.benchmarking._optional_runtime import load_chat_backend
from byte.benchmarking.contracts import BenchmarkItem
from byte.benchmarking.scoring import canonical_output, score_output, score_policy_adherence
from byte.processor.pre import last_content, normalized_last_content
from byte.processor.shared_memory import clear_shared_memory

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
    system: str
    baseline_type: str
    provider: str
    model: str
    label: str


def default_system_specs(providers: list[str]) -> list[SystemSpec]:
    specs: list[SystemSpec] = []
    for provider in providers:
        model = provider_default_model(provider)
        specs.extend(
            [
                SystemSpec(
                    system="direct",
                    baseline_type="direct",
                    provider=provider,
                    model=model,
                    label=f"{provider}_direct",
                ),
                SystemSpec(
                    system="native_cache",
                    baseline_type="provider_native_cache",
                    provider=provider,
                    model=model,
                    label=f"{provider}_native_cache",
                ),
                SystemSpec(
                    system="byte",
                    baseline_type="byte",
                    provider=provider,
                    model=model,
                    label=f"{provider}_byte",
                ),
            ]
        )
        if provider == "openai":
            specs.extend(
                [
                    SystemSpec(
                        system="langchain_redis",
                        baseline_type="langchain_redis",
                        provider=provider,
                        model=model,
                        label="openai_langchain_redis",
                    ),
                    SystemSpec(
                        system="embedding_similarity",
                        baseline_type="embedding_similarity",
                        provider=provider,
                        model=model,
                        label="openai_embedding_similarity",
                    ),
                ]
            )
    return specs


def provider_default_model(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "openai":
        return "gpt-4o-mini"
    if normalized == "anthropic":
        return "claude-3-5-sonnet-latest"
    if normalized == "deepseek":
        return "deepseek-chat"
    raise ValueError(f"Unsupported provider: {provider}")


def provider_key_env(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "openai":
        return "OPENAI_API_KEY"
    if normalized == "anthropic":
        return "ANTHROPIC_API_KEY"
    if normalized == "deepseek":
        return "DEEPSEEK_API_KEY"
    raise ValueError(f"Unsupported provider: {provider}")


class ExecutableSystem:
    def __init__(self, spec: SystemSpec) -> None:
        self.spec = spec
        self._cache_obj: Cache | None = None
        self._cache_dir = ""
        self._memory_scope = ""
        self._embedding_cache: list[dict[str, Any]] = []

    def is_available(self) -> tuple[bool, str]:
        env_key = provider_key_env(self.spec.provider)
        if not str(os.getenv(env_key, "")).strip():
            return False, f"Missing {env_key}."
        if self.spec.system == "langchain_redis" and not str(os.getenv("REDIS_URL", "")).strip():
            return False, "Missing REDIS_URL."
        return True, ""

    def begin_phase(self, warmup_items: list[BenchmarkItem]) -> None:
        if self.spec.system == "byte":
            self._cache_dir = tempfile.mkdtemp(prefix=f"benchmark-{self.spec.provider}-")
            self._memory_scope = (
                f"benchmark::{self.spec.provider}::{self.spec.system}::{int(time.time() * 1000)}"
            )
            self._cache_obj = Cache()
            config = _base_cache_config(self._memory_scope)
            init_cache(
                mode="hybrid",
                data_dir=self._cache_dir,
                cache_obj=self._cache_obj,
                pre_func=last_content,
                normalized_pre_func=normalized_last_content,
                config=config,
                exact_config=config,
                normalized_config=config,
                semantic_config=config,
            )
        self._embedding_cache = []
        for item in warmup_items:
            self.run_item(item, phase="warmup")

    def end_phase(self) -> None:
        if self._cache_obj is not None:
            current = self._cache_obj
            seen: set[int] = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                current.data_manager = None
                current = getattr(current, "next_cache", None)
        if self._cache_dir:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
        if self._memory_scope:
            clear_shared_memory(self._memory_scope)
        self._cache_obj = None
        self._cache_dir = ""
        self._memory_scope = ""
        self._embedding_cache = []

    def run_item(self, item: BenchmarkItem, *, phase: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        request_kwargs = _request_kwargs(
            item,
            provider=self.spec.provider,
            model=self._resolve_model(item),
            include_visible_context=self.spec.system != "byte",
        )
        response: dict[str, Any] | None = None
        error = ""
        status_code = 200
        try:
            if self.spec.system == "direct":
                response = _direct_request(self.spec.provider, request_kwargs)
            elif self.spec.system == "native_cache":
                response = _native_cache_request(self.spec.provider, request_kwargs)
            elif self.spec.system == "byte":
                response = _byte_request(
                    self.spec.provider,
                    request_kwargs,
                    cache_obj=self._cache_obj,
                )
            elif self.spec.system == "langchain_redis":
                response = _langchain_redis_request(self.spec.provider, request_kwargs)
            elif self.spec.system == "embedding_similarity":
                response = _embedding_similarity_request(
                    self.spec.provider,
                    request_kwargs,
                    cache_state=self._embedding_cache,
                )
            else:
                raise ValueError(f"Unsupported benchmark system: {self.spec.system}")
        except Exception as exc:
            response = {}
            error = str(exc)
            status_code = 599
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response = dict(response or {})
        response_text = _extract_text(response)
        served_via = _served_via(response)
        actual_reuse = served_via == "reuse"
        fallback_taken = served_via == "upstream"
        reuse_confidence = _reuse_confidence(response, served_via)
        trust_metadata = dict(response.get("byte_trust", {}) or {})
        upstream_calls = 1 if served_via == "upstream" else 0
        usage = _usage_fields(response.get("usage"))
        prompt_distillation = dict(response.get("byte_prompt_distillation", {}) or {})
        original_prompt_tokens = _prompt_distillation_tokens(
            prompt_distillation,
            key="original_prompt_tokens",
            fallback=usage["prompt_tokens"],
        )
        distilled_prompt_tokens = _prompt_distillation_tokens(
            prompt_distillation,
            key="distilled_prompt_tokens",
            fallback=usage["prompt_tokens"],
        )
        original_prompt_chars = int(
            prompt_distillation.get(
                "original_prompt_chars",
                max(len(_prompt_signature(request_kwargs)), 0),
            )
            or max(len(_prompt_signature(request_kwargs)), 0)
        )
        distilled_prompt_chars = int(
            prompt_distillation.get("distilled_prompt_chars", original_prompt_chars)
            or original_prompt_chars
        )
        output_correct = score_output(item, response_text, fallback_taken=fallback_taken)
        policy_adherent = score_policy_adherence(item, response_text)
        workflow_steps_skipped = (
            item.workflow_total_steps if served_via in {"reuse", "local_compute"} else 0
        )
        return {
            "system": self.spec.system,
            "provider": self.spec.provider,
            "baseline_type": self.spec.baseline_type,
            "phase": phase,
            "family": item.family,
            "scenario": item.scenario,
            "seed_id": item.seed_id,
            "variant_id": item.variant_id,
            "reuse_safe": item.reuse_safe,
            "must_fallback": item.must_fallback,
            "actual_reuse": actual_reuse,
            "fallback_taken": fallback_taken,
            "reuse_confidence": reuse_confidence,
            "reuse_decision_correct": _reuse_decision_correct(
                actual_reuse,
                fallback_taken,
                item.reuse_safe,
                item.must_fallback,
            ),
            "output_correct": output_correct,
            "policy_adherent": policy_adherent,
            "deterministic_output": output_correct if item.deterministic_expected else True,
            "deterministic_expected": item.deterministic_expected,
            "workflow_steps_skipped": workflow_steps_skipped,
            "workflow_total_steps": item.workflow_total_steps,
            "upstream_calls": upstream_calls,
            "tokens": usage,
            "original_prompt_tokens": original_prompt_tokens,
            "distilled_prompt_tokens": distilled_prompt_tokens,
            "original_prompt_chars": original_prompt_chars,
            "distilled_prompt_chars": distilled_prompt_chars,
            "prompt_token_reduction_ratio": _ratio_delta(
                original_prompt_tokens,
                distilled_prompt_tokens,
            ),
            "compression_ratio": _ratio_delta(
                original_prompt_chars,
                distilled_prompt_chars,
            ),
            "faithfulness_pass": str(prompt_distillation.get("verifier_result", "") or "")
            == "pass",
            "faithfulness_score": float(prompt_distillation.get("faithfulness_score", 1.0) or 1.0),
            "entity_preservation_rate": float(
                prompt_distillation.get("entity_preservation_rate", 1.0) or 1.0
            ),
            "schema_preservation_rate": float(
                prompt_distillation.get("schema_preservation_rate", 1.0) or 1.0
            ),
            "module_hits": int(prompt_distillation.get("module_hits", 0) or 0),
            "distillation_fallback": bool(
                str(prompt_distillation.get("fallback_reason", "") or "")
            ),
            "prompt_distillation_applied": bool(prompt_distillation.get("applied", False)),
            "cost_usd": _estimate_cost(self.spec.provider, self._resolve_model(item), usage),
            "status_code": status_code,
            "latency_ms": latency_ms,
            "trace_ref": f"{self.spec.provider}/{self.spec.system}/{phase}/{item.item_id}",
            "served_via": served_via,
            "byte_reason": str(response.get("byte_reason", "") or ""),
            "trust_abstained": bool(trust_metadata.get("abstained", False)),
            "trust_fallback_reason": str(trust_metadata.get("fallback_reason", "") or ""),
            "byte_prompt_distillation": prompt_distillation,
            "response_text": response_text,
            "canonical_output": canonical_output(item, response_text),
            "tags": list(item.tags),
            "item_id": item.item_id,
            "provider_model": str(response.get("model", "") or self._resolve_model(item)),
            "configured_model": self._resolve_model(item),
            "error": error,
        }

    def _resolve_model(self, item: BenchmarkItem) -> str:
        return str(item.model_hint or self.spec.model or provider_default_model(self.spec.provider))


def build_systems(providers: list[str], systems: list[str] | None = None) -> list[ExecutableSystem]:
    requested = set(systems or [])
    executables = []
    for spec in default_system_specs(providers):
        if requested and spec.system not in requested:
            continue
        executables.append(ExecutableSystem(spec))
    return executables


def _direct_request(provider: str, request_kwargs: dict[str, Any]) -> dict[str, Any]:
    backend = _chat_backend(provider)
    return dict(backend._llm_handler(**request_kwargs) or {})


def _native_cache_request(provider: str, request_kwargs: dict[str, Any]) -> dict[str, Any]:
    cfg = Config(native_prompt_caching=True, native_prompt_cache_min_chars=0)
    cached_kwargs = apply_native_prompt_cache(provider, request_kwargs, cfg)
    backend = _chat_backend(provider)
    response = dict(backend._llm_handler(**cached_kwargs) or {})
    response.setdefault("byte_native_cache_mode", _native_cache_mode(provider))
    return response


def _byte_request(provider: str, request_kwargs: dict[str, Any], *, cache_obj: Cache | None) -> dict[str, Any]:
    backend = _chat_backend(provider)
    return dict(backend.create(cache_obj=cache_obj, **request_kwargs) or {})


def _langchain_redis_request(provider: str, request_kwargs: dict[str, Any]) -> dict[str, Any]:
    if provider != "openai":
        raise ValueError("LangChain + Redis baseline is only available on the OpenAI track.")
    try:
        from langchain_openai import ChatOpenAI
        from redis import Redis
    except ImportError as exc:
        raise ValueError("Install the benchmark extra to run LangChain + Redis baselines.") from exc
    redis_client = Redis.from_url(os.getenv("REDIS_URL", ""))
    prompt_text = _prompt_signature(request_kwargs)
    cache_key = f"byte-benchmark:langchain:{request_kwargs.get('model', '')}:{prompt_text}"
    cached = redis_client.get(cache_key)
    if cached is not None:
        return {
            "byte": True,
            "byte_reason": "langchain_redis_reuse",
            "choices": [{"message": {"role": "assistant", "content": cached.decode('utf-8')}}],
            "model": request_kwargs.get("model", ""),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    model = ChatOpenAI(
        model=str(request_kwargs.get("model", "") or "gpt-4o-mini"),
        temperature=float(request_kwargs.get("temperature", 0.0) or 0.0),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    content = model.invoke(request_kwargs["messages"]).content
    text = content if isinstance(content, str) else str(content)
    redis_client.set(cache_key, text)
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "model": request_kwargs.get("model", ""),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _embedding_similarity_request(
    provider: str,
    request_kwargs: dict[str, Any],
    *,
    cache_state: list[dict[str, Any]],
) -> dict[str, Any]:
    if provider != "openai":
        raise ValueError("Embedding-similarity baseline is only available on the OpenAI track.")
    prompt_text = _prompt_signature(request_kwargs)
    for entry in cache_state:
        if entry["signature"] == prompt_text:
            return {
                "byte": True,
                "byte_reason": "embedding_similarity_reuse",
                "choices": [{"message": {"role": "assistant", "content": entry["content"]}}],
                "model": request_kwargs.get("model", ""),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
    response = _direct_request(provider, request_kwargs)
    cache_state.append({"signature": prompt_text, "content": _extract_text(response)})
    return response


def _chat_backend(provider: str) -> Any:
    return load_chat_backend(provider)


def _request_kwargs(
    item: BenchmarkItem,
    *,
    provider: str,
    model: str,
    include_visible_context: bool,
) -> dict[str, Any]:
    payload = copy.deepcopy(item.input_payload)
    payload["model"] = model
    payload["temperature"] = float(payload.get("temperature", 0.0) or 0.0)
    payload["max_tokens"] = int(payload.get("max_tokens", 32) or 32)
    messages = list(payload.get("messages") or [])
    context_payload = dict(payload.pop("context_payload", {}) or {})
    if context_payload:
        payload["messages"] = (
            _visible_messages(messages, context_payload) if include_visible_context else messages
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


def _visible_messages(messages: list[dict[str, Any]], context_payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not messages:
        messages = [{"role": "user", "content": ""}]
    visible = copy.deepcopy(messages)
    rendered = _render_context_payload(context_payload)
    if rendered:
        last = dict(visible[-1])
        last["content"] = f"{str(last.get('content', '')).strip()}\n\nContext:\n{rendered}".strip()
        visible[-1] = last
    return visible


def _render_context_payload(context_payload: dict[str, Any]) -> str:
    pieces = []
    for key, value in context_payload.items():
        if value in (None, "", [], {}):
            continue
        pieces.append(f"{key}: {value}")
    return "\n".join(pieces)


def _prompt_signature(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    if messages:
        return "\n".join(str(message.get("content", "") or "") for message in messages)
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    return ""


def _extract_text(response: dict[str, Any]) -> str:
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


def _usage_fields(usage: Any) -> dict[str, int]:
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


def _estimate_cost(provider: str, model: str, usage: dict[str, int]) -> float:
    pricing = _pricing_entry(provider, model)
    if not pricing:
        return 0.0
    return round(
        (usage["uncached_prompt_tokens"] / 1_000_000) * float(pricing["input"])
        + (usage["cached_prompt_tokens"] / 1_000_000) * float(pricing.get("cached_input", pricing["input"]))
        + (usage["completion_tokens"] / 1_000_000) * float(pricing["output"]),
        8,
    )


def _pricing_entry(provider: str, model: str) -> dict[str, float] | None:
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


def _served_via(response: dict[str, Any]) -> str:
    if not bool(response.get("byte", False)):
        return "upstream"
    reason = str(response.get("byte_reason", "") or "")
    if reason in _REUSE_ONLY_REASONS:
        return "reuse"
    if reason in _SHORTCUT_REASONS:
        return "local_compute"
    return "reuse"


def _reuse_confidence(response: dict[str, Any], served_via: str) -> float:
    trust = dict(response.get("byte_trust", {}) or {})
    if "calibrated_confidence" in trust:
        return round(float(trust.get("calibrated_confidence", 0.0) or 0.0), 4)
    reasoning = dict(response.get("byte_reasoning", {}) or {})
    if "confidence" in reasoning:
        return round(float(reasoning.get("confidence", 0.0) or 0.0), 4)
    if served_via == "reuse":
        return 0.93
    if served_via == "local_compute":
        return 0.20
    return 0.05


def _native_cache_mode(provider: str) -> str:
    if str(provider or "").strip().lower() == "deepseek":
        return "provider_automatic"
    return "request_hint"


def _prompt_distillation_tokens(payload: dict[str, Any], *, key: str, fallback: int) -> int:
    value = payload.get(key)
    if value in (None, ""):
        return int(fallback or 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return int(fallback or 0)


def _ratio_delta(baseline: int, observed: int) -> float:
    if int(baseline or 0) <= 0:
        return 0.0
    return round((float(baseline) - float(observed)) / float(baseline), 4)


def _reuse_decision_correct(
    actual_reuse: bool,
    fallback_taken: bool,
    reuse_safe: bool,
    must_fallback: bool,
) -> bool:
    if actual_reuse:
        return reuse_safe
    if fallback_taken:
        return must_fallback or not reuse_safe
    return True


def _base_cache_config(scope: str) -> Config:
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
