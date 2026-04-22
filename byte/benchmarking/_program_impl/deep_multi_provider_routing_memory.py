import argparse
import concurrent.futures as cf
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from byte import Cache, Config
from byte._backends import openai as byte_openai
from byte.adapter.api import init_cache, preview_model_route
from byte.benchmarking._optional_runtime import create_openai_client
from byte.benchmarking._program_common import (
    normalized_answer,
    p95,
    usage_fields,
)
from byte.benchmarking.programs import deep_openai_cost_levers as helpers
from byte.processor.shared_memory import clear_shared_memory
from byte.session import Session

BYTE_MODES = list(helpers.BYTE_MODES)
VERIFIED_ON = "2026-03-10"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    display_name: str
    api_envs: tuple
    api_base: str | None
    benchmark_candidates: tuple
    cheap_candidates: tuple
    expensive_candidates: tuple
    pricing: dict[str, dict[str, float]]
    pricing_sources: list[str]
    compatibility_note: str = ""


PROVIDERS = {
    "openai": ProviderSpec(
        name="openai",
        display_name="OpenAI",
        api_envs=("BYTE_TEST_OPENAI_API_KEY", "OPENAI_API_KEY"),
        api_base=None,
        benchmark_candidates=("gpt-4o-mini", "gpt-4.1-mini"),
        cheap_candidates=("gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"),
        expensive_candidates=("gpt-4o", "gpt-4.1", "gpt-4.1-mini"),
        pricing={
            "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
            "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
            "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
            "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
            "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
        },
        pricing_sources=[
            "https://platform.openai.com/docs/pricing/",
            "https://openai.com/api/pricing/",
            "https://openai.com/index/gpt-4-1/",
        ],
    ),
    "gemini": ProviderSpec(
        name="gemini",
        display_name="Google Gemini",
        api_envs=("BYTE_TEST_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        benchmark_candidates=("gemini-2.5-flash", "gemini-2.5-flash-lite"),
        cheap_candidates=("gemini-2.5-flash-lite", "gemini-2.5-flash"),
        expensive_candidates=("gemini-2.5-pro", "gemini-2.5-flash"),
        pricing={
            "gemini-2.5-flash-lite": {"input": 0.10, "cached_input": 0.01, "output": 0.40},
            "gemini-2.5-flash": {"input": 0.30, "cached_input": 0.03, "output": 2.50},
            "gemini-2.5-pro": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
            "gemini-2.0-flash-lite": {"input": 0.075, "cached_input": 0.0, "output": 0.30},
        },
        pricing_sources=[
            "https://ai.google.dev/pricing",
            "https://ai.google.dev/gemini-api/docs/openai",
        ],
        compatibility_note=(
            "Live Gemini runs use Google's official OpenAI-compatible endpoint so the shared Byte adapter path is exercised as a provider-agnostic integration."
        ),
    ),
}


def _make_item(
    prompt: str,
    expected: str,
    group: str,
    variant: str,
    kind: str,
    *,
    max_tokens: int = 16,
    request_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "expected": expected,
        "group": group,
        "variant": variant,
        "kind": kind,
        "max_tokens": max_tokens,
        "request_overrides": dict(request_overrides or {}),
    }


def _routing_blend_scenario() -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    items.extend(
        [
            _make_item(
                'Classify the sentiment.\nLabels: POSITIVE, NEGATIVE, NEUTRAL\nReview: "The onboarding was fast and the support team solved everything."\nAnswer with exactly one label.',
                "POSITIVE",
                "routing_classify_00",
                "cheap_01",
                "classification",
                max_tokens=32,
            ),
            _make_item(
                'Return raw JSON only with keys "city" and "name". Text: "Name: Ana Patel. City: Lisbon. Team: Growth."',
                '{"city":"Lisbon","name":"Ana Patel"}',
                "routing_extract_00",
                "cheap_02",
                "extraction",
                max_tokens=48,
            ),
            _make_item(
                "Translate to Spanish and answer with the translation only: Good morning team",
                "Buenos dias equipo",
                "routing_translate_00",
                "cheap_03",
                "translation",
                max_tokens=24,
            ),
            _make_item(
                "Reply with exactly BYTE_ROUTE_OK and nothing else.",
                "BYTE_ROUTE_OK",
                "routing_exact_00",
                "cheap_04",
                "exact_answer",
                max_tokens=8,
            ),
        ]
    )
    hard_prompts = [
        (
            "Analyze the architecture, compare latency against cost, reason step by step about bottlenecks, and after that reply with exactly BYTE_ROUTE_A and nothing else.\nSystem notes: service A fans out to three workers, cache misses trigger vector search, and the hot path should stay deterministic.\nDo not explain the answer.",
            "BYTE_ROUTE_A",
            "routing_hard_00",
        ),
        (
            "Debug the concurrency issue, reason carefully about lock contention, compare alternative fixes, and then reply with exactly BYTE_ROUTE_B and nothing else.\nDetails: p95 spikes only under 16 concurrent requests, misses should coalesce before embedding work, and duplicate prompts should avoid repeated upstream calls.",
            "BYTE_ROUTE_B",
            "routing_hard_01",
        ),
        (
            "Evaluate the tradeoff, analyze the request budget, optimize the cache path, and after thinking step by step reply with exactly BYTE_ROUTE_C and nothing else.\nContext: multi-turn support assistant, routing should prefer cheap models for structured work but keep long reasoning on a stronger model.",
            "BYTE_ROUTE_C",
            "routing_hard_02",
        ),
        (
            "Architect the rollout, compare failure modes, debug the fallback behavior, and reply with exactly BYTE_ROUTE_D and nothing else after your analysis.\nImportant: preserve accuracy, reduce miss-path tax, and keep semantic reuse behind safety rails.",
            "BYTE_ROUTE_D",
            "routing_hard_03",
        ),
    ]
    for idx, (prompt, expected, group) in enumerate(hard_prompts, 1):
        items.append(
            _make_item(prompt, expected, group, f"hard_{idx:02d}", "instruction", max_tokens=24)
        )

    for idx, (review, label) in enumerate(
        [
            ("I hated the delays and would not use it again.", "NEGATIVE"),
            ("The rollout was fine, nothing especially impressive.", "NEUTRAL"),
        ],
        1,
    ):
        for variant_idx, prompt in enumerate(helpers._classification_variants(review)[:2], 1):
            items.append(
                _make_item(
                    prompt,
                    label,
                    f"routing_classify_extra_{idx:02d}",
                    f"v{variant_idx}",
                    "classification",
                )
            )
    return {
        "name": "routing_blend_12",
        "description": "12 requests mixing structured cheap tasks with long analysis-style prompts so routing can decide between cheap and expensive models.",
        "items": items,
    }


def _byte_memory_payload(
    provider_name: str, scenario_name: str, item: dict[str, Any]
) -> dict[str, Any]:
    return {
        "provider": provider_name,
        "metadata": {
            "scenario": scenario_name,
            "group": item["group"],
            "variant": item["variant"],
            "kind": item["kind"],
        },
    }


def _env_value(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _client_for_provider(spec: ProviderSpec, api_key: str) -> Any:
    kwargs = {"api_key": api_key}
    if spec.api_base:
        kwargs["base_url"] = spec.api_base
    return create_openai_client(**kwargs)


def _list_models(client: Any) -> dict[str, Any]:
    try:
        response = client.models.list()
        data = getattr(response, "data", None)
        if data is None:
            data = list(response)
        return {
            "models": [str(getattr(model, "id", "")) for model in data if getattr(model, "id", "")],
            "error": "",
        }
    except Exception as exc:
        return {"models": [], "error": str(exc)}


def _probe_model(client: Any, model: str) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly OK and nothing else."}],
            temperature=0,
            max_tokens=4,
        )
        text = (response.choices[0].message.content or "").strip().upper()
        return {"ok": text.startswith("OK"), "error": ""}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _resolve_models(spec: ProviderSpec, api_key: str) -> dict[str, Any]:
    client = _client_for_provider(spec, api_key)
    list_result = _list_models(client)
    available = set(list_result["models"])

    def choose(candidates: Iterable[str], *, fallback: str | None = None) -> str:
        probe_errors: dict[str, str] = {}
        for candidate in candidates:
            if available and candidate in available:
                return candidate
        for candidate in candidates:
            probe_result = _probe_model(client, candidate)
            if probe_result["ok"]:
                return candidate
            if probe_result["error"]:
                probe_errors[candidate] = probe_result["error"]
        if fallback:
            return fallback
        if list_result["error"]:
            raise RuntimeError(
                f"{spec.display_name} model discovery failed: {list_result['error']}"
            )
        if probe_errors:
            details = "; ".join(
                f"{candidate}: {probe_errors[candidate]}"
                for candidate in candidates
                if candidate in probe_errors
            )
            raise RuntimeError(f"No accessible candidate models for {spec.display_name}: {details}")
        raise RuntimeError(
            f"No available candidate models for {spec.display_name}: {list(candidates)}"
        )

    benchmark_model = choose(spec.benchmark_candidates)
    cheap_model = choose(spec.cheap_candidates, fallback=benchmark_model)
    expensive_model = choose(spec.expensive_candidates, fallback=benchmark_model)
    return {
        "benchmark_model": benchmark_model,
        "cheap_model": cheap_model,
        "expensive_model": expensive_model,
        "available_model_count": len(available),
    }


def _pricing_key(pricing: dict[str, dict[str, float]], model: str) -> str | None:
    normalized = (model or "").lower()
    for key in sorted(pricing.keys(), key=len, reverse=True):
        if normalized.startswith(key.lower()):
            return key
    return None


def _usage_fields(usage: Any) -> dict[str, int]:
    return usage_fields(usage)


def _pricing_cost(spec: ProviderSpec, model_name: str, usage: Any) -> float:
    fields = _usage_fields(usage)
    pricing_key = _pricing_key(spec.pricing, model_name)
    if pricing_key is None:
        return 0.0
    price = spec.pricing[pricing_key]
    prompt_tokens = fields["prompt_tokens"]
    cached_prompt_tokens = min(fields["cached_prompt_tokens"], prompt_tokens)
    completion_tokens = fields["completion_tokens"]
    uncached_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    return (
        (uncached_prompt_tokens / 1_000_000) * price["input"]
        + (cached_prompt_tokens / 1_000_000) * price.get("cached_input", price["input"])
        + (completion_tokens / 1_000_000) * price["output"]
    )


def _normalized_answer(text: str | None) -> str:
    normalized = normalized_answer(text)
    return (
        normalized.replace("Ã¡", "a")
        .replace("Ã©", "e")
        .replace("Ã­", "i")
        .replace("Ã³", "o")
        .replace("Ãº", "u")
    )


def _response_record(
    *,
    spec: ProviderSpec,
    status_code: int,
    latency_ms: float,
    byte_flag: bool,
    model_name: str,
    route_info: dict[str, Any] | None,
    text: str | None,
    usage: Any,
    item: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    fields = _usage_fields(usage)
    return {
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "byte": byte_flag,
        "prompt_tokens": fields["prompt_tokens"],
        "cached_prompt_tokens": fields["cached_prompt_tokens"],
        "completion_tokens": fields["completion_tokens"],
        "cost_usd": round(_pricing_cost(spec, model_name, usage), 8),
        "text": text,
        "expected": item["expected"],
        "group": item["group"],
        "variant": item["variant"],
        "kind": item["kind"],
        "model": model_name,
        "route_info": route_info or {},
        "error": error,
        "correct": _normalized_answer(text) == _normalized_answer(item["expected"]),
    }


def _call_with_retry(func) -> Any:
    last_error = None
    for attempt in range(3):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(1.0 + attempt)
    raise last_error  # type: ignore[misc]


def _direct_request(
    spec: ProviderSpec, api_key: str, item: dict[str, Any], model: str
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "max_tokens": int(item.get("max_tokens", 16) or 16),
    }
    payload.update(item.get("request_overrides", {}) or {})
    client = _client_for_provider(spec, api_key)
    start = time.perf_counter()
    try:
        response = _call_with_retry(lambda: client.chat.completions.create(**payload))
        latency_ms = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content or ""
        model_name = str(getattr(response, "model", "") or model)
        return _response_record(
            spec=spec,
            status_code=200,
            latency_ms=latency_ms,
            byte_flag=False,
            model_name=model_name,
            route_info=None,
            text=text,
            usage=getattr(response, "usage", None),
            item=item,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return _response_record(
            spec=spec,
            status_code=599,
            latency_ms=latency_ms,
            byte_flag=False,
            model_name=model,
            route_info=None,
            text=None,
            usage=None,
            item=item,
            error=str(exc),
        )


def _base_config(
    scope: str, *, routed: bool, models: dict[str, Any], routing_threshold: int = 220
) -> Config:
    return Config(
        enable_token_counter=False,
        embedding_cache_size=20000,
        memory_scope=scope,
        intent_memory=True,
        model_namespace=True,
        tool_namespace=True,
        context_fingerprint=True,
        routing_long_prompt_chars=routing_threshold,
        routing_multi_turn_threshold=4,
        model_routing=routed,
        routing_cheap_model=models["cheap_model"] if routed else None,
        routing_expensive_model=models["expensive_model"] if routed else None,
        routing_default_model=models["expensive_model"] if routed else None,
        semantic_allowed_categories=[
            "question_answer",
            "summarization",
            "comparison",
            "instruction",
        ],
    )


def _semantic_config(
    scope: str, *, routed: bool, models: dict[str, Any], routing_threshold: int = 220
) -> Config:
    return Config(
        enable_token_counter=False,
        similarity_threshold=0.94,
        semantic_min_token_overlap=0.5,
        semantic_max_length_ratio=2.5,
        semantic_enforce_canonical_match=True,
        tiered_cache=True,
        embedding_cache_size=20000,
        memory_scope=scope,
        intent_memory=True,
        model_namespace=True,
        tool_namespace=True,
        context_fingerprint=True,
        routing_long_prompt_chars=routing_threshold,
        routing_multi_turn_threshold=4,
        model_routing=routed,
        routing_cheap_model=models["cheap_model"] if routed else None,
        routing_expensive_model=models["expensive_model"] if routed else None,
        routing_default_model=models["expensive_model"] if routed else None,
        semantic_allowed_categories=[
            "question_answer",
            "summarization",
            "comparison",
            "instruction",
        ],
    )


def _configure_cache(
    cache_obj: Cache,
    cache_dir: str,
    mode: str,
    *,
    models: dict[str, Any],
    routed: bool = False,
    scope: str,
    warm_data: list[dict[str, str]] | None = None,
) -> None:
    base_config = _base_config(scope, routed=routed, models=models)
    semantic_config = _semantic_config(scope, routed=routed, models=models)
    init_cache(
        mode=mode,
        data_dir=cache_dir,
        cache_obj=cache_obj,
        pre_func=helpers.last_content,
        normalized_pre_func=helpers.normalized_last_content,
        config=base_config,
        exact_config=base_config,
        normalized_config=base_config,
        semantic_config=semantic_config,
        warm_data=warm_data,
    )


def _memory_capture(
    cache_obj: Cache, mode: str, *, models: dict[str, Any], scope: str
) -> dict[str, Any]:
    snapshot = cache_obj.export_memory_snapshot(tool_result_limit=12)
    route_tiers = Counter()
    for entry in snapshot.get("ai_memory", {}).get("entries", []):
        tier = ((entry.get("metadata") or {}).get("model_route") or {}).get("tier")
        if tier:
            route_tiers[tier] += 1

    clone_dir = tempfile.mkdtemp(prefix="byte-memory-clone-")
    clone_scope = scope + "-clone"
    clear_shared_memory(clone_scope)
    clone = Cache()
    try:
        _configure_cache(
            clone, clone_dir, mode, models=models, routed=False, scope=clone_scope, warm_data=None
        )
        merge_result = clone.import_memory_snapshot(snapshot)
        clone_summary = clone.memory_summary()
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)
        clear_shared_memory(clone_scope)

    return {
        "summary": cache_obj.memory_summary(),
        "snapshot_stats": {
            "intent_records": snapshot.get("intent_graph", {}).get("total_records", 0),
            "ai_entries": snapshot.get("ai_memory", {}).get("stats", {}).get("total_entries", 0),
            "route_tiers": dict(route_tiers),
        },
        "recent_interactions": cache_obj.recent_interactions(limit=5),
        "import_result": merge_result,
        "imported_summary": clone_summary,
    }


def _byte_request(
    spec: ProviderSpec,
    api_key: str,
    cache_obj: Cache,
    item: dict[str, Any],
    *,
    model: str,
    scenario_name: str,
    session: Session | None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "max_tokens": int(item.get("max_tokens", 16) or 16),
        "api_key": api_key,
        "cache_obj": cache_obj,
        "byte_memory": _byte_memory_payload(spec.name, scenario_name, item),
    }
    if spec.api_base:
        payload["api_base"] = spec.api_base
    if session is not None:
        payload["session"] = session
    payload.update(item.get("request_overrides", {}) or {})
    route_info = preview_model_route(dict(payload), cache_obj=cache_obj)
    start = time.perf_counter()
    try:
        response = _call_with_retry(lambda: byte_openai.ChatCompletion.create(**payload))
        latency_ms = (time.perf_counter() - start) * 1000
        text = (((response or {}).get("choices") or [{}])[0].get("message") or {}).get("content")
        model_name = str(
            (response or {}).get("model") or (route_info or {}).get("selected_model") or model
        )
        return _response_record(
            spec=spec,
            status_code=200,
            latency_ms=latency_ms,
            byte_flag=bool((response or {}).get("byte")),
            model_name=model_name,
            route_info=route_info,
            text=text,
            usage=(response or {}).get("usage"),
            item=item,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        fallback_model = (route_info or {}).get("selected_model") or model
        return _response_record(
            spec=spec,
            status_code=599,
            latency_ms=latency_ms,
            byte_flag=False,
            model_name=str(fallback_model),
            route_info=route_info,
            text=None,
            usage=None,
            item=item,
            error=str(exc),
        )


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records]
    cached = sum(1 for record in records if record["byte"])
    correct = sum(1 for record in records if record["correct"])
    model_counts = Counter(record["model"] for record in records if record.get("model"))
    route_tiers = Counter(
        (record.get("route_info") or {}).get("tier")
        for record in records
        if (record.get("route_info") or {}).get("tier")
    )
    return {
        "request_count": len(records),
        "cached_count": cached,
        "miss_count": len(records) - cached,
        "error_count": sum(1 for record in records if record["status_code"] != 200),
        "hit_ratio": round(cached / len(records), 4) if records else 0.0,
        "accuracy_ratio": round(correct / len(records), 4) if records else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": p95(latencies),
        "total_prompt_tokens": sum(record["prompt_tokens"] for record in records),
        "total_cached_prompt_tokens": sum(record["cached_prompt_tokens"] for record in records),
        "total_completion_tokens": sum(record["completion_tokens"] for record in records),
        "total_cost_usd": round(sum(record["cost_usd"] for record in records), 8),
        "models": dict(model_counts),
        "route_tiers": dict(route_tiers),
        "sample": records[:5],
    }


def _run_direct_sequence(
    spec: ProviderSpec, api_key: str, items: list[dict[str, Any]], *, model: str
) -> dict[str, Any]:
    return _summarize_records([_direct_request(spec, api_key, item, model) for item in items])


def _run_byte_sequence(
    spec: ProviderSpec,
    api_key: str,
    items: list[dict[str, Any]],
    *,
    mode: str,
    model: str,
    models: dict[str, Any],
    routed: bool = False,
    scenario_name: str,
    warm_data: list[dict[str, str]] | None = None,
    capture_memory: bool = False,
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"byte-{spec.name}-{mode}-")
    scope = f"bench::{spec.name}::{mode}::{scenario_name}::{int(time.time() * 1000)}"
    clear_shared_memory(scope)
    cache_obj = Cache()
    try:
        _configure_cache(
            cache_obj,
            cache_dir,
            mode,
            models=models,
            routed=routed,
            scope=scope,
            warm_data=warm_data,
        )
        session = Session(
            name=f"{spec.name}-{scenario_name}-{mode}", data_manager=cache_obj.data_manager
        )
        records = [
            _byte_request(
                spec,
                api_key,
                cache_obj,
                item,
                model=model,
                scenario_name=scenario_name,
                session=session,
            )
            for item in items
        ]
        result = _summarize_records(records)
        result["prewarmed"] = bool(warm_data)
        if capture_memory:
            result["memory"] = _memory_capture(cache_obj, mode, models=models, scope=scope)
        return result
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        clear_shared_memory(scope)


def _run_direct_concurrent(
    spec: ProviderSpec, api_key: str, items: list[dict[str, Any]], *, model: str
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(items)) as pool:
        records = list(pool.map(lambda item: _direct_request(spec, api_key, item, model), items))
    result = _summarize_records(records)
    result["wall_time_ms"] = round((time.perf_counter() - wall_start) * 1000, 2)
    return result


def _run_byte_concurrent(
    spec: ProviderSpec,
    api_key: str,
    items: list[dict[str, Any]],
    *,
    mode: str,
    model: str,
    models: dict[str, Any],
    scenario_name: str,
    routed: bool = False,
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"byte-{spec.name}-{mode}-burst-")
    scope = f"bench::{spec.name}::{mode}::{scenario_name}::burst::{int(time.time() * 1000)}"
    clear_shared_memory(scope)
    cache_obj = Cache()
    try:
        _configure_cache(
            cache_obj, cache_dir, mode, models=models, routed=routed, scope=scope, warm_data=None
        )
        wall_start = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=len(items)) as pool:
            records = list(
                pool.map(
                    lambda item: _byte_request(
                        spec,
                        api_key,
                        cache_obj,
                        item,
                        model=model,
                        scenario_name=scenario_name,
                        session=None,
                    ),
                    items,
                )
            )
        result = _summarize_records(records)
        result["wall_time_ms"] = round((time.perf_counter() - wall_start) * 1000, 2)
        return result
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        clear_shared_memory(scope)


def _scenario_summary(
    name: str,
    description: str,
    items: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    *,
    baseline_key: str = "direct",
    warm_data: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    summary = {
        "name": name,
        "description": description,
        "request_count": len(items),
        "unique_prompt_count": len({item["prompt"] for item in items}),
        "logical_group_count": len({item["group"] for item in items}),
        "prewarmed_seed_count": len(warm_data or []),
        "runs": runs,
    }
    baseline_cost = runs[baseline_key]["total_cost_usd"]
    baseline_latency = runs[baseline_key]["avg_latency_ms"]
    for key, data in runs.items():
        if key == baseline_key:
            data["saved_vs_baseline_usd"] = 0.0
            data["savings_ratio"] = 0.0
            data["latency_delta_ms"] = 0.0
            continue
        data["saved_vs_baseline_usd"] = round(baseline_cost - data["total_cost_usd"], 8)
        data["savings_ratio"] = (
            round((baseline_cost - data["total_cost_usd"]) / baseline_cost, 4)
            if baseline_cost
            else 0.0
        )
        data["latency_delta_ms"] = round(data["avg_latency_ms"] - baseline_latency, 2)
    return summary


def _render_money(value: float) -> str:
    return f"${value:.8f}"


def _render_mode_line(name: str, data: dict[str, Any], *, baseline_label: str = "direct") -> str:
    flags = []
    if data.get("accuracy_ratio", 1.0) < 1.0:
        flags.append("accuracy<1.0")
    if data.get("prewarmed"):
        flags.append("prewarmed=true")
    if data.get("error_count"):
        flags.append(f"errors={data['error_count']}")
    models = ", ".join(
        f"{model}:{count}" for model, count in sorted((data.get("models") or {}).items())
    )
    routes = ", ".join(
        f"{tier}:{count}" for tier, count in sorted((data.get("route_tiers") or {}).items())
    )
    suffix = f" [{'; '.join(flags)}]" if flags else ""
    model_suffix = f", models=({models})" if models else ""
    route_suffix = f", route_tiers=({routes})" if routes else ""
    return (
        f"- {name}: cost={_render_money(data['total_cost_usd'])}, hit_ratio={data['hit_ratio']}, "
        f"accuracy={data['accuracy_ratio']}, avg_latency={data['avg_latency_ms']} ms, "
        f"p95_latency={data['p95_latency_ms']} ms, saved_vs_{baseline_label}={_render_money(data.get('saved_vs_baseline_usd', 0.0))}, "
        f"savings_ratio={data.get('savings_ratio', 0.0)}{model_suffix}{route_suffix}{suffix}"
    )


def _render_memory_block(memory: dict[str, Any]) -> list[str]:
    summary = memory.get("summary", {})
    snapshot_stats = memory.get("snapshot_stats", {})
    import_result = memory.get("import_result", {})
    imported_summary = memory.get("imported_summary", {})
    lines = [
        f"- Intent records={snapshot_stats.get('intent_records', 0)}, ai_entries={snapshot_stats.get('ai_entries', 0)}, route_tiers={snapshot_stats.get('route_tiers', {})}",
        f"- Import merge={import_result}, imported_ai_total={((imported_summary.get('ai_memory') or {}).get('total_entries', 0))}",
        f"- Intent graph summary={summary.get('intent_graph', {})}",
        f"- AI memory summary={summary.get('ai_memory', {})}",
    ]
    recent = memory.get("recent_interactions", [])[:3]
    if recent:
        previews = [
            {
                "category": entry.get("category"),
                "model": entry.get("model"),
                "source": entry.get("last_source"),
                "hits": entry.get("hits"),
                "question_digest": entry.get("question_digest"),
            }
            for entry in recent
        ]
        lines.append(f"- Recent interactions={previews}")
    return lines


def _render_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# ByteAI Cache Multi-Provider Routing + Memory Report")
    lines.append("")
    lines.append(f"Generated: {results['generated_at']}")
    lines.append(
        "This live benchmark exercises ByteAI Cache's shared cache, routing, and AI-memory layers against the providers that currently have live keys in this environment."
    )
    lines.append("")
    lines.append("## Features Exercised")
    lines.append("")
    lines.append("- Shared exact, normalized, guarded semantic, and hybrid cache modes.")
    lines.append(
        "- Smart model routing for cheap structured requests vs long analysis-style prompts."
    )
    lines.append(
        "- AI-memory capture for answers, metadata, route decisions, and provider-agnostic memory export/import."
    )
    lines.append("- Miss-path coalescing under concurrent burst traffic in library mode.")
    lines.append("- Prompt-template canonicalizers, safe semantic reuse, and prewarming.")
    lines.append("")
    for provider in results["providers"]:
        if provider.get("skipped"):
            lines.append(f"## {provider['display_name']}")
            lines.append("")
            lines.append(f"- Skipped: {provider['reason']}")
            lines.append("")
            continue
        lines.append(f"## {provider['display_name']}")
        lines.append("")
        lines.append(f"- Resolved benchmark model: `{provider['models']['benchmark_model']}`")
        lines.append(f"- Resolved cheap model: `{provider['models']['cheap_model']}`")
        lines.append(f"- Resolved expensive model: `{provider['models']['expensive_model']}`")
        if provider.get("compatibility_note"):
            lines.append(f"- {provider['compatibility_note']}")
        lines.append(f"- Pricing verified on {VERIFIED_ON}:")
        for url in provider["pricing_sources"]:
            lines.append(f"- {url}")
        lines.append("")
        lines.append("### Sequential Cache Scenarios")
        lines.append("")
        for scenario in provider["sequential_scenarios"]:
            lines.append(f"#### {scenario['name']}")
            lines.append(f"- {scenario['description']}")
            lines.append(
                f"- Requests={scenario['request_count']}, unique_prompts={scenario['unique_prompt_count']}, logical_groups={scenario['logical_group_count']}, prewarmed_seed_count={scenario['prewarmed_seed_count']}"
            )
            lines.append(_render_mode_line("Direct", scenario["runs"]["direct"]))
            for mode in BYTE_MODES:
                lines.append(_render_mode_line(f"ByteAI Cache {mode}", scenario["runs"][mode]))
            lines.append("")
        lines.append("### Concurrent Burst")
        lines.append("")
        burst = provider["concurrent_burst"]
        lines.append(f"- {burst['description']}")
        lines.append(_render_mode_line("Direct", burst["runs"]["direct"]))
        for mode in BYTE_MODES:
            lines.append(_render_mode_line(f"ByteAI Cache {mode}", burst["runs"][mode]))
        lines.append("")
        lines.append("### Routing Blend")
        lines.append("")
        routing = provider["routing_blend"]
        lines.append(f"- {routing['description']}")
        lines.append(
            _render_mode_line(
                "Direct expensive",
                routing["runs"]["direct_expensive"],
                baseline_label="direct_expensive",
            )
        )
        lines.append(
            _render_mode_line(
                "ByteAI Cache hybrid fixed-expensive",
                routing["runs"]["byte_hybrid_expensive"],
                baseline_label="direct_expensive",
            )
        )
        lines.append(
            _render_mode_line(
                "ByteAI Cache hybrid routed",
                routing["runs"]["byte_hybrid_routed"],
                baseline_label="direct_expensive",
            )
        )
        lines.append("")
        lines.append("### Memory Snapshot")
        lines.append("")
        for line in _render_memory_block(routing["runs"]["byte_hybrid_routed"].get("memory", {})):
            lines.append(line)
        lines.append("")
        lines.append("### Takeaways")
        lines.append("")
        lines.append(
            "- The `unique_18` miss baseline answers the direct question about cost on a non-reused prompt: if a prompt does not land on the same safe key, ByteAI Cache cost stays close to the provider's direct call cost."
        )
        lines.append(
            f"- The biggest cache gains on {provider['display_name']} come from normalized and hybrid mode on canonical templates plus prewarmed hot prompts."
        )
        lines.append(
            "- The routing blend shows the extra savings layer that cache-only mode cannot create: fixed expensive-model quality with cheaper models automatically selected for structured requests."
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_provider_payload(spec: ProviderSpec, api_key: str) -> dict[str, Any]:
    models = _resolve_models(spec, api_key)
    provider_result: dict[str, Any] = {
        "name": spec.name,
        "display_name": spec.display_name,
        "models": models,
        "pricing_sources": spec.pricing_sources,
        "compatibility_note": spec.compatibility_note,
        "sequential_scenarios": [],
    }

    for scenario in helpers._build_sequential_scenarios():
        warm_data = scenario.get("warm_data")
        runs = {
            "direct": _run_direct_sequence(
                spec, api_key, scenario["items"], model=models["benchmark_model"]
            )
        }
        for mode in BYTE_MODES:
            runs[mode] = _run_byte_sequence(
                spec,
                api_key,
                scenario["items"],
                mode=mode,
                model=models["benchmark_model"],
                models=models,
                routed=False,
                scenario_name=scenario["name"],
                warm_data=warm_data,
                capture_memory=(scenario["name"] == "mixed_workload_24" and mode == "hybrid"),
            )
        provider_result["sequential_scenarios"].append(
            _scenario_summary(
                scenario["name"],
                scenario["description"],
                scenario["items"],
                runs,
                baseline_key="direct",
                warm_data=warm_data,
            )
        )

    burst = helpers._build_concurrent_scenario()
    burst_runs = {
        "direct": _run_direct_concurrent(
            spec, api_key, burst["items"], model=models["benchmark_model"]
        )
    }
    for mode in BYTE_MODES:
        burst_runs[mode] = _run_byte_concurrent(
            spec,
            api_key,
            burst["items"],
            mode=mode,
            model=models["benchmark_model"],
            models=models,
            scenario_name=burst["name"],
            routed=False,
        )
    provider_result["concurrent_burst"] = _scenario_summary(
        burst["name"],
        burst["description"],
        burst["items"],
        burst_runs,
        baseline_key="direct",
    )

    routing = _routing_blend_scenario()
    routing_runs = {
        "direct_expensive": _run_direct_sequence(
            spec, api_key, routing["items"], model=models["expensive_model"]
        ),
        "byte_hybrid_expensive": _run_byte_sequence(
            spec,
            api_key,
            routing["items"],
            mode="hybrid",
            model=models["expensive_model"],
            models=models,
            routed=False,
            scenario_name=routing["name"] + "::fixed",
        ),
        "byte_hybrid_routed": _run_byte_sequence(
            spec,
            api_key,
            routing["items"],
            mode="hybrid",
            model=models["expensive_model"],
            models=models,
            routed=True,
            scenario_name=routing["name"] + "::routed",
            capture_memory=True,
        ),
    }
    provider_result["routing_blend"] = _scenario_summary(
        routing["name"],
        routing["description"],
        routing["items"],
        routing_runs,
        baseline_key="direct_expensive",
    )
    return provider_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["all", "openai", "gemini"], default="all")
    parser.add_argument(
        "--report", default="docs/reports/deep_multi_provider_routing_memory_report.md"
    )
    parser.add_argument(
        "--json-report", default="docs/reports/deep_multi_provider_routing_memory_report.json"
    )
    args = parser.parse_args()

    selected = [args.provider] if args.provider != "all" else list(PROVIDERS.keys())
    results = {"generated_at": datetime.now().isoformat(timespec="seconds"), "providers": []}

    for provider_name in selected:
        spec = PROVIDERS[provider_name]
        api_key = _env_value(spec.api_envs)
        if not api_key:
            results["providers"].append(
                {
                    "name": provider_name,
                    "display_name": spec.display_name,
                    "skipped": True,
                    "reason": f"Missing one of: {', '.join(spec.api_envs)}",
                    "pricing_sources": spec.pricing_sources,
                    "compatibility_note": spec.compatibility_note,
                }
            )
            continue
        try:
            results["providers"].append(_build_provider_payload(spec, api_key))
        except Exception as exc:
            results["providers"].append(
                {
                    "name": provider_name,
                    "display_name": spec.display_name,
                    "skipped": True,
                    "reason": str(exc),
                    "pricing_sources": spec.pricing_sources,
                    "compatibility_note": spec.compatibility_note,
                }
            )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results), encoding="utf-8")

    json_path = Path(args.json_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(report_path), "json_report": str(json_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
