from __future__ import annotations

import argparse
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from byte.benchmarking import systems as benchmark_systems
from byte.benchmarking.contracts import BenchmarkItem, OutputContract, RunPhase
from byte.benchmarking.corpus import load_profile
from byte.benchmarking.runner import run_suite
from byte.benchmarking.systems import provider_key_env
from byte.config import Config

DEFAULT_SYSTEMS = ["direct", "native_cache", "byte"]
DEFAULT_PHASES = [RunPhase.COLD.value, RunPhase.WARM_100.value]
DEFAULT_OUT_DIR = "artifacts/benchmarks"
DEFAULT_MAX_ITEMS_PER_FAMILY = 3

_BACKEND_MAP = {
    "openai": "byte._backends.openai",
    "anthropic": "byte._backends.anthropic",
    "deepseek": "byte._backends.deepseek",
}
_BENCHMARK_ITEM_CONTEXT = threading.local()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark.py")
    parser.add_argument("--provider", default="openai", choices=sorted(_BACKEND_MAP))
    parser.add_argument("--profile", default="tier1")
    parser.add_argument("--systems", default="")
    parser.add_argument("--phases", default="")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--live", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--max-items-per-family", type=int, default=DEFAULT_MAX_ITEMS_PER_FAMILY)
    parser.add_argument("--concurrency", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_product_benchmark(
        provider=str(args.provider),
        profile=str(args.profile),
        systems=_parse_csv(args.systems) or list(DEFAULT_SYSTEMS),
        phases=_parse_csv(args.phases) or list(DEFAULT_PHASES),
        out_dir=str(args.out_dir),
        live=str(args.live),
        compare_baseline=bool(args.compare_baseline),
        max_items_per_family=int(args.max_items_per_family),
        concurrency=int(args.concurrency),
    )
    return 0


def run_product_benchmark(
    *,
    provider: str = "openai",
    profile: str = "tier1",
    systems: list[str] | None = None,
    phases: list[str] | None = None,
    out_dir: str = DEFAULT_OUT_DIR,
    live: str = "auto",
    compare_baseline: bool = False,
    max_items_per_family: int = DEFAULT_MAX_ITEMS_PER_FAMILY,
    concurrency: int = 4,
) -> dict[str, Any]:
    selected_systems = list(systems or DEFAULT_SYSTEMS)
    if compare_baseline and not systems:
        selected_systems = list(DEFAULT_SYSTEMS)
    selected_phases = list(phases or DEFAULT_PHASES)
    root = Path(out_dir) / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=True)

    local_results = run_local_comparison(
        provider=provider,
        profile=profile,
        systems=selected_systems,
        phases=selected_phases,
        out_dir=root / "local",
        max_items_per_family=max_items_per_family,
        concurrency=concurrency,
    )
    rows = _summary_rows(local_results, section="local")

    live_results: dict[str, Any] | None = None
    live_status = "skipped"
    if live == "on" or (live == "auto" and _provider_key_available(provider)):
        try:
            live_results = run_live_comparison(
                provider=provider,
                profile=profile,
                systems=selected_systems,
                phases=selected_phases,
                out_dir=root / "live",
                max_items_per_family=max_items_per_family,
                concurrency=concurrency,
            )
            rows.extend(_summary_rows(live_results, section="live"))
            live_status = "completed"
        except Exception as exc:  # pylint: disable=broad-except
            if live == "on":
                raise
            live_status = f"failed: {exc}"
    else:
        live_status = f"skipped: missing {provider_key_env(provider)}"

    summary = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "provider": provider,
        "profile": profile,
        "systems": selected_systems,
        "phases": selected_phases,
        "live": live,
        "live_status": live_status,
        "rows": rows,
        "artifacts": {
            "root": str(root),
            "summary_json": str(root / "summary.json"),
            "summary_markdown": str(root / "summary.md"),
            "local": local_results.get("artifacts", {}),
            "live": (live_results or {}).get("artifacts", {}),
        },
    }
    _write_summary_artifacts(root, summary)
    _print_summary(summary)
    return summary


def run_local_comparison(
    *,
    provider: str,
    profile: str,
    systems: list[str],
    phases: list[str],
    out_dir: Path,
    max_items_per_family: int,
    concurrency: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_data = load_profile(
        profile,
        providers=[provider],
        max_items_per_family=max_items_per_family,
    )
    item_index = {item.item_id: item for item in profile_data.get("items", [])}
    prompt_index = {_item_prompt_signature(item): item for item in profile_data.get("items", [])}
    with _patched_backend(provider, item_index, prompt_index):
        return run_suite(
            profile=profile,
            providers=[provider],
            systems=systems,
            phases=phases,
            out_dir=str(out_dir),
            fail_on_thresholds=False,
            max_items_per_family=max_items_per_family,
            concurrency=concurrency,
        )


def run_live_comparison(
    *,
    provider: str,
    profile: str,
    systems: list[str],
    phases: list[str],
    out_dir: Path,
    max_items_per_family: int,
    concurrency: int,
) -> dict[str, Any]:
    env_key = provider_key_env(provider)
    if not _provider_key_available(provider):
        raise RuntimeError(f"missing {env_key}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_suite(
        profile=profile,
        providers=[provider],
        systems=systems,
        phases=phases,
        out_dir=str(out_dir),
        fail_on_thresholds=False,
        max_items_per_family=max_items_per_family,
        concurrency=concurrency,
    )


def _summary_rows(results: dict[str, Any], *, section: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider_name, provider_payload in (results.get("providers", {}) or {}).items():
        systems = dict(provider_payload.get("systems", {}) or {})
        for system_name, system_payload in systems.items():
            phases = dict(system_payload.get("phases", {}) or {})
            for phase_name, phase_payload in phases.items():
                summary = dict(phase_payload.get("summary", {}) or {})
                comparison = dict(phase_payload.get("comparison_to_direct", {}) or {})
                rows.append(
                    {
                        "section": section,
                        "provider": provider_name,
                        "system": system_name,
                        "phase": phase_name,
                        "requests": int(summary.get("request_count", 0) or 0),
                        "hit_rate": float(summary.get("actual_reuse_rate", 0.0) or 0.0),
                        "p50_ms": float(summary.get("p50_latency_ms", 0.0) or 0.0),
                        "p95_ms": float(summary.get("p95_latency_ms", 0.0) or 0.0),
                        "cost_delta_pct": round(
                            float(comparison.get("cost_reduction_ratio", 0.0) or 0.0) * 100.0,
                            2,
                        ),
                        "quality_pass_rate": float(summary.get("safe_answer_rate", 0.0) or 0.0),
                    }
                )
    return rows


def _write_summary_artifacts(root: Path, summary: dict[str, Any]) -> None:
    _write_utf8_text(root / "summary.json", json.dumps(summary, indent=2) + "\n")
    markdown_lines = [
        "# ByteAI Benchmark Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Provider: `{summary['provider']}`",
        f"- Profile: `{summary['profile']}`",
        f"- Live status: `{summary['live_status']}`",
        "",
    ]
    for section_name in ("local", "live"):
        section_rows = [row for row in summary.get("rows", []) if row.get("section") == section_name]
        markdown_lines.append(f"## {section_name.title()} Comparison")
        markdown_lines.append("")
        if not section_rows:
            markdown_lines.append(f"{section_name} skipped.")
            markdown_lines.append("")
            continue
        markdown_lines.extend(_markdown_table(section_rows))
        markdown_lines.append("")
    _write_utf8_text(root / "summary.md", "\n".join(markdown_lines).rstrip() + "\n")


def _print_summary(summary: dict[str, Any]) -> None:
    print("ByteAI benchmark")
    print(f"Provider: {summary['provider']} | Profile: {summary['profile']}")
    for section_name in ("local", "live"):
        section_rows = [row for row in summary.get("rows", []) if row.get("section") == section_name]
        if not section_rows:
            if section_name == "live":
                print(f"live skipped ({summary['live_status']})")
            continue
        print("")
        print(f"{section_name.upper()} comparison")
        print(_ascii_table(section_rows))
    print("")
    print(f"Artifacts: {summary['artifacts']['root']}")


def _write_utf8_text(path: Path, text: str) -> None:
    """Write stable UTF-8 text with LF line endings."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _ascii_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "system",
        "phase",
        "requests",
        "hit_rate",
        "p50_ms",
        "p95_ms",
        "cost_delta_pct",
        "quality_pass_rate",
    ]
    display_rows = [[_format_cell(row, header) for header in headers] for row in rows]
    widths = [
        max(len(header), max((len(display_row[index]) for display_row in display_rows), default=0))
        for index, header in enumerate(headers)
    ]
    lines = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        " | ".join("-" * widths[index] for index in range(len(headers))),
    ]
    for display_row in display_rows:
        lines.append(
            " | ".join(display_row[index].ljust(widths[index]) for index in range(len(headers)))
        )
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "system",
        "phase",
        "requests",
        "hit_rate",
        "p50_ms",
        "p95_ms",
        "cost_delta_pct",
        "quality_pass_rate",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row, header) for header in headers) + " |")
    return lines


def _format_cell(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if key == "requests":
        return str(int(value or 0))
    if key in {"hit_rate", "quality_pass_rate"}:
        return f"{float(value or 0.0):.2%}"
    if key in {"p50_ms", "p95_ms"}:
        return f"{float(value or 0.0):.2f}"
    if key == "cost_delta_pct":
        return f"{float(value or 0.0):.2f}%"
    return str(value or "")


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _provider_key_available(provider: str) -> bool:
    return bool(str(os.getenv(provider_key_env(provider), "")).strip())


@contextmanager
def _patched_backend(
    provider: str,
    items: dict[str, BenchmarkItem],
    prompt_index: dict[str, BenchmarkItem],
) -> Any:
    backend = import_module(_BACKEND_MAP[provider]).ChatCompletion
    env_key = provider_key_env(provider)
    previous_env = os.getenv(env_key)
    previous_llm = backend.llm
    previous_base_cache_config = benchmark_systems._base_cache_config
    previous_byte_request = benchmark_systems._byte_request
    native_cache_keys: set[str] = set()
    os.environ[env_key] = previous_env or "byte-local-benchmark"
    backend.llm = _build_fake_llm(
        items,
        prompt_index=prompt_index,
        native_cache_keys=native_cache_keys,
    )
    benchmark_systems._base_cache_config = _local_benchmark_cache_config
    benchmark_systems._byte_request = _wrap_byte_request(
        previous_byte_request,
        items=items,
        prompt_index=prompt_index,
    )
    try:
        yield
    finally:
        backend.llm = previous_llm
        benchmark_systems._base_cache_config = previous_base_cache_config
        benchmark_systems._byte_request = previous_byte_request
        if previous_env is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous_env


def _build_fake_llm(
    items: dict[str, BenchmarkItem],
    *,
    prompt_index: dict[str, BenchmarkItem],
    native_cache_keys: set[str],
) -> Any:
    def _fake_llm(*args, **kwargs) -> dict[str, Any]:  # pylint: disable=unused-argument
        metadata = dict((kwargs.get("byte_memory") or {}).get("metadata", {}) or {})
        item_id = str(metadata.get("benchmark_item_id", "") or "")
        item = items.get(item_id)
        if item is None:
            item = prompt_index.get(_prompt_signature(kwargs))
        if item is not None:
            _BENCHMARK_ITEM_CONTEXT.current_item = item
        else:
            item = getattr(_BENCHMARK_ITEM_CONTEXT, "current_item", None)
        if item is None:
            raise ValueError(f"Unknown benchmark item id: {item_id}")
        response_text = _response_text_for_item(item)
        prompt_tokens = max(16, len(_prompt_signature(kwargs)) // 4)
        completion_tokens = max(8, len(response_text) // 4)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        prompt_cache_key = str(kwargs.get("prompt_cache_key", "") or "")
        if prompt_cache_key:
            if prompt_cache_key in native_cache_keys:
                usage["prompt_tokens_details"] = {"cached_tokens": prompt_tokens}
                usage["prompt_cache_hit_tokens"] = prompt_tokens
                usage["prompt_cache_miss_tokens"] = 0
            else:
                native_cache_keys.add(prompt_cache_key)
                usage["prompt_cache_hit_tokens"] = 0
                usage["prompt_cache_miss_tokens"] = prompt_tokens
        return {
            "id": f"byte-benchmark-{item.item_id}",
            "object": "chat.completion",
            "created": int(datetime.utcnow().timestamp()),
            "model": str(kwargs.get("model", "") or item.model_hint or ""),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": response_text},
                }
            ],
            "usage": usage,
        }

    return _fake_llm


def _response_text_for_item(item: BenchmarkItem) -> str:
    if item.output_contract is OutputContract.JSON_SCHEMA:
        schema = item.expected_value if isinstance(item.expected_value, dict) else {}
        return json.dumps(_example_from_schema(schema), sort_keys=True)
    return str(item.expected_value)


def _example_from_schema(schema: dict[str, Any]) -> Any:
    schema_type = str(schema.get("type", "") or "")
    if schema_type == "object":
        properties = dict(schema.get("properties", {}) or {})
        required = list(schema.get("required", []) or [])
        payload = {}
        for key in required or properties.keys():
            payload[key] = _example_from_schema(dict(properties.get(key, {}) or {}))
        return payload
    if schema_type == "array":
        item_schema = dict(schema.get("items", {}) or {})
        return [_example_from_schema(item_schema)]
    if schema_type == "string":
        return "byte"
    if schema_type == "number":
        return 1.0
    if schema_type == "integer":
        return 1
    if schema_type == "boolean":
        return True
    return {}


def _prompt_signature(request_kwargs: dict[str, Any]) -> str:
    messages = list(request_kwargs.get("messages") or [])
    if messages:
        return "\n".join(str(message.get("content", "") or "") for message in messages)
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    return ""


def _item_prompt_signature(item: BenchmarkItem) -> str:
    return _prompt_signature(dict(item.input_payload or {}))


def _wrap_byte_request(original_request, *, items: dict[str, BenchmarkItem], prompt_index: dict[str, BenchmarkItem]) -> Any:
    def _wrapped(provider: str, request_kwargs: dict[str, Any], *, cache_obj) -> Any:
        metadata = dict((request_kwargs.get("byte_memory") or {}).get("metadata", {}) or {})
        item_id = str(metadata.get("benchmark_item_id", "") or "")
        current_item = items.get(item_id) or prompt_index.get(_prompt_signature(request_kwargs))
        if current_item is not None:
            _BENCHMARK_ITEM_CONTEXT.current_item = current_item
        try:
            return original_request(provider, request_kwargs, cache_obj=cache_obj)
        finally:
            if hasattr(_BENCHMARK_ITEM_CONTEXT, "current_item"):
                delattr(_BENCHMARK_ITEM_CONTEXT, "current_item")

    return _wrapped


def _local_benchmark_cache_config(scope: str) -> Config:
    return Config(
        enable_token_counter=False,
        similarity_threshold=1.0,
        semantic_min_token_overlap=1.0,
        semantic_max_length_ratio=1.0,
        semantic_enforce_canonical_match=True,
        adaptive_threshold=False,
        cache_admission_min_score=1.0,
        native_prompt_caching=True,
        native_prompt_cache_min_chars=0,
        intent_memory=False,
        execution_memory=False,
        reasoning_memory=False,
        verified_reuse_for_coding=False,
        verified_reuse_for_all=False,
        delta_generation=False,
        planner_enabled=False,
        failure_memory=False,
        tenant_policy_learning=False,
        context_compiler=False,
        dynamic_context_budget=False,
        negative_context_memory=False,
        prompt_distillation=False,
        prompt_distillation_mode="disabled",
        prompt_module_mode="disabled",
        evidence_verification=True,
        unique_output_guard=True,
        context_only_unique_prompts=True,
        grounded_context_only=True,
        output_contract_enforcement=True,
        ambiguity_detection=False,
        trust_mode="guarded",
        query_risk_mode="hybrid",
        confidence_mode="calibrated",
        confidence_backend="hybrid",
        semantic_cache_verifier_mode="hybrid",
        semantic_cache_promotion_mode="shadow",
        deterministic_execution=True,
        deterministic_contract_mode="enforced",
        model_namespace=True,
        tool_namespace=True,
        context_fingerprint=True,
        memory_scope=scope,
    )


__all__ = ["build_parser", "main", "run_product_benchmark"]
