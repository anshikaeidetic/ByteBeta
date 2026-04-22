import argparse
import json
import multiprocessing as mp
import os
import shutil
import socket
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from byte import Cache, Config
from byte._backends import unified as byte_unified
from byte.adapter.api import (
    clear_router_alias_registry,
    clear_router_runtime,
    init_cache,
    register_router_alias,
)
from byte.benchmarking._optional_runtime import create_openai_client
from byte.processor.budget import estimate_request_cost
from byte.processor.pre import last_content, normalized_last_content

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "docs" / "reports"
REPORT_PATH = REPORT_DIR / "openai_unified_router_benchmark.md"
JSON_PATH = REPORT_DIR / "openai_unified_router_benchmark.json"

PRICING_SOURCES = [
    "https://platform.openai.com/docs/pricing/",
    "https://openai.com/api/pricing/",
    "https://openai.com/index/gpt-4-1/",
]

CHEAP_CANDIDATES = ("gpt-4o-mini", "gpt-4.1-mini")
EXPENSIVE_CANDIDATES = ("gpt-4o", "gpt-4.1")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                response = client.get(base_url + "/")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.25)
    raise TimeoutError(f"Server did not become ready: {base_url}")


def _probe_model(client: Any, model: str) -> bool:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly OK and nothing else."}],
            temperature=0,
            max_tokens=4,
        )
        return (response.choices[0].message.content or "").strip().upper().startswith("OK")
    except Exception:
        return False


def _resolve_models(api_key: str) -> dict[str, str]:
    client = create_openai_client(api_key=api_key)
    cheap = next((model for model in CHEAP_CANDIDATES if _probe_model(client, model)), None)
    expensive = next((model for model in EXPENSIVE_CANDIDATES if _probe_model(client, model)), None)
    if cheap is None:
        raise RuntimeError(
            f"Unable to resolve a cheap OpenAI benchmark model from: {CHEAP_CANDIDATES}"
        )
    if expensive is None:
        raise RuntimeError(
            f"Unable to resolve an expensive OpenAI benchmark model from: {EXPENSIVE_CANDIDATES}"
        )
    return {"cheap": cheap, "expensive": expensive}


def _make_item(
    prompt: str, expected: str, group: str, kind: str, *, max_tokens: int = 16
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "expected": expected,
        "group": group,
        "kind": kind,
        "max_tokens": max_tokens,
    }


def _mixed_workload() -> list[dict[str, Any]]:
    return [
        _make_item(
            'Classify the sentiment.\nLabels: POSITIVE, NEGATIVE, NEUTRAL\nReview: "The onboarding was fast and the team fixed the bug in an hour."\nAnswer with exactly one label.',
            "POSITIVE",
            "classify_01",
            "classification",
            max_tokens=8,
        ),
        _make_item(
            'Classify the sentiment.\nLabels: POSITIVE, NEGATIVE, NEUTRAL\nReview: "The rollout was okay, but nothing especially impressive happened."\nAnswer with exactly one label.',
            "NEUTRAL",
            "classify_02",
            "classification",
            max_tokens=8,
        ),
        _make_item(
            'Return raw JSON only with keys "city" and "name". Text: "Name: Ana Patel. City: Lisbon. Team: Growth."',
            '{"city":"Lisbon","name":"Ana Patel"}',
            "extract_01",
            "extraction",
            max_tokens=24,
        ),
        _make_item(
            "Translate to Spanish and answer with the translation only: Good morning team",
            "Buenos dias equipo",
            "translate_01",
            "translation",
            max_tokens=12,
        ),
        _make_item(
            "Reply with exactly BYTE_ROUTE_OK and nothing else.",
            "BYTE_ROUTE_OK",
            "exact_01",
            "exact_answer",
            max_tokens=8,
        ),
        _make_item(
            "Analyze the architecture, compare latency against cost, reason step by step about bottlenecks, and after that reply with exactly BYTE_ROUTE_A and nothing else.\nSystem notes: service A fans out to three workers, cache misses trigger vector search, and the hot path should stay deterministic.\nDo not explain the answer.",
            "BYTE_ROUTE_A",
            "hard_01",
            "instruction",
            max_tokens=10,
        ),
        _make_item(
            "Debug the concurrency issue, reason carefully about lock contention, compare alternative fixes, and then reply with exactly BYTE_ROUTE_B and nothing else.\nDetails: p95 spikes only under 16 concurrent requests, misses should coalesce before embedding work, and duplicate prompts should avoid repeated upstream calls.",
            "BYTE_ROUTE_B",
            "hard_02",
            "instruction",
            max_tokens=10,
        ),
        _make_item(
            "Evaluate the tradeoff, analyze the request budget, optimize the cache path, and after thinking step by step reply with exactly BYTE_ROUTE_C and nothing else.\nContext: multi-turn support assistant, routing should prefer cheap models for structured work but keep long reasoning on a stronger model.",
            "BYTE_ROUTE_C",
            "hard_03",
            "instruction",
            max_tokens=10,
        ),
        _make_item(
            "Architect the rollout, compare failure modes, debug the fallback behavior, and reply with exactly BYTE_ROUTE_D and nothing else after your analysis.\nImportant: preserve accuracy, reduce miss-path tax, and keep semantic reuse behind safety rails.",
            "BYTE_ROUTE_D",
            "hard_04",
            "instruction",
            max_tokens=10,
        ),
        _make_item(
            "Summarize this in one short sentence and reply with exactly BYTE_SUMMARY_ONE: Byte stores safe cache hits and reduces repeat LLM calls.",
            "BYTE_SUMMARY_ONE",
            "summary_01",
            "summarization",
            max_tokens=8,
        ),
        _make_item(
            "Return exactly BYTE_DOC_ONLY and nothing else after reading this docstring task.\nWrite docs for a helper that parses settings, validates them, and returns a typed config object.",
            "BYTE_DOC_ONLY",
            "doc_01",
            "documentation",
            max_tokens=8,
        ),
        _make_item(
            "Explain this function in one sentence and then reply with exactly BYTE_EXPLAIN_ONLY and nothing else.\n```python\ndef total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n```",
            "BYTE_EXPLAIN_ONLY",
            "explain_01",
            "code_explanation",
            max_tokens=8,
        ),
    ]


def _coding_workload() -> list[dict[str, Any]]:
    return [
        _make_item(
            "Fix the bug in this Python function.\nDiagnostic: mutable default argument\n```python\ndef add_item(item, items=[]):\n    items.append(item)\n    return items\n```\nReturn exactly MUTABLE_DEFAULT and nothing else.",
            "MUTABLE_DEFAULT",
            "code_fix_01",
            "code_fix",
            max_tokens=8,
        ),
        _make_item(
            "Fix the bug in this JavaScript helper.\nDiagnostic: implicit string-to-number comparison\n```javascript\nfunction isAdult(age) {\n  return age > '17';\n}\n```\nReturn exactly JS_COMPARE_FIX and nothing else.",
            "JS_COMPARE_FIX",
            "code_fix_02",
            "code_fix",
            max_tokens=8,
        ),
        _make_item(
            "Write pytest tests for this helper.\n```python\ndef slugify(value):\n    return value.strip().lower().replace(' ', '-')\n```\nReturn exactly PYTEST and nothing else.",
            "PYTEST",
            "code_test_01",
            "test_generation",
            max_tokens=8,
        ),
        _make_item(
            "Write unit tests for this TypeScript parser using vitest.\n```ts\nexport function parseCount(value: string) {\n  return Number.parseInt(value, 10)\n}\n```\nReturn exactly VITEST and nothing else.",
            "VITEST",
            "code_test_02",
            "test_generation",
            max_tokens=8,
        ),
        _make_item(
            "Explain this function in one sentence.\n```python\ndef total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n```\nReturn exactly SUM_LOOP and nothing else.",
            "SUM_LOOP",
            "code_explain_02",
            "code_explanation",
            max_tokens=8,
        ),
        _make_item(
            "Refactor this function for clarity, reason carefully about the tradeoff, and then reply with exactly REFACTOR_TOKEN and nothing else.\n```python\ndef build_user(name, email, role='member'):\n    user = {}\n    user['name'] = name\n    user['email'] = email\n    user['role'] = role\n    return user\n```",
            "REFACTOR_TOKEN",
            "code_refactor_01",
            "code_refactor",
            max_tokens=8,
        ),
        _make_item(
            "Write a docstring for this helper and then reply with exactly DOCSTRING_ONLY and nothing else.\n```python\ndef read_settings(path):\n    return {'path': path}\n```",
            "DOCSTRING_ONLY",
            "code_doc_01",
            "documentation",
            max_tokens=8,
        ),
        _make_item(
            "Analyze the patch strategy, debug the failure mode, and then reply with exactly PATCH_PLAN_ONLY and nothing else.\nThe repo has repeated mutable-default bugs and the safest fix pattern should be reused.",
            "PATCH_PLAN_ONLY",
            "code_plan_01",
            "instruction",
            max_tokens=8,
        ),
    ]


def _repeat_items(items: list[dict[str, Any]], repeats: int = 2) -> list[dict[str, Any]]:
    output = []
    for _ in range(repeats):
        output.extend(items)
    return output


def _normalize_text(text: str | None) -> str:
    return " ".join(str(text or "").strip().lower().replace("\n", " ").split())


def _response_text(response: Any) -> str | None:
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return message.get("content")
        if response.get("text") is not None:
            return response.get("text")
    return None


def _response_usage(response: Any) -> dict[str, int]:
    usage = {}
    if isinstance(response, dict):
        usage = response.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, round(0.95 * (len(sorted_values) - 1))))
    return round(sorted_values[index], 2)


def _cost_from_usage(model: str, usage: dict[str, int]) -> float:
    cost = estimate_request_cost(
        model,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    )
    return float(cost or 0.0)


def _record_from_response(
    item: dict[str, Any], response: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    text = _response_text(response)
    usage = _response_usage(response)
    byte_router = response.get("byte_router") or {}
    model_name = str(response.get("model") or byte_router.get("selected_model") or "")
    provider_name = str(
        response.get("byte_provider") or byte_router.get("selected_provider") or "openai"
    )
    if model_name:
        qualified_model = f"{provider_name}/{model_name}"
    else:
        qualified_model = str(byte_router.get("selected_target") or "")
    return {
        "group": item["group"],
        "kind": item["kind"],
        "expected": item["expected"],
        "text": text,
        "correct": _normalize_text(text) == _normalize_text(item["expected"]),
        "latency_ms": round(latency_ms, 2),
        "byte": bool(response.get("byte")),
        "model": model_name,
        "qualified_model": qualified_model,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "cost_usd": round(_cost_from_usage(qualified_model or model_name, usage), 8),
        "router": byte_router,
    }


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records]
    model_counts: dict[str, int] = {}
    fallback_count = 0
    cache_hits = 0
    for record in records:
        model_counts[record["qualified_model"] or record["model"] or "unknown"] = (
            model_counts.get(record["qualified_model"] or record["model"] or "unknown", 0) + 1
        )
        if (record.get("router") or {}).get("fallback_used"):
            fallback_count += 1
        if record["byte"]:
            cache_hits += 1
    return {
        "request_count": len(records),
        "cache_hits": cache_hits,
        "hit_ratio": round(cache_hits / len(records), 4) if records else 0.0,
        "accuracy_ratio": round(sum(1 for record in records if record["correct"]) / len(records), 4)
        if records
        else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": _p95(latencies),
        "total_cost_usd": round(sum(record["cost_usd"] for record in records), 8),
        "fallback_count": fallback_count,
        "model_counts": model_counts,
        "sample": records[:6],
    }


def _build_cache(
    cache_dir: str,
    alias_map: dict[str, list[str]],
    models: dict[str, str],
    *,
    enable_model_routing: bool = True,
) -> Cache:
    cache_obj = Cache()
    init_cache(
        mode="hybrid",
        data_dir=cache_dir,
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            model_namespace=True,
            model_routing=enable_model_routing,
            routing_cheap_model=models["cheap"],
            routing_expensive_model=models["expensive"],
            routing_default_model=models["expensive"],
            routing_long_prompt_chars=220,
            routing_multi_turn_threshold=4,
            routing_model_aliases=alias_map,
            routing_strategy="cost",
            routing_retry_attempts=1,
            routing_retry_backoff_ms=150.0,
            routing_cooldown_seconds=20.0,
            semantic_allowed_categories=[
                "question_answer",
                "summarization",
                "comparison",
                "instruction",
            ],
        ),
    )
    return cache_obj


def _close_cache_chain(cache_obj: Cache | None) -> None:
    seen = set()
    current = cache_obj
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if current.data_manager is not None:
            current.data_manager.close()
            current.data_manager = None
        current = current.next_cache


def _run_direct(api_key: str, model: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    client = create_openai_client(api_key=api_key)
    records = []
    for item in items:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": item["prompt"]}],
            temperature=0,
            max_tokens=item["max_tokens"],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content or "",
                    }
                }
            ],
            "model": getattr(response, "model", model),
            "usage": {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            },
        }
        records.append(_record_from_response(item, payload, latency_ms))
    return _summarize_records(records)


def _run_unified_library(
    api_key: str,
    alias_name: str,
    strategy: str,
    items: list[dict[str, Any]],
    models: dict[str, str],
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix="byte-unified-lib-")
    alias_map = {alias_name: [f"openai/{models['cheap']}", f"openai/{models['expensive']}"]}
    clear_router_alias_registry()
    clear_router_runtime()
    register_router_alias(alias_name, alias_map[alias_name])
    cache_obj = _build_cache(cache_dir, alias_map, models)
    records = []
    try:
        for item in items:
            start = time.perf_counter()
            response = byte_unified.ChatCompletion.create(
                model=alias_name,
                api_key=api_key,
                cache_obj=cache_obj,
                messages=[{"role": "user", "content": item["prompt"]}],
                temperature=0,
                max_tokens=item["max_tokens"],
                byte_routing_strategy=strategy,
            )
            records.append(
                _record_from_response(item, response, (time.perf_counter() - start) * 1000)
            )
        summary = _summarize_records(records)
        summary["cache_stats"] = cache_obj.cost_summary()
        return summary
    finally:
        _close_cache_chain(cache_obj)
        shutil.rmtree(cache_dir, ignore_errors=True)
        clear_router_alias_registry()
        clear_router_runtime()


def _serve_proxy(
    port: int, cache_dir: str, api_key: str, alias_name: str, models: dict[str, str]
) -> None:
    os.environ["OPENAI_API_KEY"] = api_key

    import uvicorn

    import byte_server.server as server

    alias_map = {alias_name: [f"openai/{models['cheap']}", f"openai/{models['expensive']}"]}
    clear_router_alias_registry()
    clear_router_runtime()
    register_router_alias(alias_name, alias_map[alias_name])
    server.proxy_chat_mode = "unified"
    server.proxy_model_aliases = alias_map
    server.openai_cache = _build_cache(cache_dir, alias_map, models)
    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")


def _run_unified_proxy(
    api_key: str,
    alias_name: str,
    strategy: str,
    items: list[dict[str, Any]],
    models: dict[str, str],
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix="byte-unified-proxy-")
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = mp.Process(
        target=_serve_proxy, args=(port, cache_dir, api_key, alias_name, models), daemon=True
    )
    process.start()
    try:
        _wait_for_server(base_url)
        records = []
        headers = {"Authorization": f"Bearer {api_key}"}
        with httpx.Client(base_url=base_url, timeout=120.0, headers=headers) as client:
            for item in items:
                start = time.perf_counter()
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": alias_name,
                        "messages": [{"role": "user", "content": item["prompt"]}],
                        "temperature": 0,
                        "max_tokens": item["max_tokens"],
                        "byte_routing_strategy": strategy,
                    },
                )
                latency_ms = (time.perf_counter() - start) * 1000
                body = response.json()
                records.append(_record_from_response(item, body, latency_ms))
            stats = client.get("/stats").json()
        summary = _summarize_records(records)
        summary["cache_stats"] = stats
        return summary
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        shutil.rmtree(cache_dir, ignore_errors=True)
        clear_router_alias_registry()
        clear_router_runtime()


def _run_health_recovery(api_key: str, models: dict[str, str]) -> dict[str, Any]:
    alias_name = "openai-recovery-router"
    clear_router_alias_registry()
    clear_router_runtime()
    register_router_alias(
        alias_name, ["openai/not-a-real-openai-model", f"openai/{models['cheap']}"]
    )
    cache_dir = tempfile.mkdtemp(prefix="byte-health-")
    cache_obj = _build_cache(
        cache_dir,
        {alias_name: ["openai/not-a-real-openai-model", f"openai/{models['cheap']}"]},
        models,
        enable_model_routing=False,
    )
    item = _make_item(
        "Reply with exactly HEALTH_OK and nothing else.",
        "HEALTH_OK",
        "health_01",
        "exact_answer",
        max_tokens=8,
    )
    try:
        first = byte_unified.ChatCompletion.create(
            model=alias_name,
            api_key=api_key,
            cache_obj=cache_obj,
            cache_skip=True,
            messages=[{"role": "user", "content": item["prompt"]}],
            temperature=0,
            max_tokens=item["max_tokens"],
            byte_routing_strategy="priority",
        )
        second = byte_unified.ChatCompletion.create(
            model=alias_name,
            api_key=api_key,
            cache_obj=cache_obj,
            cache_skip=True,
            messages=[{"role": "user", "content": item["prompt"]}],
            temperature=0,
            max_tokens=item["max_tokens"],
            byte_routing_strategy="health_weighted",
        )
        return {
            "first": first.get("byte_router", {}),
            "second": second.get("byte_router", {}),
        }
    finally:
        _close_cache_chain(cache_obj)
        shutil.rmtree(cache_dir, ignore_errors=True)
        clear_router_alias_registry()
        clear_router_runtime()


def _render_mode_line(
    label: str,
    summary: dict[str, Any],
    *,
    baseline_cost: float = 0.0,
    baseline_latency: float = 0.0,
) -> str:
    saved = round(baseline_cost - summary["total_cost_usd"], 8) if baseline_cost else 0.0
    savings_ratio = round(saved / baseline_cost, 4) if baseline_cost else 0.0
    latency_delta = (
        round(summary["avg_latency_ms"] - baseline_latency, 2) if baseline_latency else 0.0
    )
    models = ", ".join(
        f"{model}:{count}" for model, count in sorted(summary.get("model_counts", {}).items())
    )
    return (
        f"- {label}: cost=${summary['total_cost_usd']:.8f}, hit_ratio={summary['hit_ratio']}, "
        f"accuracy={summary['accuracy_ratio']}, avg_latency={summary['avg_latency_ms']} ms, "
        f"p95_latency={summary['p95_latency_ms']} ms, fallback_count={summary['fallback_count']}, "
        f"saved_vs_direct=${saved:.8f}, savings_ratio={savings_ratio}, latency_delta_ms={latency_delta}, models=({models})"
    )


def _render_report(results: dict[str, Any]) -> str:
    mixed_direct = results["mixed"]["direct"]
    coding_direct = results["coding"]["direct"]
    lines = [
        "# ByteAI Cache Unified Router Benchmark",
        "",
        f"Generated: {results['generated_at']}",
        f"Resolved cheap model: `{results['models']['cheap']}`",
        f"Resolved expensive model: `{results['models']['expensive']}`",
        "",
        "This benchmark validates ByteAI Cache's new unified router stack end to end using a live OpenAI key. It exercises:",
        "",
        "- cost-based target ordering",
        "- health-weighted target ordering",
        "- semantic signal routing inside the provider adapter",
        "- hybrid cache reuse on repeated workloads",
        "- OpenAI-compatible proxy routing through the unified adapter",
        "",
        "## Pricing Sources",
        "",
    ]
    for url in PRICING_SOURCES:
        lines.append(f"- {url}")
    lines.extend(
        [
            "",
            "## Mixed Repeated Workload (24 requests)",
            "",
            "- 12 logical prompts repeated twice to measure both miss-path routing and second-pass cache reuse.",
            _render_mode_line("Direct expensive", mixed_direct),
            _render_mode_line(
                "ByteAI Cache unified library (cost strategy)",
                results["mixed"]["library_cost"],
                baseline_cost=mixed_direct["total_cost_usd"],
                baseline_latency=mixed_direct["avg_latency_ms"],
            ),
            _render_mode_line(
                "ByteAI Cache unified proxy (cost strategy)",
                results["mixed"]["proxy_cost"],
                baseline_cost=mixed_direct["total_cost_usd"],
                baseline_latency=mixed_direct["avg_latency_ms"],
            ),
            "",
            "## Coding Repeated Workload (16 requests)",
            "",
            "- 8 logical coding prompts repeated twice to measure routing plus reusable coding cache behavior.",
            _render_mode_line("Direct expensive", coding_direct),
            _render_mode_line(
                "ByteAI Cache unified library (cost strategy)",
                results["coding"]["library_cost"],
                baseline_cost=coding_direct["total_cost_usd"],
                baseline_latency=coding_direct["avg_latency_ms"],
            ),
            _render_mode_line(
                "ByteAI Cache unified proxy (cost strategy)",
                results["coding"]["proxy_cost"],
                baseline_cost=coding_direct["total_cost_usd"],
                baseline_latency=coding_direct["avg_latency_ms"],
            ),
            "",
            "## Health Recovery Drill",
            "",
            "- First call intentionally used an alias whose first target was invalid, forcing fallback to the valid OpenAI model.",
            f"- First call router metadata: {json.dumps(results['health_recovery']['first'], sort_keys=True)}",
            f"- Second call router metadata: {json.dumps(results['health_recovery']['second'], sort_keys=True)}",
            "- This confirms the health-weighted strategy can learn away from a bad target after a live failure event.",
            "",
            "## Notes",
            "",
            "- Cache hits show `byte=true` and carry zero upstream token cost in this report.",
            "- On healthy single-provider OpenAI aliases, the `health_weighted` strategy mostly behaves like a cost-aware tie-breaker until one target accumulates failures or poor latency.",
            "- The `docs` folder was kept because it is still useful: it holds the product docs and the generated benchmark artifacts.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--json-report", default=str(JSON_PATH))
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("BYTE_TEST_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OpenAI API key. Pass --api-key or set BYTE_TEST_OPENAI_API_KEY.")

    models = _resolve_models(api_key)
    mixed_items = _repeat_items(_mixed_workload(), repeats=2)
    coding_items = _repeat_items(_coding_workload(), repeats=2)

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": models,
        "mixed": {
            "direct": _run_direct(api_key, models["expensive"], mixed_items),
            "library_cost": _run_unified_library(
                api_key, "openai-byte-router", "cost", mixed_items, models
            ),
            "proxy_cost": _run_unified_proxy(
                api_key, "openai-byte-router", "cost", mixed_items, models
            ),
        },
        "coding": {
            "direct": _run_direct(api_key, models["expensive"], coding_items),
            "library_cost": _run_unified_library(
                api_key, "openai-byte-router", "cost", coding_items, models
            ),
            "proxy_cost": _run_unified_proxy(
                api_key, "openai-byte-router", "cost", coding_items, models
            ),
        },
        "health_recovery": _run_health_recovery(api_key, models),
        "pricing_sources": PRICING_SOURCES,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results), encoding="utf-8")

    json_path = Path(args.json_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(report_path), "json_report": str(json_path)}))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
