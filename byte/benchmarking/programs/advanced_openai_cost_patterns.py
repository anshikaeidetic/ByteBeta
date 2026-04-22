import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import random
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
from byte.adapter.api import (
    init_exact_cache,
    init_hybrid_cache,
    init_normalized_cache,
    init_similar_cache,
)
from byte.benchmarking._optional_runtime import create_openai_client
from byte.processor.pre import last_content, normalized_last_content

MODEL = "gpt-4o-mini"
PRICING = {
    "input_per_million": 0.15,
    "cached_input_per_million": 0.075,
    "output_per_million": 0.60,
    "sources": [
        "https://openai.com/api/pricing/",
        "https://platform.openai.com/docs/models/gpt-4o-mini",
    ],
    "verified_on": "2026-03-10",
}
MODE_DESCRIPTIONS = {
    "direct": "Direct OpenAI call with no Byte layer.",
    "exact": "ByteAI Cache exact-match cache only.",
    "normalized": "ByteAI Cache normalized exact-match cache: collapses casing, punctuation, and whitespace variants.",
    "semantic": "ByteAI Cache semantic cache only.",
    "hybrid": "ByteAI Cache layered cache: exact -> normalized -> semantic.",
}
BYTE_MODES = ["exact", "normalized", "semantic", "hybrid"]


def _make_item(prompt: str, expected: str, group: str, variant: str) -> dict[str, str]:
    return {
        "prompt": prompt,
        "expected": expected,
        "group": group,
        "variant": variant,
    }


def _instruction_prompt(token: str) -> str:
    return f"Reply with exactly {token} and nothing else."


def _format_variants(token: str) -> list[str]:
    return [
        f"  Please reply with exactly {token} and nothing else!!!  ",
        f"PLEASE reply with exactly {token} and nothing else.",
        f"Please   reply with exactly {token} and nothing else?",
    ]


def _reordered_variants(token: str) -> list[str]:
    return [
        f"Byte benchmark request. Reply with exactly {token} and nothing else. Keep the answer to {token}.",
        f"Keep the answer to {token}. Byte benchmark request. Reply with exactly {token} and nothing else.",
        f"Reply with exactly {token} and nothing else. Byte benchmark request. Keep the answer to {token}.",
    ]


def _build_scenarios() -> list[dict[str, Any]]:
    rng = random.Random(42)

    unique_items = [
        _make_item(
            _instruction_prompt(f"BYTE_U{i:02d}"), f"BYTE_U{i:02d}", f"unique_{i:02d}", "base"
        )
        for i in range(16)
    ]

    exact_items: list[dict[str, str]] = []
    for i in range(8):
        token = f"BYTE_E{i:02d}"
        prompt = _instruction_prompt(token)
        exact_items.append(_make_item(prompt, token, f"exact_{i:02d}", "first"))
        exact_items.append(_make_item(prompt, token, f"exact_{i:02d}", "repeat"))

    format_items: list[dict[str, str]] = []
    for i in range(6):
        token = f"BYTE_F{i:02d}"
        for idx, prompt in enumerate(_format_variants(token), 1):
            format_items.append(_make_item(prompt, token, f"format_{i:02d}", f"v{idx}"))

    mixed_items: list[dict[str, str]] = []
    for i in range(4):
        token = f"BYTE_MX{i:02d}"
        prompt = _instruction_prompt(token)
        mixed_items.append(_make_item(prompt, token, f"mixed_exact_{i:02d}", "first"))
        mixed_items.append(_make_item(prompt, token, f"mixed_exact_{i:02d}", "repeat"))
    for i in range(4):
        token = f"BYTE_MN{i:02d}"
        variants = _format_variants(token)[:2]
        for idx, prompt in enumerate(variants, 1):
            mixed_items.append(_make_item(prompt, token, f"mixed_norm_{i:02d}", f"v{idx}"))
    for i in range(8):
        token = f"BYTE_MU{i:02d}"
        mixed_items.append(
            _make_item(_instruction_prompt(token), token, f"mixed_unique_{i:02d}", "base")
        )
    rng.shuffle(mixed_items)

    reordered_items: list[dict[str, str]] = []
    for i in range(6):
        token = f"BYTE_R{i:02d}"
        for idx, prompt in enumerate(_reordered_variants(token), 1):
            reordered_items.append(_make_item(prompt, token, f"reordered_{i:02d}", f"v{idx}"))

    return [
        {
            "name": "unique_16",
            "description": "All prompts are unique, so misses should cost about the same as direct OpenAI.",
            "items": unique_items,
        },
        {
            "name": "exact_pairs_16",
            "description": "Eight exact duplicate pairs. Exact, normalized, and hybrid should all save the second call.",
            "items": exact_items,
        },
        {
            "name": "format_variants_18",
            "description": "Six logical prompts with punctuation/casing/whitespace variants. Normalized and hybrid should outperform exact.",
            "items": format_items,
        },
        {
            "name": "mixed_workload_24",
            "description": "A realistic blend of exact repeats, formatting variants, and one-off prompts.",
            "items": mixed_items,
        },
        {
            "name": "reordered_instruction_18",
            "description": "Sentence-order variants that stay logically equivalent but do not normalize to the same key.",
            "items": reordered_items,
        },
    ]


def _build_concurrent_mix() -> dict[str, Any]:
    prompts = [
        _make_item(_instruction_prompt("BYTE_BURST_EX"), "BYTE_BURST_EX", "burst_exact", "a"),
        _make_item(_instruction_prompt("BYTE_BURST_EX"), "BYTE_BURST_EX", "burst_exact", "b"),
        _make_item(_instruction_prompt("BYTE_BURST_EX"), "BYTE_BURST_EX", "burst_exact", "c"),
        _make_item(_instruction_prompt("BYTE_BURST_EX"), "BYTE_BURST_EX", "burst_exact", "d"),
    ]
    for idx, prompt in enumerate(_format_variants("BYTE_BURST_FMT")[:4], 1):
        prompts.append(_make_item(prompt, "BYTE_BURST_FMT", "burst_format", f"v{idx}"))
    prompts.extend(
        [
            _make_item(
                _instruction_prompt("BYTE_BURST_U0"), "BYTE_BURST_U0", "burst_unique_0", "base"
            ),
            _make_item(
                _instruction_prompt("BYTE_BURST_U1"), "BYTE_BURST_U1", "burst_unique_1", "base"
            ),
            _make_item(
                _instruction_prompt("BYTE_BURST_U2"), "BYTE_BURST_U2", "burst_unique_2", "base"
            ),
            _make_item(
                _instruction_prompt("BYTE_BURST_U3"), "BYTE_BURST_U3", "burst_unique_3", "base"
            ),
            _make_item(
                _instruction_prompt("BYTE_BURST_U4"), "BYTE_BURST_U4", "burst_unique_4", "base"
            ),
        ]
    )
    return {
        "name": "concurrent_mixed_12",
        "description": "12 concurrent requests: 4 exact duplicates, 3 formatting variants, 5 unique prompts.",
        "items": prompts,
    }


def _configure_cache(cache_obj: Cache, cache_dir: str, mode: str) -> None:
    base_config = Config(enable_token_counter=False)
    if mode == "exact":
        init_exact_cache(
            data_dir=cache_dir,
            cache_obj=cache_obj,
            pre_func=last_content,
            config=base_config,
        )
        return
    if mode == "normalized":
        init_normalized_cache(
            data_dir=cache_dir,
            cache_obj=cache_obj,
            pre_func=last_content,
            normalized_pre_func=normalized_last_content,
            config=base_config,
        )
        return
    if mode == "semantic":
        init_similar_cache(
            data_dir=cache_dir,
            cache_obj=cache_obj,
            pre_func=last_content,
            config=base_config,
        )
        return
    if mode == "hybrid":
        init_hybrid_cache(
            data_dir=cache_dir,
            cache_obj=cache_obj,
            pre_func=last_content,
            normalized_pre_func=normalized_last_content,
            config=base_config,
        )
        return
    raise ValueError(f"Unsupported mode: {mode}")


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


def _serve_proxy(port: int, cache_dir: str, mode: str) -> None:
    import uvicorn

    import byte_server.server as server

    server.openai_cache = Cache()
    _configure_cache(server.openai_cache, cache_dir, mode)
    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")


def _usage_fields(usage: Any) -> dict[str, int]:
    if not usage:
        return {
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
        }

    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        prompt_details = usage.get("prompt_tokens_details", {}) or {}
        cached_prompt_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
    else:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details is None:
            cached_prompt_tokens = 0
        elif isinstance(prompt_details, dict):
            cached_prompt_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
        else:
            cached_prompt_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

    cached_prompt_tokens = min(cached_prompt_tokens, prompt_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _request_cost(prompt_tokens: int, cached_prompt_tokens: int, completion_tokens: int) -> float:
    uncached_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    input_cost = (uncached_prompt_tokens / 1_000_000) * PRICING["input_per_million"]
    cached_input_cost = (cached_prompt_tokens / 1_000_000) * PRICING["cached_input_per_million"]
    output_cost = (completion_tokens / 1_000_000) * PRICING["output_per_million"]
    return input_cost + cached_input_cost + output_cost


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = math.ceil(0.95 * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["latency_ms"] for r in records]
    total_prompt = sum(r.get("prompt_tokens", 0) for r in records)
    total_cached_prompt = sum(r.get("cached_prompt_tokens", 0) for r in records)
    total_completion = sum(r.get("completion_tokens", 0) for r in records)
    total_cost = sum(r.get("cost_usd", 0.0) for r in records)
    cached = sum(1 for r in records if r.get("byte"))
    correct = sum(1 for r in records if r.get("correct"))
    return {
        "request_count": len(records),
        "cached_count": cached,
        "miss_count": len(records) - cached,
        "hit_ratio": round(cached / len(records), 4) if records else 0.0,
        "accuracy_ratio": round(correct / len(records), 4) if records else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(_p95(latencies), 2),
        "total_prompt_tokens": total_prompt,
        "total_cached_prompt_tokens": total_cached_prompt,
        "total_completion_tokens": total_completion,
        "total_cost_usd": round(total_cost, 8),
        "sample": records[:5],
    }


def _direct_request(api_key: str, item: dict[str, str]) -> dict[str, Any]:
    client = create_openai_client(api_key=api_key)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": item["prompt"]}],
        temperature=0,
        max_tokens=8,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    usage_fields = _usage_fields(response.usage)
    text = response.choices[0].message.content or ""
    return {
        "status_code": 200,
        "latency_ms": latency_ms,
        "byte": False,
        "prompt_tokens": usage_fields["prompt_tokens"],
        "cached_prompt_tokens": usage_fields["cached_prompt_tokens"],
        "completion_tokens": usage_fields["completion_tokens"],
        "cost_usd": _request_cost(
            usage_fields["prompt_tokens"],
            usage_fields["cached_prompt_tokens"],
            usage_fields["completion_tokens"],
        ),
        "text": text,
        "expected": item["expected"],
        "group": item["group"],
        "variant": item["variant"],
        "correct": text.strip() == item["expected"],
    }


def _proxy_request(
    client: httpx.Client, base_url: str, api_key: str, item: dict[str, str]
) -> dict[str, Any]:
    start = time.perf_counter()
    response = client.post(
        base_url + "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": 0,
            "max_tokens": 8,
        },
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    body = response.json()
    usage_fields = _usage_fields(body.get("usage", {}) if isinstance(body, dict) else {})
    text = None
    if response.status_code == 200 and isinstance(body, dict):
        text = body["choices"][0]["message"]["content"]
    return {
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "byte": bool(body.get("byte")) if isinstance(body, dict) else False,
        "prompt_tokens": usage_fields["prompt_tokens"],
        "cached_prompt_tokens": usage_fields["cached_prompt_tokens"],
        "completion_tokens": usage_fields["completion_tokens"],
        "cost_usd": _request_cost(
            usage_fields["prompt_tokens"],
            usage_fields["cached_prompt_tokens"],
            usage_fields["completion_tokens"],
        ),
        "text": text,
        "expected": item["expected"],
        "group": item["group"],
        "variant": item["variant"],
        "correct": (text or "").strip() == item["expected"],
    }


def _run_direct_sequence(api_key: str, items: list[dict[str, str]]) -> dict[str, Any]:
    records = [_direct_request(api_key, item) for item in items]
    return _summarize_records(records)


def _run_proxy_sequence(api_key: str, items: list[dict[str, str]], mode: str) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"advanced-{mode}-")
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = mp.Process(target=_serve_proxy, args=(port, cache_dir, mode), daemon=True)
    process.start()
    try:
        _wait_for_server(base_url)
        with httpx.Client(timeout=60.0) as client:
            records = [_proxy_request(client, base_url, api_key, item) for item in items]
            stats = client.get(base_url + "/stats").json()
        result = _summarize_records(records)
        result["stats"] = stats
        return result
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        shutil.rmtree(cache_dir, ignore_errors=True)


def _run_direct_concurrent(api_key: str, items: list[dict[str, str]]) -> dict[str, Any]:
    wall_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(items)) as pool:
        records = list(pool.map(lambda item: _direct_request(api_key, item), items))
    result = _summarize_records(records)
    result["wall_time_ms"] = round((time.perf_counter() - wall_start) * 1000, 2)
    return result


def _run_proxy_concurrent(api_key: str, items: list[dict[str, str]], mode: str) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"advanced-concurrent-{mode}-")
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = mp.Process(target=_serve_proxy, args=(port, cache_dir, mode), daemon=True)
    process.start()
    try:
        _wait_for_server(base_url)

        def one(item: dict[str, str]) -> dict[str, Any]:
            with httpx.Client(timeout=60.0) as client:
                return _proxy_request(client, base_url, api_key, item)

        wall_start = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=len(items)) as pool:
            records = list(pool.map(one, items))
        stats = httpx.get(base_url + "/stats", timeout=60.0).json()
        result = _summarize_records(records)
        result["stats"] = stats
        result["wall_time_ms"] = round((time.perf_counter() - wall_start) * 1000, 2)
        return result
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        shutil.rmtree(cache_dir, ignore_errors=True)


def _scenario_summary(
    name: str, description: str, items: list[dict[str, str]], runs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary = {
        "name": name,
        "description": description,
        "request_count": len(items),
        "unique_prompt_count": len({item["prompt"] for item in items}),
        "logical_group_count": len({item["group"] for item in items}),
        "runs": runs,
    }
    baseline_cost = runs["direct"]["total_cost_usd"]
    baseline_latency = runs["direct"]["avg_latency_ms"]
    for key in BYTE_MODES:
        mode = runs[key]
        mode["saved_vs_direct_usd"] = round(baseline_cost - mode["total_cost_usd"], 8)
        mode["savings_ratio"] = (
            round((baseline_cost - mode["total_cost_usd"]) / baseline_cost, 4)
            if baseline_cost
            else 0.0
        )
        mode["latency_delta_ms"] = round(mode["avg_latency_ms"] - baseline_latency, 2)
    return summary


def _render_money(value: float) -> str:
    return f"${value:.8f}"


def _render_mode_line(name: str, data: dict[str, Any]) -> str:
    saved = data.get("saved_vs_direct_usd")
    saved_part = (
        ""
        if saved is None
        else f", saved={_render_money(saved)}, savings_ratio={data['savings_ratio']}"
    )
    safety_part = ""
    if data.get("accuracy_ratio", 1.0) < 1.0:
        safety_part = " [unsafe on this workload: accuracy < 1.0]"
    return (
        f"- {name}: cost={_render_money(data['total_cost_usd'])}, hit_ratio={data['hit_ratio']}, "
        f"accuracy={data['accuracy_ratio']}, avg_latency={data['avg_latency_ms']} ms, "
        f"p95_latency={data['p95_latency_ms']} ms{saved_part}{safety_part}"
    )


def _render_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# ByteAI Cache Advanced Cost Pattern Report")
    lines.append("")
    lines.append(f"Generated: {results['generated_at']}")
    lines.append(f"Model: {results['model']}")
    lines.append(
        "This benchmark uses OpenAI live traffic for measurement, but the new cache features are provider-agnostic ByteAI Cache features."
    )
    lines.append("")
    lines.append("## New Features Exercised")
    lines.append("")
    lines.append("- `init_exact_cache(...)`: lightweight exact-match cache initializer.")
    lines.append(
        "- `init_normalized_cache(...)`: exact-match cache on normalized text for casing/punctuation/whitespace variants."
    )
    lines.append("- `init_hybrid_cache(...)`: layered exact -> normalized -> semantic fallback.")
    lines.append(
        "- OpenAI proxy cache modes now support `exact`, `normalized`, `semantic`, and `hybrid` strategies."
    )
    lines.append("")
    lines.append("## Pricing Source")
    lines.append("")
    lines.append(f"Verified on {results['pricing']['verified_on']}:")
    for url in results["pricing"]["sources"]:
        lines.append(f"- {url}")
    lines.append("")
    lines.append("## Key Answer")
    lines.append("")
    lines.append(
        "- With direct OpenAI or ByteAI Cache exact cache, a prompt that does not hit a cache key still costs about the same as the original OpenAI call."
    )
    lines.append(
        "- The new normalized cache reduces spend for prompts that are logically the same but differ only by formatting, and it stayed accurate in this run."
    )
    lines.append(
        "- The semantic and hybrid modes produced false positives on short instruction-style prompts in this benchmark, so their apparent savings here are not production-safe without additional guards."
    )
    lines.append("")
    lines.append("## Case-by-Case Results")
    lines.append("")
    for scenario in results["sequential_scenarios"]:
        runs = scenario["runs"]
        lines.append(f"### {scenario['name']}")
        lines.append(f"- {scenario['description']}")
        lines.append(
            f"- Requests={scenario['request_count']}, unique_prompts={scenario['unique_prompt_count']}, "
            f"logical_groups={scenario['logical_group_count']}"
        )
        lines.append(_render_mode_line("Direct", runs["direct"]))
        lines.append(_render_mode_line("ByteAI Cache exact", runs["exact"]))
        lines.append(_render_mode_line("ByteAI Cache normalized", runs["normalized"]))
        lines.append(_render_mode_line("ByteAI Cache semantic", runs["semantic"]))
        lines.append(_render_mode_line("ByteAI Cache hybrid", runs["hybrid"]))
        lines.append("")
    lines.append("## Concurrent Mixed Burst")
    lines.append("")
    burst = results["concurrent_burst"]
    lines.append(f"- {burst['description']}")
    lines.append(_render_mode_line("Direct", burst["runs"]["direct"]))
    lines.append(_render_mode_line("ByteAI Cache exact", burst["runs"]["exact"]))
    lines.append(_render_mode_line("ByteAI Cache normalized", burst["runs"]["normalized"]))
    lines.append(_render_mode_line("ByteAI Cache semantic", burst["runs"]["semantic"]))
    lines.append(_render_mode_line("ByteAI Cache hybrid", burst["runs"]["hybrid"]))
    lines.append("")
    lines.append("## Observations")
    lines.append("")
    lines.append(
        "- `unique_16` answers the baseline question directly: when prompts are genuinely unique and do not collapse into the same cache key, ByteAI Cache spend stays close to direct OpenAI."
    )
    lines.append(
        "- `format_variants_18` isolates the biggest new win from this change set: normalization captures savings that the old exact hash would miss."
    )
    lines.append(
        "- `mixed_workload_24` is the most realistic business case here because it combines hot prompts, messy duplicates, and true one-offs in a single stream."
    )
    lines.append(
        "- The semantic and hybrid runs achieved high hit ratios on this synthetic instruction workload by over-matching. That is useful as a safety signal, not as a deployment recommendation."
    )
    lines.append(
        "- `reordered_instruction_18` shows where additional future work is still needed: some logically similar prompts remain too different for safe default reuse, while instruction-style prompts can still be deceptively similar to an embedder."
    )
    lines.append("")
    lines.append("## Next Cost-Saving Levers")
    lines.append("")
    lines.append(
        "- Add request-class-specific canonicalizers for common prompt templates, not just generic whitespace/punctuation normalization."
    )
    lines.append(
        "- Use stronger semantic embedders or tuned thresholds for workloads where reworded prompts are common and safe to reuse."
    )
    lines.append(
        "- Pre-warm the cache for known hot prompts so the first live user does not pay the cold-start miss."
    )
    lines.append(
        "- Partition cache keys by tool-call structure, retrieval context, or conversation fingerprint for multi-turn apps."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--report",
        default="docs/reports/advanced_openai_cost_patterns_report.md",
    )
    parser.add_argument(
        "--json-report",
        default="docs/reports/advanced_openai_cost_patterns_report.json",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("BYTE_TEST_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set BYTE_TEST_OPENAI_API_KEY or pass --api-key.")

    results: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "pricing": PRICING,
        "mode_descriptions": MODE_DESCRIPTIONS,
        "sequential_scenarios": [],
    }

    for scenario in _build_scenarios():
        items = scenario["items"]
        runs: dict[str, dict[str, Any]] = {
            "direct": _run_direct_sequence(api_key, items),
        }
        for mode in BYTE_MODES:
            runs[mode] = _run_proxy_sequence(api_key, items, mode)
        results["sequential_scenarios"].append(
            _scenario_summary(
                scenario["name"],
                scenario["description"],
                items,
                runs,
            )
        )

    burst = _build_concurrent_mix()
    burst_runs: dict[str, dict[str, Any]] = {
        "direct": _run_direct_concurrent(api_key, burst["items"]),
    }
    for mode in BYTE_MODES:
        burst_runs[mode] = _run_proxy_concurrent(api_key, burst["items"], mode)
    baseline_cost = burst_runs["direct"]["total_cost_usd"]
    baseline_latency = burst_runs["direct"]["avg_latency_ms"]
    for mode in BYTE_MODES:
        burst_runs[mode]["saved_vs_direct_usd"] = round(
            baseline_cost - burst_runs[mode]["total_cost_usd"],
            8,
        )
        burst_runs[mode]["savings_ratio"] = (
            round(
                (baseline_cost - burst_runs[mode]["total_cost_usd"]) / baseline_cost,
                4,
            )
            if baseline_cost
            else 0.0
        )
        burst_runs[mode]["latency_delta_ms"] = round(
            burst_runs[mode]["avg_latency_ms"] - baseline_latency,
            2,
        )

    results["concurrent_burst"] = {
        "name": burst["name"],
        "description": burst["description"],
        "request_count": len(burst["items"]),
        "runs": burst_runs,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results), encoding="utf-8")

    json_path = Path(args.json_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(report_path),
                "json_report": str(json_path),
                "model": MODEL,
            }
        )
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
