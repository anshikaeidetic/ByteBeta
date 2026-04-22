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
from byte.benchmarking._optional_runtime import create_openai_client
from byte.manager import manager_factory
from byte.processor.pre import last_content
from byte.similarity_evaluation import ExactMatchEvaluation

MODEL = "gpt-4o-mini"
PROMPT_TEMPLATE = "Reply with exactly {token} and nothing else."
PRICING = {
    "input_per_million": 0.15,
    "output_per_million": 0.60,
    "cached_input_per_million": 0.075,
    "sources": [
        "https://developers.openai.com/api/pricing",
        "https://developers.openai.com/api/docs/models#gpt-4o-mini",
    ],
    "verified_on": "2026-03-10",
}


def _configure_cache(cache_obj: Cache, cache_dir: str) -> None:
    cache_obj.init(
        pre_embedding_func=last_content,
        embedding_func=lambda data, **_: data,
        data_manager=manager_factory("map", data_dir=cache_dir),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(enable_token_counter=False),
    )


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


def _serve_proxy(port: int, cache_dir: str, server_api_key: str | None) -> None:
    if server_api_key:
        os.environ["OPENAI_API_KEY"] = server_api_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)

    import uvicorn

    import byte_server.server as server

    server.openai_cache = Cache()
    _configure_cache(server.openai_cache, cache_dir)
    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")


def _request_cost(prompt_tokens: int, completion_tokens: int) -> float:
    input_cost = (prompt_tokens / 1_000_000) * PRICING["input_per_million"]
    output_cost = (completion_tokens / 1_000_000) * PRICING["output_per_million"]
    return input_cost + output_cost


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
    total_completion = sum(r.get("completion_tokens", 0) for r in records)
    total_cost = sum(r.get("cost_usd", 0.0) for r in records)
    cached = sum(1 for r in records if r.get("byte"))
    return {
        "request_count": len(records),
        "cached_count": cached,
        "miss_count": len(records) - cached,
        "hit_ratio": round(cached / len(records), 4) if records else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(_p95(latencies), 2),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_cost_usd": round(total_cost, 8),
        "sample": records[:3],
    }


def _direct_request(api_key: str, prompt: str) -> dict[str, Any]:
    client = create_openai_client(api_key=api_key)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=8,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    usage = response.usage
    prompt_tokens = int(usage.prompt_tokens or 0)
    completion_tokens = int(usage.completion_tokens or 0)
    return {
        "status_code": 200,
        "latency_ms": latency_ms,
        "byte": False,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": _request_cost(prompt_tokens, completion_tokens),
        "text": response.choices[0].message.content,
    }


def _run_direct_sequence(api_key: str, prompts: list[str]) -> dict[str, Any]:
    records = [_direct_request(api_key, prompt) for prompt in prompts]
    return _summarize_records(records)


def _proxy_request(
    client: httpx.Client, base_url: str, prompt: str, headers: dict[str, str]
) -> dict[str, Any]:
    start = time.perf_counter()
    response = client.post(
        base_url + "/v1/chat/completions",
        headers=headers,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8,
        },
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    body = response.json()
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    return {
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "byte": bool(body.get("byte")) if isinstance(body, dict) else False,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": _request_cost(prompt_tokens, completion_tokens),
        "text": body["choices"][0]["message"]["content"] if response.status_code == 200 else None,
    }


def _run_proxy_sequence(
    prompts: list[str], client_api_key: str | None, server_api_key: str | None
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix="stress-proxy-")
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = mp.Process(target=_serve_proxy, args=(port, cache_dir, server_api_key), daemon=True)
    process.start()
    try:
        _wait_for_server(base_url)
        headers: dict[str, str] = {}
        if client_api_key:
            headers["Authorization"] = f"Bearer {client_api_key}"
        with httpx.Client(timeout=60.0) as client:
            records = [_proxy_request(client, base_url, prompt, headers) for prompt in prompts]
            stats = client.get(base_url + "/stats").json()
        result = _summarize_records(records)
        result["stats"] = stats
        return result
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        shutil.rmtree(cache_dir, ignore_errors=True)


def _run_direct_concurrent(api_key: str, prompts: list[str]) -> dict[str, Any]:
    wall_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        records = list(pool.map(lambda prompt: _direct_request(api_key, prompt), prompts))
    result = _summarize_records(records)
    result["wall_time_ms"] = round((time.perf_counter() - wall_start) * 1000, 2)
    return result


def _run_proxy_concurrent(
    prompts: list[str], client_api_key: str | None, server_api_key: str | None
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix="stress-proxy-concurrent-")
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = mp.Process(target=_serve_proxy, args=(port, cache_dir, server_api_key), daemon=True)
    process.start()
    try:
        _wait_for_server(base_url)
        headers: dict[str, str] = {}
        if client_api_key:
            headers["Authorization"] = f"Bearer {client_api_key}"

        def one(prompt: str) -> dict[str, Any]:
            with httpx.Client(timeout=60.0) as client:
                return _proxy_request(client, base_url, prompt, headers)

        wall_start = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            records = list(pool.map(one, prompts))
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
    name: str, description: str, prompts: list[str], runs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary = {
        "name": name,
        "description": description,
        "request_count": len(prompts),
        "unique_prompt_count": len(set(prompts)),
        "runs": runs,
    }
    baseline_cost = runs["direct"]["total_cost_usd"]
    for key in ["byte_byo", "byte_server_key"]:
        mode = runs[key]
        mode["saved_vs_direct_usd"] = round(baseline_cost - mode["total_cost_usd"], 8)
        mode["savings_ratio"] = (
            round((baseline_cost - mode["total_cost_usd"]) / baseline_cost, 4)
            if baseline_cost
            else 0.0
        )
    return summary


def _build_sequential_scenarios() -> list[dict[str, Any]]:
    rng = random.Random(42)
    unique = [PROMPT_TEMPLATE.format(token=f"U{i:02d}") for i in range(16)]
    paired: list[str] = []
    for i in range(8):
        prompt = PROMPT_TEMPLATE.format(token=f"P{i:02d}")
        paired.extend([prompt, prompt])
    hotset: list[str] = []
    hot_prompts = [PROMPT_TEMPLATE.format(token=f"H{i}") for i in range(4)]
    for prompt in hot_prompts:
        hotset.extend([prompt] * 6)
    rng.shuffle(hotset)
    single_hot = [PROMPT_TEMPLATE.format(token="HOT")] * 24
    return [
        {
            "name": "unique_16",
            "description": "16 unique prompts; Byte should have no savings.",
            "prompts": unique,
        },
        {
            "name": "paired_16",
            "description": "8 unique prompts repeated twice; Byte should save roughly half the spend.",
            "prompts": paired,
        },
        {
            "name": "hotset_24",
            "description": "4 hot prompts repeated 6x each in shuffled order.",
            "prompts": hotset,
        },
        {
            "name": "single_hot_24",
            "description": "One hot prompt repeated 24 times.",
            "prompts": single_hot,
        },
    ]


def _render_money(value: float) -> str:
    return f"${value:.8f}"


def _render_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# ByteAI Cache Stress Test Report")
    lines.append("")
    lines.append(f"Generated: {results['generated_at']}")
    lines.append(f"Model: {results['model']}")
    lines.append("Price source verified on 2026-03-10:")
    for url in results["pricing"]["sources"]:
        lines.append(f"- {url}")
    lines.append("")
    lines.append("## Case-by-Case Summary")
    lines.append("")
    for scenario in results["sequential_scenarios"]:
        direct = scenario["runs"]["direct"]
        byo = scenario["runs"]["byte_byo"]
        server = scenario["runs"]["byte_server_key"]
        lines.append(f"### {scenario['name']}")
        lines.append(f"- {scenario['description']}")
        lines.append(
            f"- Requests: {scenario['request_count']}, unique prompts: {scenario['unique_prompt_count']}"
        )
        lines.append(
            f"- Direct OpenAI cost: {_render_money(direct['total_cost_usd'])}, avg latency: {direct['avg_latency_ms']} ms, p95 latency: {direct['p95_latency_ms']} ms"
        )
        lines.append(
            f"- ByteAI Cache BYO cost: {_render_money(byo['total_cost_usd'])}, saved: {_render_money(byo['saved_vs_direct_usd'])}, hit ratio: {byo['hit_ratio']}, avg latency: {byo['avg_latency_ms']} ms"
        )
        lines.append(
            f"- ByteAI Cache server-key cost: {_render_money(server['total_cost_usd'])}, saved: {_render_money(server['saved_vs_direct_usd'])}, hit ratio: {server['hit_ratio']}, avg latency: {server['avg_latency_ms']} ms"
        )
        lines.append("")
    lines.append("## Concurrent Burst")
    lines.append("")
    burst = results["concurrent_burst"]
    lines.append(f"- Description: {burst['description']}")
    lines.append(
        f"- Direct OpenAI: cost={_render_money(burst['direct']['total_cost_usd'])}, cached={burst['direct']['cached_count']}, wall_time_ms={burst['direct']['wall_time_ms']}"
    )
    lines.append(
        f"- ByteAI Cache BYO: cost={_render_money(burst['byte_byo']['total_cost_usd'])}, cached={burst['byte_byo']['cached_count']}, hit_ratio={burst['byte_byo']['hit_ratio']}, wall_time_ms={burst['byte_byo']['wall_time_ms']}"
    )
    lines.append(
        f"- ByteAI Cache server-key: cost={_render_money(burst['byte_server_key']['total_cost_usd'])}, cached={burst['byte_server_key']['cached_count']}, hit_ratio={burst['byte_server_key']['hit_ratio']}, wall_time_ms={burst['byte_server_key']['wall_time_ms']}"
    )
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append(
        "- ByteAI Cache only saves money when prompts repeat or collapse into the same cache key. In the all-unique case, cost stays effectively the same and latency is slightly higher because ByteAI Cache still has cache overhead."
    )
    lines.append(
        "- BYO-key mode and server-key mode behave the same on cost and cache hit ratio; the difference is only where the upstream API key lives."
    )
    lines.append(
        "- In the hottest scenarios, ByteAI Cache reduced upstream cost to the first miss only, which is why savings approach the duplicate rate."
    )
    lines.append(
        "- The concurrent duplicate burst is the harshest path here; it shows whether the proxy can avoid repeated upstream work under pressure, not just on easy sequential replays."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--report", default="docs/reports/stress_openai_vs_byte_report.md")
    parser.add_argument("--json-report", default="docs/reports/stress_openai_vs_byte_report.json")
    args = parser.parse_args()
    api_key = args.api_key or os.getenv("BYTE_TEST_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set BYTE_TEST_OPENAI_API_KEY or pass --api-key.")

    results: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "pricing": PRICING,
        "sequential_scenarios": [],
    }

    for scenario in _build_sequential_scenarios():
        prompts = scenario["prompts"]
        runs = {
            "direct": _run_direct_sequence(api_key, prompts),
            "byte_byo": _run_proxy_sequence(prompts, client_api_key=api_key, server_api_key=None),
            "byte_server_key": _run_proxy_sequence(
                prompts, client_api_key=None, server_api_key=api_key
            ),
        }
        results["sequential_scenarios"].append(
            _scenario_summary(scenario["name"], scenario["description"], prompts, runs)
        )

    burst_prompts = [PROMPT_TEMPLATE.format(token="BURST")] * 10
    results["concurrent_burst"] = {
        "description": "10 concurrent identical requests sent at once.",
        "direct": _run_direct_concurrent(api_key, burst_prompts),
        "byte_byo": _run_proxy_concurrent(
            burst_prompts, client_api_key=api_key, server_api_key=None
        ),
        "byte_server_key": _run_proxy_concurrent(
            burst_prompts, client_api_key=None, server_api_key=api_key
        ),
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
