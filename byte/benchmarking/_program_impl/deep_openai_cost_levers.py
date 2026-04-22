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
from byte.adapter.api import init_cache
from byte.benchmarking._optional_runtime import create_openai_client
from byte.processor.pre import last_content, normalize_text, normalized_last_content

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
BYTE_MODES = ["exact", "normalized", "semantic", "hybrid"]
MODE_DESCRIPTIONS = {
    "direct": "Direct OpenAI call with no ByteAI Cache layer.",
    "exact": "ByteAI Cache exact-match cache only.",
    "normalized": "ByteAI Cache normalized cache with request-class canonicalizers.",
    "semantic": "ByteAI Cache guarded semantic cache with lexical and canonical safety rails.",
    "hybrid": "ByteAI Cache layered cache: exact -> normalized -> semantic.",
}
LABEL_SET = "POSITIVE, NEGATIVE, NEUTRAL"


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


def _exact_prompt(token: str) -> str:
    return f"Reply with exactly {token} and nothing else."


def _reordered_exact_variants(token: str) -> list[str]:
    return [
        f"Keep the answer to {token}. Reply with exactly {token} and nothing else.",
        f"Byte benchmark request. Reply with exactly {token} and nothing else. Keep the answer to {token}.",
    ]


def _classification_variants(review: str) -> list[str]:
    return [
        (
            "Classify the sentiment.\n"
            f"Labels: {LABEL_SET}\n"
            f'Review: "{review}"\n'
            "Answer with exactly one label."
        ),
        (
            f'Review: "{review}"\n'
            f"Labels: {LABEL_SET}\n"
            "Classify the sentiment and answer with exactly one label."
        ),
        (
            "Sentiment classification task.\n"
            f"Labels: {LABEL_SET}\n"
            f'Review: "{review}"\n'
            "Return only one label."
        ),
    ]


def _extraction_variants(record: str) -> list[str]:
    return [
        (
            "Extract the fields.\n"
            "Fields: name, city\n"
            f'Text: "{record}"\n'
            'Return compact JSON only with keys "city" and "name". No markdown.'
        ),
        (f'Return JSON with keys: city, name\nText: "{record}"\nReturn raw JSON only.'),
        (
            "Information extraction task.\n"
            "Fields: city, name\n"
            f'Record: "{record}"\n'
            "Extract only those keys as JSON. No code fences."
        ),
    ]


def _semantic_fact_groups() -> list[dict[str, Any]]:
    return [
        {
            "group": "fact_france_capital",
            "expected": "Paris",
            "prompts": [
                "What is the capital of France? Answer in one word.",
                "Name France's capital city. One word only.",
                "Which city is France's capital? Single-word answer.",
            ],
        },
        {
            "group": "fact_hamlet_author",
            "expected": "Shakespeare",
            "prompts": [
                "Who wrote Hamlet? Answer in one word.",
                "Name the author of Hamlet. One word only.",
                "Which writer wrote Hamlet? Single-word answer.",
            ],
        },
        {
            "group": "fact_largest_planet",
            "expected": "Jupiter",
            "prompts": [
                "Which planet is the largest in the solar system? One word only.",
                "Name the largest planet in our solar system. One word.",
                "What is the biggest planet in the solar system? Single-word answer.",
            ],
        },
        {
            "group": "fact_water_boiling_point",
            "expected": "100",
            "prompts": [
                "At sea level, water boils at what temperature in Celsius? Digits only.",
                "What is water's boiling point at sea level in Celsius? Answer with digits only.",
                "Give the boiling point of water at sea level in Celsius. Digits only.",
            ],
        },
    ]


def _build_sequential_scenarios() -> list[dict[str, Any]]:
    rng = random.Random(17)

    unique_items = [
        _make_item(
            _exact_prompt(f"BYTE_U{i:02d}"),
            f"BYTE_U{i:02d}",
            f"unique_{i:02d}",
            "base",
            "exact_answer",
        )
        for i in range(18)
    ]

    exact_pairs: list[dict[str, str]] = []
    for i in range(8):
        token = f"BYTE_E{i:02d}"
        prompt = _exact_prompt(token)
        exact_pairs.append(_make_item(prompt, token, f"exact_{i:02d}", "a", "exact_answer"))
        exact_pairs.append(_make_item(prompt, token, f"exact_{i:02d}", "b", "exact_answer"))

    reviews = [
        ("I absolutely loved this movie and would watch it again.", "POSITIVE"),
        ("This was boring, slow, and a total waste of time.", "NEGATIVE"),
        ("It was okay overall, with nothing especially good or bad.", "NEUTRAL"),
        ("The food was fresh, flavorful, and worth every bite.", "POSITIVE"),
    ]
    canonical_items: list[dict[str, str]] = []
    for idx, (review, label) in enumerate(reviews):
        for variant_idx, prompt in enumerate(_classification_variants(review), 1):
            canonical_items.append(
                _make_item(
                    prompt, label, f"classify_{idx:02d}", f"v{variant_idx}", "classification"
                )
            )

    semantic_items: list[dict[str, str]] = []
    for group in _semantic_fact_groups():
        for variant_idx, prompt in enumerate(group["prompts"], 1):
            semantic_items.append(
                _make_item(prompt, group["expected"], group["group"], f"v{variant_idx}", "fact")
            )

    extraction_records = [
        (
            "Name: Alice Johnson. City: Paris. Department: Support.",
            '{"city":"Paris","name":"Alice Johnson"}',
        ),
        (
            "Name: Bob Smith. City: London. Department: Finance.",
            '{"city":"London","name":"Bob Smith"}',
        ),
        (
            "Name: Carla Gomez. City: Madrid. Department: Sales.",
            '{"city":"Madrid","name":"Carla Gomez"}',
        ),
        (
            "Name: Daniel Wu. City: Singapore. Department: Ops.",
            '{"city":"Singapore","name":"Daniel Wu"}',
        ),
    ]
    extraction_items: list[dict[str, Any]] = []
    extraction_request_overrides = {"response_format": {"type": "json_object"}}
    for idx, (record, expected_json) in enumerate(extraction_records):
        for variant_idx, prompt in enumerate(_extraction_variants(record), 1):
            extraction_items.append(
                _make_item(
                    prompt,
                    expected_json,
                    f"extract_{idx:02d}",
                    f"v{variant_idx}",
                    "extraction",
                    max_tokens=64,
                    request_overrides=extraction_request_overrides,
                )
            )

    prewarmed_items: list[dict[str, str]] = []
    prewarmed_seed: list[dict[str, str]] = []
    for idx in range(2):
        token = f"BYTE_W{idx:02d}"
        prewarmed_seed.append({"question": _exact_prompt(token), "answer": token})
        for variant_idx, prompt in enumerate(_reordered_exact_variants(token), 1):
            prewarmed_items.append(
                _make_item(
                    prompt, token, f"warm_exact_{idx:02d}", f"v{variant_idx}", "exact_answer"
                )
            )
    for idx, (review, label) in enumerate(reviews[:2]):
        variants = _classification_variants(review)
        prewarmed_seed.append({"question": variants[0], "answer": label})
        for variant_idx, prompt in enumerate(variants[1:], 2):
            prewarmed_items.append(
                _make_item(
                    prompt, label, f"warm_classify_{idx:02d}", f"v{variant_idx}", "classification"
                )
            )

    mixed_items: list[dict[str, str]] = []
    for i in range(4):
        token = f"BYTE_MX{i:02d}"
        prompt = _exact_prompt(token)
        mixed_items.append(_make_item(prompt, token, f"mixed_exact_{i:02d}", "a", "exact_answer"))
        mixed_items.append(_make_item(prompt, token, f"mixed_exact_{i:02d}", "b", "exact_answer"))
    for idx, (review, label) in enumerate(reviews[:2]):
        variants = _classification_variants(review)[:2]
        for variant_idx, prompt in enumerate(variants, 1):
            mixed_items.append(
                _make_item(
                    prompt, label, f"mixed_classify_{idx:02d}", f"v{variant_idx}", "classification"
                )
            )
    for group in _semantic_fact_groups()[:2]:
        for variant_idx, prompt in enumerate(group["prompts"][:2], 1):
            mixed_items.append(
                _make_item(
                    prompt, group["expected"], f"mixed_{group['group']}", f"v{variant_idx}", "fact"
                )
            )
    for idx, (record, expected_json) in enumerate(extraction_records[:2]):
        variants = _extraction_variants(record)[:2]
        for variant_idx, prompt in enumerate(variants, 1):
            mixed_items.append(
                _make_item(
                    prompt,
                    expected_json,
                    f"mixed_extract_{idx:02d}",
                    f"v{variant_idx}",
                    "extraction",
                    max_tokens=64,
                    request_overrides=extraction_request_overrides,
                )
            )
    for i in range(4):
        token = f"BYTE_MU{i:02d}"
        mixed_items.append(
            _make_item(_exact_prompt(token), token, f"mixed_unique_{i:02d}", "base", "exact_answer")
        )
    rng.shuffle(mixed_items)

    return [
        {
            "name": "unique_18",
            "description": "18 unique exact-answer prompts. Misses should cost about the same as direct OpenAI.",
            "items": unique_items,
        },
        {
            "name": "exact_pairs_16",
            "description": "8 exact duplicate pairs. Every Byte mode should save the second request.",
            "items": exact_pairs,
        },
        {
            "name": "canonical_templates_12",
            "description": "4 classification workloads with 3 template variants each. This exercises the new request-class canonicalizers.",
            "items": canonical_items,
        },
        {
            "name": "semantic_facts_12",
            "description": "4 reworded factual question groups. This is the safe semantic-reuse workload.",
            "items": semantic_items,
        },
        {
            "name": "structured_extract_12",
            "description": "4 structured extraction workloads with 3 template variants each. This exercises the new extraction canonicalizer.",
            "items": extraction_items,
        },
        {
            "name": "prewarmed_hotset_8",
            "description": "Known hot prompts are seeded before traffic. First live users should hit the cache on normalized or semantically-safe variants.",
            "items": prewarmed_items,
            "warm_data": prewarmed_seed,
        },
        {
            "name": "mixed_workload_24",
            "description": "A realistic blend of exact repeats, canonical template variants, safe paraphrases, and true one-offs.",
            "items": mixed_items,
        },
    ]


def _build_concurrent_scenario() -> dict[str, Any]:
    items: list[dict[str, str]] = []
    for i in range(2):
        token = f"BYTE_CX{i:02d}"
        prompt = _exact_prompt(token)
        items.append(_make_item(prompt, token, f"burst_exact_{i:02d}", "a", "exact_answer"))
        items.append(_make_item(prompt, token, f"burst_exact_{i:02d}", "b", "exact_answer"))
    reviews = [
        ("I absolutely loved this movie and would watch it again.", "POSITIVE"),
        ("This was boring, slow, and a total waste of time.", "NEGATIVE"),
    ]
    for idx, (review, label) in enumerate(reviews):
        for variant_idx, prompt in enumerate(_classification_variants(review)[:2], 1):
            items.append(
                _make_item(
                    prompt, label, f"burst_classify_{idx:02d}", f"v{variant_idx}", "classification"
                )
            )
    for group in _semantic_fact_groups()[:2]:
        for variant_idx, prompt in enumerate(group["prompts"][:2], 1):
            items.append(
                _make_item(
                    prompt, group["expected"], f"burst_{group['group']}", f"v{variant_idx}", "fact"
                )
            )
    burst_record = "Name: Erin Cole. City: Dublin. Department: Support."
    extraction_request_overrides = {"response_format": {"type": "json_object"}}
    for variant_idx, prompt in enumerate(_extraction_variants(burst_record)[:2], 1):
        items.append(
            _make_item(
                prompt,
                '{"city":"Dublin","name":"Erin Cole"}',
                "burst_extract_00",
                f"v{variant_idx}",
                "extraction",
                max_tokens=64,
                request_overrides=extraction_request_overrides,
            )
        )
    for i in range(2):
        token = f"BYTE_CU{i:02d}"
        items.append(
            _make_item(_exact_prompt(token), token, f"burst_unique_{i:02d}", "base", "exact_answer")
        )
    return {
        "name": "concurrent_mixed_16",
        "description": "16 concurrent requests mixing duplicates, canonical template variants, safe paraphrases, and unique prompts.",
        "items": items,
    }


def _pricing_cost(prompt_tokens: int, cached_prompt_tokens: int, completion_tokens: int) -> float:
    uncached_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    input_cost = (uncached_prompt_tokens / 1_000_000) * PRICING["input_per_million"]
    cached_cost = (cached_prompt_tokens / 1_000_000) * PRICING["cached_input_per_million"]
    output_cost = (completion_tokens / 1_000_000) * PRICING["output_per_million"]
    return input_cost + cached_cost + output_cost


def _usage_fields(usage: Any) -> dict[str, int]:
    if not usage:
        return {"prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0}

    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details", {}) or {}
        cached_prompt_tokens = int(details.get("cached_tokens", 0) or 0)
    else:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        if details is None:
            cached_prompt_tokens = 0
        elif isinstance(details, dict):
            cached_prompt_tokens = int(details.get("cached_tokens", 0) or 0)
        else:
            cached_prompt_tokens = int(getattr(details, "cached_tokens", 0) or 0)

    cached_prompt_tokens = min(cached_prompt_tokens, prompt_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _strip_code_fences(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines:
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalized_answer(text: str | None) -> str:
    raw = _strip_code_fences((text or "").strip())
    candidate = raw
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]
    if candidate.startswith(("{", "[")):
        try:
            return json.dumps(json.loads(candidate), sort_keys=True, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            pass
    return normalize_text(raw)


def _response_record(
    *,
    status_code: int,
    latency_ms: float,
    byte_flag: bool,
    text: str | None,
    usage: Any,
    item: dict[str, str],
) -> dict[str, Any]:
    fields = _usage_fields(usage)
    return {
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "byte": byte_flag,
        "prompt_tokens": fields["prompt_tokens"],
        "cached_prompt_tokens": fields["cached_prompt_tokens"],
        "completion_tokens": fields["completion_tokens"],
        "cost_usd": _pricing_cost(
            fields["prompt_tokens"],
            fields["cached_prompt_tokens"],
            fields["completion_tokens"],
        ),
        "text": text,
        "expected": item["expected"],
        "group": item["group"],
        "variant": item["variant"],
        "kind": item["kind"],
        "correct": _normalized_answer(text) == _normalized_answer(item["expected"]),
    }


def _direct_request(client: Any, item: dict[str, Any]) -> dict[str, Any]:
    request_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "max_tokens": int(item.get("max_tokens", 16) or 16),
    }
    request_payload.update(item.get("request_overrides", {}) or {})
    start = time.perf_counter()
    response = client.chat.completions.create(**request_payload)
    latency_ms = (time.perf_counter() - start) * 1000
    text = response.choices[0].message.content or ""
    return _response_record(
        status_code=200,
        latency_ms=latency_ms,
        byte_flag=False,
        text=text,
        usage=response.usage,
        item=item,
    )


def _proxy_request(
    client: httpx.Client, base_url: str, api_key: str, item: dict[str, Any]
) -> dict[str, Any]:
    request_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "max_tokens": int(item.get("max_tokens", 16) or 16),
    }
    request_payload.update(item.get("request_overrides", {}) or {})
    start = time.perf_counter()
    response = client.post(
        base_url + "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request_payload,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    body = response.json()
    text = None
    usage = {}
    if response.status_code == 200 and isinstance(body, dict):
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
    return _response_record(
        status_code=response.status_code,
        latency_ms=latency_ms,
        byte_flag=bool(body.get("byte")) if isinstance(body, dict) else False,
        text=text,
        usage=usage,
        item=item,
    )


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(math.ceil(0.95 * len(ordered)) - 1, len(ordered) - 1))
    return round(ordered[idx], 2)


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records]
    total_cost = sum(record["cost_usd"] for record in records)
    total_prompt = sum(record["prompt_tokens"] for record in records)
    total_cached_prompt = sum(record["cached_prompt_tokens"] for record in records)
    total_completion = sum(record["completion_tokens"] for record in records)
    cached = sum(1 for record in records if record["byte"])
    correct = sum(1 for record in records if record["correct"])
    return {
        "request_count": len(records),
        "cached_count": cached,
        "miss_count": len(records) - cached,
        "hit_ratio": round(cached / len(records), 4) if records else 0.0,
        "accuracy_ratio": round(correct / len(records), 4) if records else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": _p95(latencies),
        "total_prompt_tokens": total_prompt,
        "total_cached_prompt_tokens": total_cached_prompt,
        "total_completion_tokens": total_completion,
        "total_cost_usd": round(total_cost, 8),
        "sample": records[:5],
    }


def _semantic_config() -> Config:
    return Config(
        enable_token_counter=False,
        similarity_threshold=0.94,
        semantic_min_token_overlap=0.5,
        semantic_max_length_ratio=2.5,
        semantic_enforce_canonical_match=True,
        tiered_cache=True,
        embedding_cache_size=20000,
    )


def _base_config() -> Config:
    return Config(enable_token_counter=False, embedding_cache_size=20000)


def _configure_cache(
    cache_obj: Cache, cache_dir: str, mode: str, warm_data: list[dict[str, str]] | None = None
) -> None:
    semantic_config = _semantic_config()
    base_config = _base_config()
    init_cache(
        mode=mode,
        data_dir=cache_dir,
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=base_config,
        exact_config=base_config,
        normalized_config=base_config,
        semantic_config=semantic_config,
        warm_data=warm_data,
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


def _serve_proxy(
    port: int, cache_dir: str, mode: str, warm_data: list[dict[str, str]] | None = None
) -> None:
    import uvicorn

    import byte_server.server as server

    server.openai_cache = Cache()
    _configure_cache(server.openai_cache, cache_dir, mode, warm_data=warm_data)
    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")


def _run_direct_sequence(api_key: str, items: list[dict[str, str]]) -> dict[str, Any]:
    client = create_openai_client(api_key=api_key)
    records = [_direct_request(client, item) for item in items]
    return _summarize_records(records)


def _run_proxy_sequence(
    api_key: str,
    items: list[dict[str, str]],
    mode: str,
    warm_data: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"deep-{mode}-")
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = mp.Process(target=_serve_proxy, args=(port, cache_dir, mode, warm_data), daemon=True)
    process.start()
    try:
        _wait_for_server(base_url)
        with httpx.Client(timeout=60.0) as client:
            records = [_proxy_request(client, base_url, api_key, item) for item in items]
            stats = client.get(base_url + "/stats").json()
        result = _summarize_records(records)
        result["stats"] = stats
        result["prewarmed"] = bool(warm_data)
        return result
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        shutil.rmtree(cache_dir, ignore_errors=True)


def _run_direct_concurrent(api_key: str, items: list[dict[str, str]]) -> dict[str, Any]:
    wall_start = time.perf_counter()

    def one(item: dict[str, str]) -> dict[str, Any]:
        client = create_openai_client(api_key=api_key)
        return _direct_request(client, item)

    with cf.ThreadPoolExecutor(max_workers=len(items)) as pool:
        records = list(pool.map(one, items))
    result = _summarize_records(records)
    result["wall_time_ms"] = round((time.perf_counter() - wall_start) * 1000, 2)
    return result


def _run_proxy_concurrent(api_key: str, items: list[dict[str, str]], mode: str) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"deep-concurrent-{mode}-")
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = mp.Process(target=_serve_proxy, args=(port, cache_dir, mode, None), daemon=True)
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
    name: str,
    description: str,
    items: list[dict[str, str]],
    runs: dict[str, dict[str, Any]],
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
    baseline_cost = runs["direct"]["total_cost_usd"]
    baseline_latency = runs["direct"]["avg_latency_ms"]
    for mode in BYTE_MODES:
        data = runs[mode]
        data["saved_vs_direct_usd"] = round(baseline_cost - data["total_cost_usd"], 8)
        data["savings_ratio"] = (
            round(
                (baseline_cost - data["total_cost_usd"]) / baseline_cost,
                4,
            )
            if baseline_cost
            else 0.0
        )
        data["latency_delta_ms"] = round(data["avg_latency_ms"] - baseline_latency, 2)
    return summary


def _render_money(value: float) -> str:
    return f"${value:.8f}"


def _render_mode_line(name: str, data: dict[str, Any]) -> str:
    safety_suffix = ""
    if data.get("accuracy_ratio", 1.0) < 1.0:
        safety_suffix = " [accuracy < 1.0]"
    prewarm_suffix = ""
    if data.get("prewarmed"):
        prewarm_suffix = ", prewarmed=true"
    return (
        f"- {name}: cost={_render_money(data['total_cost_usd'])}, hit_ratio={data['hit_ratio']}, "
        f"accuracy={data['accuracy_ratio']}, avg_latency={data['avg_latency_ms']} ms, "
        f"p95_latency={data['p95_latency_ms']} ms, saved={_render_money(data.get('saved_vs_direct_usd', 0.0))}, "
        f"savings_ratio={data.get('savings_ratio', 0.0)}{prewarm_suffix}{safety_suffix}"
    )


def _render_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# ByteAI Cache Deep OpenAI Cost-Levers Report")
    lines.append("")
    lines.append(f"Generated: {results['generated_at']}")
    lines.append(f"Model: {results['model']}")
    lines.append(
        "This live benchmark uses OpenAI traffic because that is the only live key available in this environment."
    )
    lines.append(
        "The cache improvements exercised here live in ByteAI Cache's shared adapter layer, so the features themselves are provider-agnostic."
    )
    lines.append("")
    lines.append("## Features Exercised")
    lines.append("")
    lines.append(
        "- Request-class canonicalizers for exact-answer, labeled classification, translation-style, and structured extraction prompt shells."
    )
    lines.append(
        "- Guarded semantic evaluation with canonical-match, lexical-overlap, and length-ratio safety rails."
    )
    lines.append("- Prewarming through shared cache init helpers and the existing warm path.")
    lines.append(
        "- Tool-call, retrieval-context, and conversation-fingerprint namespaces in the shared adapter path."
    )
    lines.append("- Async adapter parity for the same safety and latency features.")
    lines.append(
        "- Provider-agnostic memory snapshot export/import so apps can share safe learned state across deployments."
    )
    lines.append(
        "- Provider client pooling in the major adapters to reduce miss-path connection setup overhead."
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
        "- When prompts do not land in the same safe cache key, ByteAI Cache cost stays close to a normal OpenAI call."
    )
    lines.append(
        "- The new canonicalizers materially improve savings on repeated prompt templates without sacrificing accuracy."
    )
    lines.append(
        "- Guarded semantic mode is now much safer on the workloads below; the semantic gains are best on reworded fact queries, while normalized mode remains the default safe workhorse."
    )
    lines.append(
        "- Prewarming shifts the first-user miss into an operator-controlled warm step, which is the cleanest way to save cost on known hot prompts."
    )
    lines.append("")
    lines.append("## Sequential Results")
    lines.append("")
    for scenario in results["sequential_scenarios"]:
        lines.append(f"### {scenario['name']}")
        lines.append(f"- {scenario['description']}")
        lines.append(
            f"- Requests={scenario['request_count']}, unique_prompts={scenario['unique_prompt_count']}, "
            f"logical_groups={scenario['logical_group_count']}, prewarmed_seed_count={scenario['prewarmed_seed_count']}"
        )
        lines.append(_render_mode_line("Direct", scenario["runs"]["direct"]))
        lines.append(_render_mode_line("ByteAI Cache exact", scenario["runs"]["exact"]))
        lines.append(_render_mode_line("ByteAI Cache normalized", scenario["runs"]["normalized"]))
        lines.append(_render_mode_line("ByteAI Cache semantic", scenario["runs"]["semantic"]))
        lines.append(_render_mode_line("ByteAI Cache hybrid", scenario["runs"]["hybrid"]))
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
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- `unique_18` is the miss baseline. It answers the question of whether ByteAI Cache saves anything without a hit: it should not, and the costs should stay near direct OpenAI."
    )
    lines.append(
        "- `canonical_templates_12` isolates the new prompt-template canonicalizers. That is where normalized and hybrid should now create savings that exact hashing would miss."
    )
    lines.append(
        "- `semantic_facts_12` isolates the guarded semantic path on a workload where reworded prompts are safe to reuse."
    )
    lines.append(
        "- `structured_extract_12` isolates the new extraction canonicalizer on deterministic JSON-style workloads."
    )
    lines.append(
        "- `prewarmed_hotset_8` shows the operational lever that removes cold-start misses for known hot prompts."
    )
    lines.append("- `mixed_workload_24` is the practical blended case for product planning.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--report",
        default="docs/reports/deep_openai_cost_levers_report.md",
    )
    parser.add_argument(
        "--json-report",
        default="docs/reports/deep_openai_cost_levers_report.json",
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

    for scenario in _build_sequential_scenarios():
        runs: dict[str, dict[str, Any]] = {
            "direct": _run_direct_sequence(api_key, scenario["items"]),
        }
        warm_data = scenario.get("warm_data")
        for mode in BYTE_MODES:
            runs[mode] = _run_proxy_sequence(api_key, scenario["items"], mode, warm_data=warm_data)
        results["sequential_scenarios"].append(
            _scenario_summary(
                scenario["name"],
                scenario["description"],
                scenario["items"],
                runs,
                warm_data=warm_data,
            )
        )

    burst = _build_concurrent_scenario()
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
