import argparse
import copy
import json
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from byte import Cache  # pylint: disable=wrong-import-position
from byte._backends import openai as byte_openai  # pylint: disable=wrong-import-position
from byte.benchmarking._program_common import call_with_retry
from byte.benchmarking.programs import deep_openai_coding_benchmark as coding
from byte.benchmarking.programs import deep_openai_comprehensive_workload_benchmark as comprehensive
from byte.benchmarking.programs import deep_openai_surface_benchmark as surface
from byte.processor.shared_memory import (
    clear_shared_memory,  # pylint: disable=wrong-import-position
)

REPORT_DIR = REPO_ROOT / "docs" / "reports"
DEFAULT_REPORT = REPORT_DIR / "openai_prompt_stress_benchmark.md"
DEFAULT_JSON_REPORT = REPORT_DIR / "openai_prompt_stress_benchmark.json"
CHAT_MODEL = coding.CHEAP_MODEL
BYTE_MODES = ["exact", "normalized", "hybrid"]
CONCURRENT_BURST_SIZE = 20


def _clone_item(
    item: dict[str, Any], *, bucket: str, variant: str, group: str = ""
) -> dict[str, Any]:
    cloned = copy.deepcopy(item)
    cloned["bucket"] = bucket
    cloned["request_style"] = "standard"
    cloned["variant"] = variant
    if group:
        cloned["group"] = group
    return cloned


def _bucket_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("bucket") or "unknown") for item in items)
    return dict(sorted(counts.items()))


def _build_exact_repeat_bucket() -> list[dict[str, Any]]:
    support_items = surface._build_normal_chat_scenario()["items"]  # pylint: disable=protected-access
    document_items = surface._build_document_scenario()["items"]  # pylint: disable=protected-access
    coding_scenarios = coding._build_sequential_scenarios()  # pylint: disable=protected-access
    seeds = [
        support_items[0],
        support_items[2],
        document_items[2],
        coding_scenarios[0]["items"][0],
        coding_scenarios[1]["items"][0],
    ]
    items: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds, 1):
        for repetition in range(4):
            items.append(
                _clone_item(
                    seed,
                    bucket="exact_repeat",
                    group=f"stress_exact_{seed_index:02d}",
                    variant=f"r{repetition + 1}",
                )
            )
    return items


def _build_normalized_variant_bucket() -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in coding._build_sequential_scenarios():  # pylint: disable=protected-access
        if scenario["name"] == "cursor_prewarmed_hotset_4":
            continue
        for item in scenario["items"]:
            grouped[item["group"]].append(item)
    for scenario_builder in (surface._build_normal_chat_scenario, surface._build_document_scenario):  # pylint: disable=protected-access
        for item in scenario_builder()["items"]:
            grouped[item["group"]].append(item)

    preferred_groups = [
        "support_billing",
        "support_technical",
        "support_shipping",
        "general_qa_paris",
        "doc_summary_incident",
        "doc_extract_invoice",
        "doc_classify_clause",
        "doc_draft_release_note",
        "cursor_bug_mutable_default",
        "cursor_explain_linear",
    ]
    selected_groups = []
    for group_name in preferred_groups:
        variants = grouped.get(group_name) or []
        variant_ids = {item["variant"] for item in variants}
        if len(variants) >= 2 and len(variant_ids) >= 2:
            selected_groups.append(group_name)
    if len(selected_groups) < 10:
        for group_name in sorted(grouped):
            if group_name in selected_groups:
                continue
            variants = grouped[group_name]
            variant_ids = {item["variant"] for item in variants}
            if len(variants) >= 2 and len(variant_ids) >= 2:
                selected_groups.append(group_name)
            if len(selected_groups) >= 10:
                break

    items: list[dict[str, Any]] = []
    for group_index, group_name in enumerate(selected_groups, 1):
        variants = sorted(grouped[group_name], key=lambda item: str(item["variant"]))[:2]
        for variant_index, item in enumerate(variants, 1):
            items.append(
                _clone_item(
                    item,
                    bucket="normalized_variant",
                    group=f"stress_norm_{group_index:02d}",
                    variant=f"v{variant_index}",
                )
            )
    return items


def _build_coding_mixed_bucket() -> list[dict[str, Any]]:
    coding_scenarios = coding._build_sequential_scenarios()  # pylint: disable=protected-access
    mixed = []
    for scenario in coding_scenarios:
        if scenario["name"] == "cursor_mixed_workload_16":
            mixed.extend(copy.deepcopy(scenario["items"]))
    routing_items = copy.deepcopy(coding._build_routing_scenario()["items"])  # pylint: disable=protected-access
    selected = mixed + routing_items[:4]
    items: list[dict[str, Any]] = []
    for index, item in enumerate(selected[:20], 1):
        items.append(
            _clone_item(
                item,
                bucket="coding_mixed",
                group=f"stress_coding_{index:02d}",
                variant=f"s{index:02d}",
            )
        )
    return items


def _build_contextual_item(
    session_id: str,
    token: str,
    group: str,
    short_prompt: str,
    *,
    variant: str,
) -> dict[str, Any]:
    repo_summary = comprehensive._repo_summary_payload()  # pylint: disable=protected-access
    changed_files = comprehensive._changed_files_payload()  # pylint: disable=protected-access
    changed_hunks = comprehensive._changed_hunks_payload()  # pylint: disable=protected-access
    retrieval_context = comprehensive._retrieval_context_payload()  # pylint: disable=protected-access
    document_context = comprehensive._document_context_payload()  # pylint: disable=protected-access
    support_articles = comprehensive._support_articles_payload()  # pylint: disable=protected-access
    full_prompt = (
        f"{short_prompt}\n\n"
        f"Repo summary:\n{json.dumps(repo_summary, indent=2)}\n\n"
        f"Changed files:\n{json.dumps(changed_files, indent=2)}\n\n"
        f"Changed hunks:\n{json.dumps(changed_hunks, indent=2)}\n\n"
        f"Retrieval context:\n{json.dumps(retrieval_context, indent=2)}\n\n"
        f"Document context:\n{json.dumps(document_context, indent=2)}\n\n"
        f"Support articles:\n{json.dumps(support_articles, indent=2)}"
    )
    return {
        "expected": token,
        "group": group,
        "variant": variant,
        "kind": "instruction",
        "bucket": "shared_context_unique",
        "request_style": "contextual",
        "max_tokens": 8,
        "direct_messages": [
            {"role": "system", "content": comprehensive._COMMON_SYSTEM},  # pylint: disable=protected-access
            {"role": "user", "content": full_prompt},
        ],
        "byte_messages": [
            {"role": "system", "content": comprehensive._COMMON_SYSTEM},  # pylint: disable=protected-access
            {"role": "user", "content": short_prompt},
        ],
        "byte_context": {
            "byte_repo_summary": repo_summary,
            "byte_changed_files": changed_files,
            "byte_changed_hunks": changed_hunks,
            "byte_retrieval_context": retrieval_context,
            "byte_document_context": document_context,
            "byte_support_articles": support_articles,
            "byte_session_id": session_id,
        },
    }


def _build_shared_context_unique_bucket() -> list[dict[str, Any]]:
    prompt_specs = [
        (
            "support",
            "Review the shared support and repo context, then reply exactly {token} and nothing else.",
        ),
        (
            "document",
            "Review the shared document and repo context, then reply exactly {token} and nothing else.",
        ),
        (
            "coding",
            "Review the shared code and retrieval context, then reply exactly {token} and nothing else.",
        ),
        (
            "summary",
            "Review the shared workspace context, then reply exactly {token} and nothing else.",
        ),
    ]
    items: list[dict[str, Any]] = []
    for session_index in range(5):
        session_id = f"stress-session-{session_index + 1:02d}"
        for prompt_index, (group_stub, prompt_template) in enumerate(prompt_specs, 1):
            token = f"CTX_{group_stub.upper()}_{session_index + 1:02d}"
            items.append(
                _build_contextual_item(
                    session_id,
                    token,
                    group=f"stress_context_{group_stub}_{session_index + 1:02d}",
                    short_prompt=prompt_template.format(token=token),
                    variant=f"v{prompt_index}",
                )
            )
    return items


def _build_plain_unique_bucket() -> list[dict[str, Any]]:
    prompts = [
        "Single-token benchmark request {index:02d}. Topic: duplicate billing in region {index:02d}. Reply exactly STRESS_UNIQUE_{index:02d} and nothing else.",
        "Single-token benchmark request {index:02d}. Topic: isolated export incident {index:02d}. Reply exactly STRESS_UNIQUE_{index:02d} and nothing else.",
        "Single-token benchmark request {index:02d}. Topic: release note draft {index:02d}. Reply exactly STRESS_UNIQUE_{index:02d} and nothing else.",
        "Single-token benchmark request {index:02d}. Topic: logistics escalation {index:02d}. Reply exactly STRESS_UNIQUE_{index:02d} and nothing else.",
        "Single-token benchmark request {index:02d}. Topic: code review follow-up {index:02d}. Reply exactly STRESS_UNIQUE_{index:02d} and nothing else.",
    ]
    items: list[dict[str, Any]] = []
    for index in range(20):
        token = f"STRESS_UNIQUE_{index + 1:02d}"
        template = prompts[index % len(prompts)]
        item = coding._make_item(  # pylint: disable=protected-access
            template.format(index=index + 1),
            token,
            f"stress_unique_{index + 1:02d}",
            "base",
            "instruction",
            max_tokens=6,
        )
        items.append(
            _clone_item(
                item,
                bucket="plain_unique",
                group=f"stress_unique_{index + 1:02d}",
                variant="base",
            )
        )
    return items


def _build_stress_items() -> list[dict[str, Any]]:
    items = []
    items.extend(_build_exact_repeat_bucket())
    items.extend(_build_normalized_variant_bucket())
    items.extend(_build_coding_mixed_bucket())
    items.extend(_build_shared_context_unique_bucket())
    items.extend(_build_plain_unique_bucket())
    if len(items) != 100:
        raise ValueError(f"Expected 100 stress items, found {len(items)}")
    return items


def _burst_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority_buckets = {"exact_repeat", "normalized_variant", "shared_context_unique"}
    burst = [copy.deepcopy(item) for item in items if item["bucket"] in priority_buckets][
        :CONCURRENT_BURST_SIZE
    ]
    for index, item in enumerate(burst, 1):
        item["variant"] = f"burst{index:02d}"
    return burst


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def _enrich_record(
    record: dict[str, Any], item: dict[str, Any], *, byte_reason: str = ""
) -> dict[str, Any]:
    enriched = dict(record)
    enriched["bucket"] = item.get("bucket", "")
    enriched["request_style"] = item.get("request_style", "standard")
    reason = str(byte_reason or "").strip()
    if not reason:
        reason = "cache_hit" if record.get("byte") else "miss"
    enriched["byte_reason"] = reason
    return enriched


def _direct_request(api_key: str, item: dict[str, Any]) -> dict[str, Any]:
    if item.get("request_style") == "contextual":
        record = comprehensive._direct_context_request(api_key, item)  # pylint: disable=protected-access
    else:
        record = coding._direct_request(api_key, item, CHAT_MODEL)  # pylint: disable=protected-access
    return _enrich_record(record, item)


def _extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return str(response.get("text") or "")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for entry in content:
            if isinstance(entry, dict):
                parts.append(str(entry.get("text") or entry.get("content") or ""))
            else:
                parts.append(str(entry))
        return "".join(parts).strip()
    return str(content or "")


def _byte_request(
    api_key: str, cache_obj: Cache, item: dict[str, Any], *, scenario_name: str
) -> dict[str, Any]:
    if item.get("request_style") == "contextual":
        payload = {
            "model": CHAT_MODEL,
            "messages": item["byte_messages"],
            "temperature": 0,
            "max_tokens": int(item.get("max_tokens", 8) or 8),
            "api_key": api_key,
            "cache_obj": cache_obj,
            **dict(item.get("byte_context") or {}),
            "byte_memory": {
                "provider": "openai",
                "metadata": {
                    "scenario": scenario_name,
                    "group": item["group"],
                    "variant": item["variant"],
                    "kind": item["kind"],
                    "bucket": item["bucket"],
                },
            },
        }
    else:
        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": 0,
            "max_tokens": int(item.get("max_tokens", 12) or 12),
            "api_key": api_key,
            "cache_obj": cache_obj,
            "byte_memory": {
                "provider": "openai",
                "metadata": {
                    "scenario": scenario_name,
                    "group": item["group"],
                    "variant": item["variant"],
                    "kind": item["kind"],
                    "bucket": item["bucket"],
                },
            },
        }
    start = time.perf_counter()
    try:
        response = call_with_retry(lambda: byte_openai.ChatCompletion.create(**payload))
        latency_ms = (time.perf_counter() - start) * 1000
        record = coding._response_record(  # pylint: disable=protected-access
            status_code=200,
            latency_ms=latency_ms,
            byte_flag=bool(response.get("byte")),
            model_name=str(response.get("model") or CHAT_MODEL),
            route_info=response.get("byte_router"),
            text=_extract_text(response),
            usage=response.get("usage"),
            item=item,
        )
        return _enrich_record(record, item, byte_reason=str(response.get("byte_reason") or "miss"))
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.perf_counter() - start) * 1000
        record = coding._response_record(  # pylint: disable=protected-access
            status_code=599,
            latency_ms=latency_ms,
            byte_flag=False,
            model_name=CHAT_MODEL,
            route_info=None,
            text=None,
            usage=None,
            item=item,
            error=str(exc),
        )
        return _enrich_record(record, item, byte_reason="error")


def _breakdown(
    records: list[dict[str, Any]], field: str, *, limit: int = 0
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(field) or "unknown")].append(record)
    ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    if limit > 0:
        ordered = ordered[:limit]
    breakdown: dict[str, dict[str, Any]] = {}
    for name, group_records in ordered:
        latencies = [float(record.get("latency_ms", 0.0) or 0.0) for record in group_records]
        breakdown[name] = {
            "request_count": len(group_records),
            "cached_count": sum(1 for record in group_records if record.get("byte")),
            "hit_ratio": round(
                sum(1 for record in group_records if record.get("byte")) / len(group_records), 4
            ),
            "accuracy_ratio": round(
                sum(1 for record in group_records if record.get("correct")) / len(group_records), 4
            ),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": _percentile(latencies, 0.95),
            "total_prompt_tokens": sum(
                int(record.get("prompt_tokens", 0) or 0) for record in group_records
            ),
            "total_cost_usd": round(
                sum(float(record.get("cost_usd", 0.0) or 0.0) for record in group_records), 8
            ),
        }
    return breakdown


def _summarize_records(
    records: list[dict[str, Any]], *, wall_time_ms: float = 0.0
) -> dict[str, Any]:
    base = coding._summarize_records(records)  # pylint: disable=protected-access
    latencies = [float(record.get("latency_ms", 0.0) or 0.0) for record in records]
    total_prompt_tokens = int(base.get("total_prompt_tokens", 0) or 0)
    total_completion_tokens = int(base.get("total_completion_tokens", 0) or 0)
    elapsed_s = (
        (wall_time_ms / 1000.0)
        if wall_time_ms > 0
        else (sum(latencies) / 1000.0 if latencies else 0.0)
    )
    base.update(
        {
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p90_latency_ms": _percentile(latencies, 0.90),
            "p99_latency_ms": _percentile(latencies, 0.99),
            "min_latency_ms": round(min(latencies), 2) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
            "stdev_latency_ms": round(statistics.pstdev(latencies), 2)
            if len(latencies) > 1
            else 0.0,
            "avg_prompt_tokens_per_request": round(total_prompt_tokens / len(records), 2)
            if records
            else 0.0,
            "avg_completion_tokens_per_request": round(total_completion_tokens / len(records), 2)
            if records
            else 0.0,
            "avg_cost_usd": round(float(base.get("total_cost_usd", 0.0) or 0.0) / len(records), 8)
            if records
            else 0.0,
            "cost_per_1000_requests_usd": round(
                (float(base.get("total_cost_usd", 0.0) or 0.0) / len(records)) * 1000, 4
            )
            if records
            else 0.0,
            "throughput_rps": round(len(records) / elapsed_s, 2) if elapsed_s > 0 else 0.0,
            "wall_time_ms": round(wall_time_ms, 2)
            if wall_time_ms > 0
            else round(sum(latencies), 2),
            "bucket_breakdown": _breakdown(records, "bucket"),
            "kind_breakdown": _breakdown(records, "kind"),
            "top_group_breakdown": _breakdown(records, "group", limit=12),
            "byte_reason_counts": dict(
                sorted(
                    Counter(
                        str(record.get("byte_reason") or "")
                        for record in records
                        if str(record.get("byte_reason") or "").strip()
                    ).items()
                )
            ),
            "sample": records[:8],
        }
    )
    return base


def _apply_baseline(summary: dict[str, Any], baseline: dict[str, Any]) -> None:
    baseline_cost = float(baseline.get("total_cost_usd", 0.0) or 0.0)
    baseline_latency = float(baseline.get("avg_latency_ms", 0.0) or 0.0)
    baseline_prompt_tokens = int(baseline.get("total_prompt_tokens", 0) or 0)
    baseline_completion_tokens = int(baseline.get("total_completion_tokens", 0) or 0)
    summary["saved_vs_direct_usd"] = round(
        baseline_cost - float(summary.get("total_cost_usd", 0.0) or 0.0), 8
    )
    summary["savings_ratio"] = (
        round((baseline_cost - float(summary.get("total_cost_usd", 0.0) or 0.0)) / baseline_cost, 4)
        if baseline_cost
        else 0.0
    )
    summary["latency_delta_ms"] = round(
        float(summary.get("avg_latency_ms", 0.0) or 0.0) - baseline_latency, 2
    )
    summary["prompt_token_delta"] = (
        int(summary.get("total_prompt_tokens", 0) or 0) - baseline_prompt_tokens
    )
    summary["prompt_token_reduction_ratio"] = (
        round(
            (baseline_prompt_tokens - int(summary.get("total_prompt_tokens", 0) or 0))
            / baseline_prompt_tokens,
            4,
        )
        if baseline_prompt_tokens
        else 0.0
    )
    summary["completion_token_delta"] = (
        int(summary.get("total_completion_tokens", 0) or 0) - baseline_completion_tokens
    )
    summary["completion_token_reduction_ratio"] = (
        round(
            (baseline_completion_tokens - int(summary.get("total_completion_tokens", 0) or 0))
            / baseline_completion_tokens,
            4,
        )
        if baseline_completion_tokens
        else 0.0
    )


def _run_direct_sequence(
    api_key: str, items: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wall_start = time.perf_counter()
    records = [_direct_request(api_key, item) for item in items]
    wall_time_ms = (time.perf_counter() - wall_start) * 1000
    return records, _summarize_records(records, wall_time_ms=wall_time_ms)


def _run_byte_sequence(
    api_key: str, items: list[dict[str, Any]], *, mode: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_dir = tempfile.mkdtemp(prefix=f"stress-{mode}-")
    scope = f"stress::{mode}::{int(time.time() * 1000)}"
    clear_shared_memory(scope)
    cache_obj = Cache()
    try:
        comprehensive._configure_cache(cache_obj, cache_dir, mode, scope=scope)  # pylint: disable=protected-access
        wall_start = time.perf_counter()
        records = [
            _byte_request(api_key, cache_obj, item, scenario_name=f"stress_prompt_{mode}")
            for item in items
        ]
        wall_time_ms = (time.perf_counter() - wall_start) * 1000
        summary = _summarize_records(records, wall_time_ms=wall_time_ms)
        summary["memory"] = {
            "summary": cache_obj.memory_summary(),
            "recent_interactions": cache_obj.recent_interactions(limit=6),
        }
        return records, summary
    finally:
        comprehensive._release_cache(cache_obj, scope, cache_dir)  # pylint: disable=protected-access


def _run_direct_concurrent(api_key: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        records = list(pool.map(lambda item: _direct_request(api_key, item), items))
    wall_time_ms = (time.perf_counter() - wall_start) * 1000
    return _summarize_records(records, wall_time_ms=wall_time_ms)


def _run_byte_concurrent(api_key: str, items: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"stress-burst-{mode}-")
    scope = f"stress::burst::{mode}::{int(time.time() * 1000)}"
    clear_shared_memory(scope)
    cache_obj = Cache()
    try:
        comprehensive._configure_cache(cache_obj, cache_dir, mode, scope=scope)  # pylint: disable=protected-access
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            records = list(
                pool.map(
                    lambda item: _byte_request(
                        api_key, cache_obj, item, scenario_name=f"stress_burst_{mode}"
                    ),
                    items,
                )
            )
        wall_time_ms = (time.perf_counter() - wall_start) * 1000
        return _summarize_records(records, wall_time_ms=wall_time_ms)
    finally:
        comprehensive._release_cache(cache_obj, scope, cache_dir)  # pylint: disable=protected-access


def _render_money(value: float) -> str:
    return f"${value:.8f}"


def _render_mode_line(name: str, data: dict[str, Any]) -> str:
    return (
        f"- {name}: cost={_render_money(float(data.get('total_cost_usd', 0.0) or 0.0))}, "
        f"savings_ratio={data.get('savings_ratio', 0.0)}, hit_ratio={data.get('hit_ratio', 0.0)}, "
        f"accuracy={data.get('accuracy_ratio', 0.0)}, avg_latency={data.get('avg_latency_ms', 0.0)} ms, "
        f"p95={data.get('p95_latency_ms', 0.0)} ms, p99={data.get('p99_latency_ms', 0.0)} ms, "
        f"prompt_tokens={data.get('total_prompt_tokens', 0)}, prompt_token_reduction_ratio={data.get('prompt_token_reduction_ratio', 0.0)}"
    )


def _render_breakdown_table(title: str, breakdown: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Segment | Requests | Hit Ratio | Accuracy | Avg Latency (ms) | P95 (ms) | Cost (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in breakdown.items():
        lines.append(
            f"| {name} | {data.get('request_count', 0)} | {data.get('hit_ratio', 0.0)} | {data.get('accuracy_ratio', 0.0)} | "
            f"{data.get('avg_latency_ms', 0.0)} | {data.get('p95_latency_ms', 0.0)} | {_render_money(float(data.get('total_cost_usd', 0.0) or 0.0))} |"
        )
    lines.append("")
    return lines


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# ByteAI Cache OpenAI 100-Request Prompt Stress Benchmark",
        "",
        f"Generated: {results['generated_at']}",
        f"Chat model: `{results['chat_model']}`",
        "",
        "## Workload Composition",
        "",
        f"- Total requests: {results['request_count']}",
        f"- Concurrency burst size: {results['concurrent_burst_size']}",
        f"- Buckets: {results['composition']}",
        "",
        "## Sequential Overall",
        "",
    ]
    lines.append(_render_mode_line("Direct", results["sequential"]["direct"]))
    for mode in BYTE_MODES:
        lines.append(_render_mode_line(f"ByteAI Cache {mode}", results["sequential"][mode]))
    lines.extend(
        [
            "",
            "## Sequential Detail Table",
            "",
            "| Mode | Cost (USD) | Savings | Hit Ratio | Accuracy | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Prompt Tokens | Prompt Reduction | Throughput (RPS) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode_name in ["direct", *BYTE_MODES]:
        data = results["sequential"][mode_name]
        lines.append(
            f"| {mode_name} | {_render_money(float(data.get('total_cost_usd', 0.0) or 0.0))} | {data.get('savings_ratio', 0.0)} | "
            f"{data.get('hit_ratio', 0.0)} | {data.get('accuracy_ratio', 0.0)} | {data.get('avg_latency_ms', 0.0)} | "
            f"{data.get('p50_latency_ms', 0.0)} | {data.get('p95_latency_ms', 0.0)} | {data.get('p99_latency_ms', 0.0)} | "
            f"{data.get('total_prompt_tokens', 0)} | {data.get('prompt_token_reduction_ratio', 0.0)} | {data.get('throughput_rps', 0.0)} |"
        )
    lines.extend(
        [
            "",
            "## Bucket Breakdown",
            "",
        ]
    )
    lines.extend(
        _render_breakdown_table(
            "Direct By Bucket", results["sequential"]["direct"]["bucket_breakdown"]
        )
    )
    lines.extend(
        _render_breakdown_table(
            "Exact By Bucket", results["sequential"]["exact"]["bucket_breakdown"]
        )
    )
    lines.extend(
        _render_breakdown_table(
            "Normalized By Bucket", results["sequential"]["normalized"]["bucket_breakdown"]
        )
    )
    lines.extend([])
    lines.extend(
        _render_breakdown_table(
            "Hybrid By Bucket", results["sequential"]["hybrid"]["bucket_breakdown"]
        )
    )
    lines.extend(
        _render_breakdown_table("Hybrid By Kind", results["sequential"]["hybrid"]["kind_breakdown"])
    )
    lines.extend(
        [
            "## Cache Behavior",
            "",
            f"- Hybrid hit reasons: {results['sequential']['hybrid'].get('byte_reason_counts', {})}",
            f"- Normalized hit reasons: {results['sequential']['normalized'].get('byte_reason_counts', {})}",
            f"- Exact hit reasons: {results['sequential']['exact'].get('byte_reason_counts', {})}",
            "",
            "## Concurrent Burst",
            "",
            _render_mode_line("Direct burst", results["concurrent"]["direct"]),
            _render_mode_line("ByteAI Cache hybrid burst", results["concurrent"]["hybrid"]),
            "",
            "## Notes",
            "",
            "- This report stress-tests prompt workloads only. Image/audio/media surface metrics remain in the separate surface benchmark.",
            "- Live execution used OpenAI for runtime verification because that is the available provider key.",
            "- The runtime features under test are implemented in the shared Byte stack, so the same compiler, memory, and cache behavior applies across the other adapters.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_reports(report_path: Path, json_path: Path, results: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a 100-request OpenAI prompt stress benchmark for ByteAI Cache."
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key. Falls back to BYTE_TEST_OPENAI_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--json-report", default=str(DEFAULT_JSON_REPORT))
    args = parser.parse_args()

    api_key = (
        args.api_key
        or comprehensive.os.getenv("BYTE_TEST_OPENAI_API_KEY")
        or comprehensive.os.getenv("OPENAI_API_KEY")
    )  # pylint: disable=protected-access
    if not api_key:
        raise SystemExit("Missing API key. Set BYTE_TEST_OPENAI_API_KEY or pass --api-key.")

    items = _build_stress_items()
    burst_items = _burst_items(items)

    _, direct_summary = _run_direct_sequence(api_key, items)
    sequential = {"direct": direct_summary}
    for mode in BYTE_MODES:
        _, summary = _run_byte_sequence(api_key, items, mode=mode)
        _apply_baseline(summary, direct_summary)
        sequential[mode] = summary

    concurrent_direct = _run_direct_concurrent(api_key, burst_items)
    concurrent_hybrid = _run_byte_concurrent(api_key, burst_items, mode="hybrid")
    _apply_baseline(concurrent_hybrid, concurrent_direct)
    _apply_baseline(concurrent_direct, concurrent_direct)

    results = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "chat_model": CHAT_MODEL,
        "request_count": len(items),
        "concurrent_burst_size": len(burst_items),
        "composition": _bucket_counts(items),
        "artifacts": {
            "report": str(Path(args.report).resolve()),
            "json_report": str(Path(args.json_report).resolve()),
        },
        "sequential": sequential,
        "concurrent": {
            "direct": concurrent_direct,
            "hybrid": concurrent_hybrid,
        },
    }
    _write_reports(Path(args.report), Path(args.json_report), results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
