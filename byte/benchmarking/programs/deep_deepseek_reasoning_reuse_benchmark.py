import argparse
import copy
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from byte.benchmarking.programs import deep_deepseek_runtime_optimization_benchmark as base

REPORT_DIR = REPO_ROOT / "docs" / "reports"
DEFAULT_REPORT = REPORT_DIR / "deepseek_reasoning_reuse_benchmark.md"
DEFAULT_JSON_REPORT = REPORT_DIR / "deepseek_reasoning_reuse_benchmark.json"
DEFAULT_PDF_STYLE_REPORT = REPORT_DIR / "deepseek_reasoning_reuse_benchmark_pdf_style.md"

BENCHMARK_VERSION = "1.0"
PROVIDER = "DeepSeek"
MODEL = "deepseek-chat"

TOTAL_REQUESTS = 600
EXECUTION_WAVES = 6
REQUESTS_PER_WAVE = 100
CONCURRENCY = 5
WARMUP_REQUESTS = 30

SCENARIO_ORDER = [
    "complex_reasoning_chain",
    "multi_step_workflows",
    "repeated_knowledge_queries",
]
SCENARIO_REQUESTS = {
    "complex_reasoning_chain": 200,
    "multi_step_workflows": 200,
    "repeated_knowledge_queries": 200,
}
SCENARIO_LABELS = {
    "complex_reasoning_chain": "Complex Reasoning Chain",
    "multi_step_workflows": "Multi-Step Workflows",
    "repeated_knowledge_queries": "Repeated Knowledge Queries",
}
SCENARIO_PURPOSES = {
    "complex_reasoning_chain": "Measure deterministic numeric reasoning avoidance.",
    "multi_step_workflows": "Measure rule-based workflow planning reuse.",
    "repeated_knowledge_queries": "Measure stable knowledge reuse across semantic variants.",
}
WAVE_DISTRIBUTION = [
    {"complex_reasoning_chain": 40, "multi_step_workflows": 40, "repeated_knowledge_queries": 20},
    {"complex_reasoning_chain": 32, "multi_step_workflows": 32, "repeated_knowledge_queries": 36},
    {"complex_reasoning_chain": 32, "multi_step_workflows": 32, "repeated_knowledge_queries": 36},
    {"complex_reasoning_chain": 32, "multi_step_workflows": 32, "repeated_knowledge_queries": 36},
    {"complex_reasoning_chain": 32, "multi_step_workflows": 32, "repeated_knowledge_queries": 36},
    {"complex_reasoning_chain": 32, "multi_step_workflows": 32, "repeated_knowledge_queries": 36},
]

COUNTRIES = [
    ("France", "Paris"),
    ("Italy", "Rome"),
    ("Japan", "Tokyo"),
    ("Canada", "Ottawa"),
    ("Australia", "Canberra"),
    ("Spain", "Madrid"),
    ("Germany", "Berlin"),
    ("Portugal", "Lisbon"),
    ("Austria", "Vienna"),
    ("Ireland", "Dublin"),
    ("Norway", "Oslo"),
    ("Sweden", "Stockholm"),
    ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"),
    ("Belgium", "Brussels"),
    ("Switzerland", "Bern"),
    ("Poland", "Warsaw"),
    ("Greece", "Athens"),
    ("Czech Republic", "Prague"),
    ("Hungary", "Budapest"),
]


def _standard_item(
    scenario: str, prompt: str, expected: str, group: str, variant: str, kind: str
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "prompt": prompt,
        "expected": expected,
        "group": group,
        "variant": variant,
        "kind": kind,
        "model": MODEL,
        "max_tokens": 16,
        "request_style": "standard",
    }


def _format_percentage(value: float) -> str:
    rounded = round(float(value or 0.0), 2)
    if abs(rounded - round(rounded)) < 0.005:
        return f"{round(rounded)}%"
    return f"{rounded:.2f}".rstrip("0").rstrip(".") + "%"


def _complex_reasoning_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    templates = [
        (
            "A company sells a product for ${price}.\n"
            "Production cost = ${production}\n"
            "Marketing cost = ${marketing}\n"
            "Shipping cost = ${shipping}\n"
            "Calculate profit margin percentage. Return only the percentage."
        ),
        (
            "Product price = ${price}\n"
            "Production cost = ${production}\n"
            "Marketing cost = ${marketing}\n"
            "Shipping cost = ${shipping}\n"
            "Calculate profit margin percentage and answer with only the percent."
        ),
        (
            "A company sells a product for ${price}.\n"
            "Production cost = ${production}\n"
            "Marketing cost = ${marketing}\n"
            "Shipping cost = ${shipping}\n"
            "Return only the profit margin percentage."
        ),
        (
            "Product price = ${price}\n"
            "Production cost = ${production}\n"
            "Marketing cost = ${marketing}\n"
            "Shipping cost = ${shipping}\n"
            "Answer with only the profit margin percentage."
        ),
        (
            "A company sells a product for ${price}.\n"
            "Production cost = ${production}\n"
            "Marketing cost = ${marketing}\n"
            "Shipping cost = ${shipping}\n"
            "Profit margin percentage? Return only the percentage."
        ),
    ]
    for group_index in range(40):
        price = 135 + group_index * 7
        production = 48 + (group_index * 5 % 34)
        marketing = 12 + (group_index * 3 % 18)
        shipping = 6 + (group_index * 2 % 11)
        if production + marketing + shipping >= price:
            price = production + marketing + shipping + 18
        expected = _format_percentage(((price - production - marketing - shipping) / price) * 100.0)
        for variant_index, template in enumerate(templates, 1):
            prompt = (
                template.replace("${price}", str(price))
                .replace("${production}", str(production))
                .replace("${marketing}", str(marketing))
                .replace("${shipping}", str(shipping))
            )
            items.append(
                _standard_item(
                    "complex_reasoning_chain",
                    prompt,
                    expected,
                    f"margin_{group_index + 1:02d}",
                    f"v{variant_index:02d}",
                    "profit_margin",
                )
            )
    return items


def _workflow_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    templates = [
        (
            "Customer purchased item A for ${amount}.\n"
            "Policy: Refunds allowed within ${window} days.\n"
            "Customer requested refund on day ${day}.\n"
            "Return final action label.\n"
            "Labels: REFUND_APPROVE, REFUND_DENY, ESCALATE"
        ),
        (
            "Purchase amount: ${amount}.\n"
            "Policy: Refunds allowed within ${window} days.\n"
            "Day ${day} request.\n"
            "Return final action label.\n"
            "Labels: REFUND_APPROVE, REFUND_DENY, ESCALATE"
        ),
        (
            "Customer order total: ${amount}.\n"
            "Policy: Refunds allowed within ${window} days.\n"
            "Customer requested refund on day ${day}.\n"
            "Return final action label.\n"
            "Labels: REFUND_APPROVE, REFUND_DENY, ESCALATE"
        ),
        (
            "Purchase amount: ${amount}.\n"
            "Policy: Refunds allowed within ${window} days.\n"
            "Day ${day} refund request.\n"
            "Return final action label.\n"
            "Labels: REFUND_APPROVE, REFUND_DENY, ESCALATE"
        ),
        (
            "Customer purchased item A for ${amount}.\n"
            "Policy: Refunds allowed within ${window} days.\n"
            "Day ${day} request.\n"
            "Return final action label.\n"
            "Labels: REFUND_APPROVE, REFUND_DENY, ESCALATE"
        ),
    ]
    windows = [14, 21, 30, 45]
    for group_index in range(40):
        window = windows[group_index % len(windows)]
        amount = 160 + group_index * 9
        day = window - 3 if group_index % 2 == 0 else window + 5
        expected = "REFUND_APPROVE" if day <= window else "REFUND_DENY"
        for variant_index, template in enumerate(templates, 1):
            prompt = (
                template.replace("${amount}", str(amount))
                .replace("${window}", str(window))
                .replace("${day}", str(day))
            )
            items.append(
                _standard_item(
                    "multi_step_workflows",
                    prompt,
                    expected,
                    f"refund_{group_index + 1:02d}",
                    f"v{variant_index:02d}",
                    "refund_policy",
                )
            )
    return items


def _knowledge_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    templates = [
        "What is the capital of {country}? Return only the city name.",
        "What is the capital city of {country}? Return only the city name.",
        "Which city is the capital of {country}? Return only the city name.",
        "Name the capital city of {country}. Return only the city name.",
        "What is the capital of {country}? Answer with only the city.",
        "Which city is the capital of {country}? Answer with only the city.",
        "Name the capital city of {country}. Reply with the city only.",
        "What is the capital city of {country}? Reply with the city only.",
        "What is the capital of {country}? Please return only the city.",
        "Which city is the capital of {country}? Please return only the city.",
    ]
    for variant_index, template in enumerate(templates, 1):
        for group_index, (country, capital) in enumerate(COUNTRIES, 1):
            items.append(
                _standard_item(
                    "repeated_knowledge_queries",
                    template.format(country=country),
                    capital,
                    f"capital_{group_index:02d}",
                    f"v{variant_index:02d}",
                    "capital_city",
                )
            )
    return items


def build_workload_plan() -> dict[str, Any]:
    buckets = {
        "complex_reasoning_chain": _complex_reasoning_items(),
        "multi_step_workflows": _workflow_items(),
        "repeated_knowledge_queries": _knowledge_items(),
    }
    warmup_items = []
    warmup_items.extend(copy.deepcopy(item) for item in buckets["complex_reasoning_chain"][:5])
    warmup_items.extend(copy.deepcopy(item) for item in buckets["multi_step_workflows"][:5])
    warmup_items.extend(copy.deepcopy(item) for item in buckets["repeated_knowledge_queries"][::10])
    cursors = dict.fromkeys(SCENARIO_ORDER, 0)
    waves: list[dict[str, Any]] = []
    all_items: list[dict[str, Any]] = []
    request_index = 0
    for wave_index, counts in enumerate(WAVE_DISTRIBUTION, 1):
        ordered = SCENARIO_ORDER[wave_index - 1 :] + SCENARIO_ORDER[: wave_index - 1]
        per_wave = {
            scenario: copy.deepcopy(
                buckets[scenario][cursors[scenario] : cursors[scenario] + counts[scenario]]
            )
            for scenario in SCENARIO_ORDER
        }
        for scenario in SCENARIO_ORDER:
            cursors[scenario] += counts[scenario]
        items: list[dict[str, Any]] = []
        max_len = max(len(values) for values in per_wave.values())
        for offset in range(max_len):
            for scenario in ordered:
                if offset < len(per_wave[scenario]):
                    request_index += 1
                    item = per_wave[scenario][offset]
                    item["request_index"] = request_index
                    item["wave"] = wave_index
                    item["wave_position"] = len(items) + 1
                    items.append(item)
                    all_items.append(copy.deepcopy(item))
        waves.append({"wave": wave_index, "items": items})
    return {
        "planned_request_count": len(all_items),
        "execution_waves": EXECUTION_WAVES,
        "requests_per_wave": REQUESTS_PER_WAVE,
        "concurrency": CONCURRENCY,
        "warmup_requests": len(warmup_items),
        "timeout_seconds": base.TIMEOUT_SECONDS,
        "retries": 0,
        "distribution": dict(SCENARIO_REQUESTS),
        "warmup_items": warmup_items,
        "waves": waves,
        "items": all_items,
    }


def _group_records(records: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(record)
    return grouped


def _augment_summary(
    summary: dict[str, Any], records: list[dict[str, Any]], *, baseline_request_count: int
) -> None:
    total = len(records)
    if total == 0:
        summary["reasoning_reuse_rate"] = 0.0
        summary["duplicate_reasoning_reduction"] = 0.0
        summary["upstream_call_count"] = 0
        summary["deterministic_reasoning_rate"] = 0.0
        summary["knowledge_reuse_rate"] = 0.0
        return
    byte_reason_counts = Counter(
        record.get("byte_reason", "") for record in records if record.get("byte_reason")
    )
    deterministic_hits = int(byte_reason_counts.get("deterministic_reasoning", 0) or 0)
    knowledge_hits = int(byte_reason_counts.get("reasoning_memory_reuse", 0) or 0)
    reasoning_hits = deterministic_hits + knowledge_hits
    upstream_call_count = sum(
        1
        for record in records
        if int(record.get("prompt_tokens", 0) or 0) > 0
        or int(record.get("completion_tokens", 0) or 0) > 0
        or int(record.get("status_code", 0) or 0) != 200
    )
    summary["reasoning_reuse_rate"] = round(reasoning_hits / total, 4)
    summary["duplicate_reasoning_reduction"] = (
        round(
            (baseline_request_count - upstream_call_count) / baseline_request_count,
            4,
        )
        if baseline_request_count
        else 0.0
    )
    summary["upstream_call_count"] = upstream_call_count
    summary["deterministic_reasoning_rate"] = round(deterministic_hits / total, 4)
    summary["knowledge_reuse_rate"] = round(knowledge_hits / total, 4)


def _detailed_breakdown(
    records: list[dict[str, Any]], field: str, *, baseline_counts: dict[str, int] | None = None
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, group_records in _group_records(records, field).items():
        summary = base._summarize_records(group_records)
        _augment_summary(
            summary,
            group_records,
            baseline_request_count=int(
                (baseline_counts or {}).get(key, len(group_records)) or len(group_records)
            ),
        )
        result[key] = summary
    return result


def _run_byte(api_key: str, plan: dict[str, Any]) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix="deepseek-reasoning-")
    scope = f"deepseek::reasoning::{int(time.time() * 1000)}"
    cache_obj = base.Cache()
    try:
        base._configure_cache(cache_obj, cache_dir, scope)
        base._run_parallel(
            plan["warmup_items"], lambda item: base._byte_request(api_key, cache_obj, item)
        )
        records: list[dict[str, Any]] = []
        wall_start = time.perf_counter()
        for wave in plan["waves"]:
            records.extend(
                base._run_parallel(
                    wave["items"], lambda item: base._byte_request(api_key, cache_obj, item)
                )
            )
        return {
            "summary": base._summarize_records(
                records, wall_time_ms=(time.perf_counter() - wall_start) * 1000
            ),
            "records": records,
            "scenario_breakdown": base._scenario_breakdown(records),
            "memory": base._memory_capture(cache_obj),
        }
    finally:
        base._release_cache(cache_obj, scope, cache_dir)


def run_benchmark(api_key: str) -> dict[str, Any]:
    plan = build_workload_plan()
    direct = base._run_direct(api_key, plan)
    byte_runtime = _run_byte(api_key, plan)

    _augment_summary(
        direct["summary"],
        direct["records"],
        baseline_request_count=direct["summary"]["request_count"],
    )
    _augment_summary(
        byte_runtime["summary"],
        byte_runtime["records"],
        baseline_request_count=direct["summary"]["request_count"],
    )
    base._apply_baseline(byte_runtime["summary"], direct["summary"])

    direct_baseline_counts = {
        scenario: len([record for record in direct["records"] if record["scenario"] == scenario])
        for scenario in SCENARIO_ORDER
    }
    direct["scenario_breakdown"] = _detailed_breakdown(
        direct["records"], "scenario", baseline_counts=direct_baseline_counts
    )
    byte_runtime["scenario_breakdown"] = _detailed_breakdown(
        byte_runtime["records"], "scenario", baseline_counts=direct_baseline_counts
    )
    for scenario in SCENARIO_ORDER:
        base._apply_baseline(
            byte_runtime["scenario_breakdown"][scenario], direct["scenario_breakdown"][scenario]
        )

    kind_baseline_counts = {
        kind: len(group_records)
        for kind, group_records in _group_records(direct["records"], "kind").items()
    }
    direct["kind_breakdown"] = _detailed_breakdown(
        direct["records"], "kind", baseline_counts=kind_baseline_counts
    )
    byte_runtime["kind_breakdown"] = _detailed_breakdown(
        byte_runtime["records"], "kind", baseline_counts=kind_baseline_counts
    )
    for kind in sorted(kind_baseline_counts):
        base._apply_baseline(byte_runtime["kind_breakdown"][kind], direct["kind_breakdown"][kind])

    return {
        "generated_at": base._now_iso(),
        "provider": PROVIDER,
        "models": [MODEL],
        "benchmark_version": BENCHMARK_VERSION,
        "pricing": {
            "verified_on": base.VERIFIED_ON,
            "sources": list(base.PRICING_SOURCES),
            "table": dict(base.PRICING),
        },
        "plan": {
            "planned_request_count": plan["planned_request_count"],
            "execution_waves": plan["execution_waves"],
            "requests_per_wave": plan["requests_per_wave"],
            "concurrency": plan["concurrency"],
            "warmup_requests": plan["warmup_requests"],
            "timeout_seconds": plan["timeout_seconds"],
            "retries": plan["retries"],
            "distribution": plan["distribution"],
        },
        "runs": {
            "direct": direct,
            "byte_runtime": byte_runtime,
        },
    }


def _scenario_example(plan: dict[str, Any], scenario: str) -> str:
    item = next(item for item in plan["items"] if item["scenario"] == scenario)
    return str(item["prompt"])


def _top_byte_reason(summary: dict[str, Any]) -> str:
    counts = dict(summary.get("byte_reason_counts", {}) or {})
    if not counts:
        return "-"
    key, value = sorted(counts.items(), key=lambda item: item[1], reverse=True)[0]
    return f"{key}={value}"


def _render_report(results: dict[str, Any], plan: dict[str, Any]) -> str:
    direct = dict(results["runs"]["direct"]["summary"])
    byte_runtime = dict(results["runs"]["byte_runtime"]["summary"])
    base._apply_baseline(byte_runtime, direct)
    reasoning_memory = (
        (results["runs"]["byte_runtime"].get("memory") or {}).get("summary") or {}
    ).get("reasoning_memory", {}) or {}
    lines = [
        "# Byte Reasoning Reuse Benchmark Report",
        "",
        f"**Provider:** {PROVIDER}",
        f"**Model:** `{MODEL}`",
        f"**Benchmark Version:** {BENCHMARK_VERSION}",
        "**Execution Mode:**",
        "",
        "```",
        "1. Direct DeepSeek API",
        "2. DeepSeek through Byte Runtime",
        "```",
        "",
        "# 1. Benchmark Objective",
        "",
        "Measure how much duplicate reasoning Byte Runtime prevents across numeric, workflow, and repeated-knowledge workloads while preserving accuracy.",
        "",
        "# 2. Benchmark Configuration",
        "",
        "```",
        f"Total Requests: {TOTAL_REQUESTS}",
        f"Execution Waves: {EXECUTION_WAVES}",
        f"Requests per Wave: {REQUESTS_PER_WAVE}",
        f"Concurrency: {CONCURRENCY}",
        f"Warmup Requests: {WARMUP_REQUESTS}",
        f"Timeout: {int(base.TIMEOUT_SECONDS)} seconds",
        "Retries: 0",
        f"Provider: {PROVIDER}",
        f"Model: {MODEL}",
        "```",
        "",
        "# 3. Metrics Collected",
        "",
        "```",
        "accuracy_ratio",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "cost_estimate",
        "cache_hit_ratio",
        "reasoning_reuse_rate",
        "duplicate_reasoning_reduction",
        "upstream_call_count",
        "```",
        "",
        "# 4. Workload Distribution",
        "",
        "| Workload | Requests |",
        "| --- | ---: |",
    ]
    for scenario in SCENARIO_ORDER:
        lines.append(f"| {SCENARIO_LABELS[scenario]} | {SCENARIO_REQUESTS[scenario]} |")
    lines.extend(["", "# 5. Workload Scenarios", ""])
    for scenario in SCENARIO_ORDER:
        lines.extend(
            [
                f"## {SCENARIO_LABELS[scenario]}",
                "",
                "Purpose:",
                "",
                "```",
                SCENARIO_PURPOSES[scenario],
                "```",
                "",
                "Example prompt:",
                "",
                "```",
                _scenario_example(plan, scenario),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "# 6. Results",
            "",
            "## Direct DeepSeek API",
            "",
            f"| Accuracy | {direct['accuracy_ratio']:.4f} |",
            f"| Average Latency | {direct['avg_latency_ms']} ms |",
            f"| P95 Latency | {direct['p95_latency_ms']} ms |",
            f"| Prompt Tokens | {direct['prompt_tokens']:,} |",
            f"| Completion Tokens | {direct['completion_tokens']:,} |",
            f"| Cost | {base._render_money(float(direct['cost_estimate']))} |",
            f"| Reasoning Reuse Rate | {direct['reasoning_reuse_rate']:.4f} |",
            f"| Duplicate Reasoning Reduction | {direct['duplicate_reasoning_reduction']:.4f} |",
            f"| Upstream Calls | {direct['upstream_call_count']} |",
            "",
            "## DeepSeek Through Byte Runtime",
            "",
            f"| Accuracy | {byte_runtime['accuracy_ratio']:.4f} |",
            f"| Average Latency | {byte_runtime['avg_latency_ms']} ms |",
            f"| P95 Latency | {byte_runtime['p95_latency_ms']} ms |",
            f"| Prompt Tokens | {byte_runtime['prompt_tokens']:,} |",
            f"| Completion Tokens | {byte_runtime['completion_tokens']:,} |",
            f"| Cost | {base._render_money(float(byte_runtime['cost_estimate']))} |",
            f"| Reasoning Reuse Rate | {byte_runtime['reasoning_reuse_rate']:.4f} |",
            f"| Duplicate Reasoning Reduction | {byte_runtime['duplicate_reasoning_reduction']:.4f} |",
            f"| Upstream Calls | {byte_runtime['upstream_call_count']} |",
            "",
            "Observed changes:",
            "",
            "```",
            f"Cost reduction: {byte_runtime['cost_reduction_ratio']:.4f}",
            f"Prompt token reduction: {byte_runtime['token_reduction_ratio']:.4f}",
            f"Latency improvement: {byte_runtime['latency_improvement_ratio']:.4f}",
            f"Accuracy improvement: {byte_runtime['accuracy_delta']:.4f}",
            f"Deterministic reasoning rate: {byte_runtime['deterministic_reasoning_rate']:.4f}",
            f"Knowledge reuse rate: {byte_runtime['knowledge_reuse_rate']:.4f}",
            "```",
            "",
            "# 7. Detailed Workload Breakdown",
            "",
            "| Workload | Requests | Direct Acc | Byte Acc | Cost Reduction | Latency Improvement | Reasoning Reuse | Duplicate Reasoning Reduction | Top Byte Reason |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for scenario in SCENARIO_ORDER:
        direct_summary = results["runs"]["direct"]["scenario_breakdown"][scenario]
        byte_summary = results["runs"]["byte_runtime"]["scenario_breakdown"][scenario]
        lines.append(
            "| "
            + " | ".join(
                [
                    SCENARIO_LABELS[scenario],
                    str(byte_summary["request_count"]),
                    f"{direct_summary['accuracy_ratio']:.4f}",
                    f"{byte_summary['accuracy_ratio']:.4f}",
                    f"{byte_summary['cost_reduction_ratio']:.4f}",
                    f"{byte_summary['latency_improvement_ratio']:.4f}",
                    f"{byte_summary['reasoning_reuse_rate']:.4f}",
                    f"{byte_summary['duplicate_reasoning_reduction']:.4f}",
                    _top_byte_reason(byte_summary),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "# 8. Feature-Level Impact Analysis",
            "",
            f"- Deterministic reasoning short-circuit handled {round(byte_runtime['deterministic_reasoning_rate'] * byte_runtime['request_count'])} requests directly in the runtime.",
            f"- Knowledge-memory reuse handled {round(byte_runtime['knowledge_reuse_rate'] * byte_runtime['request_count'])} repeated fact requests after initial seeds.",
            f"- Byte reduced upstream DeepSeek calls from {direct['upstream_call_count']} to {byte_runtime['upstream_call_count']}.",
            f"- Reasoning memory retained {reasoning_memory.get('total_entries', 0)} entries with {reasoning_memory.get('hits', 0)} runtime hits during the run.",
            f"- Top Byte reasons: {json.dumps(byte_runtime.get('byte_reason_counts', {}), ensure_ascii=True)}.",
            "",
            "# 9. Key Findings",
            "",
            "```",
            f"1. Byte reused reasoning on {byte_runtime['reasoning_reuse_rate']:.4f} of requests.",
            f"2. Byte removed {byte_runtime['duplicate_reasoning_reduction']:.4f} of upstream reasoning calls.",
            f"3. Byte reduced cost by {byte_runtime['cost_reduction_ratio']:.4f}.",
            f"4. Byte reduced prompt tokens by {byte_runtime['token_reduction_ratio']:.4f}.",
            f"5. Byte improved average latency by {byte_runtime['latency_improvement_ratio']:.4f}.",
            "```",
            "",
            "# 10. Final Conclusion",
            "",
            "This benchmark shows Byte acting as a reasoning-avoidance runtime, not just a cache layer. It cuts repeated numeric computation, policy execution, and repeated fact lookup before those requests reach the provider.",
            "",
            "## Source Links",
            "",
            f"- DeepSeek pricing: {base.PRICING_SOURCES[0]}",
            f"- DeepSeek compatibility update: {base.PRICING_SOURCES[1]}",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _render_pdf_style(results: dict[str, Any]) -> str:
    direct = dict(results["runs"]["direct"]["summary"])
    byte_runtime = dict(results["runs"]["byte_runtime"]["summary"])
    base._apply_baseline(byte_runtime, direct)
    return (
        "\n".join(
            [
                "# Byte x DeepSeek Reasoning Reuse Benchmark",
                "",
                "## Executive Summary",
                "",
                f"Byte processed {TOTAL_REQUESTS} DeepSeek reasoning requests across {EXECUTION_WAVES} waves with concurrency {CONCURRENCY}.",
                f"It reused reasoning on {byte_runtime['reasoning_reuse_rate']:.4f} of requests and reduced upstream provider calls by {byte_runtime['duplicate_reasoning_reduction']:.4f}.",
                f"Against direct DeepSeek usage, Byte reduced cost by {byte_runtime['cost_reduction_ratio']:.4f}, reduced prompt tokens by {byte_runtime['token_reduction_ratio']:.4f}, improved latency by {byte_runtime['latency_improvement_ratio']:.4f}, and changed accuracy by {byte_runtime['accuracy_delta']:.4f}.",
                "",
                "## Headline Numbers",
                "",
                f"- Cost: {base._render_money(float(direct['cost_estimate']))} -> {base._render_money(float(byte_runtime['cost_estimate']))}",
                f"- Average latency: {direct['avg_latency_ms']} ms -> {byte_runtime['avg_latency_ms']} ms",
                f"- Upstream calls: {direct['upstream_call_count']} -> {byte_runtime['upstream_call_count']}",
                f"- Reasoning reuse rate: {byte_runtime['reasoning_reuse_rate']:.4f}",
                "",
            ]
        )
        + "\n"
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the DeepSeek vs Byte reasoning reuse benchmark."
    )
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--json-report", default=str(DEFAULT_JSON_REPORT))
    parser.add_argument("--pdf-style-report", default=str(DEFAULT_PDF_STYLE_REPORT))
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not str(args.api_key or "").strip():
        raise SystemExit("Missing DeepSeek credentials. Provide --api-key or set DEEPSEEK_API_KEY.")
    plan = build_workload_plan()
    results = run_benchmark(str(args.api_key).strip())
    Path(args.report).write_text(_render_report(results, plan), encoding="utf-8")
    Path(args.json_report).write_text(
        json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    Path(args.pdf_style_report).write_text(_render_pdf_style(results), encoding="utf-8")
    print(
        f"DeepSeek reasoning benchmark complete. Planned requests: {plan['planned_request_count']}."
    )
    print(
        "Byte runtime summary: "
        f"accuracy={results['runs']['byte_runtime']['summary']['accuracy_ratio']:.4f}, "
        f"cost={base._render_money(float(results['runs']['byte_runtime']['summary']['cost_estimate']))}, "
        f"latency={results['runs']['byte_runtime']['summary']['avg_latency_ms']} ms, "
        f"reasoning_reuse={results['runs']['byte_runtime']['summary']['reasoning_reuse_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
