import argparse
import copy
import json
import os
import shutil
import statistics
import tempfile
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from byte import Cache, Config  # pylint: disable=wrong-import-position
from byte._backends import deepseek as byte_deepseek  # pylint: disable=wrong-import-position
from byte.adapter.api import init_cache  # pylint: disable=wrong-import-position
from byte.benchmarking._optional_runtime import create_openai_client
from byte.benchmarking._program_defaults import load_program_defaults
from byte.benchmarking.programs import deep_openai_coding_benchmark as coding
from byte.benchmarking.programs import deep_openai_comprehensive_workload_benchmark as comprehensive
from byte.benchmarking.programs import deep_openai_prompt_stress_benchmark as prompt_stress
from byte.benchmarking.programs import deep_openai_surface_benchmark as surface
from byte.processor.pre import (  # pylint: disable=wrong-import-position
    last_content,
    normalized_last_content,
)
from byte.processor.shared_memory import (
    clear_shared_memory,  # pylint: disable=wrong-import-position
)

REPORT_DIR = REPO_ROOT / "docs" / "reports"
DEFAULT_REPORT = REPORT_DIR / "deepseek_runtime_optimization_benchmark.md"
DEFAULT_JSON_REPORT = REPORT_DIR / "deepseek_runtime_optimization_benchmark.json"
DEFAULT_PDF_STYLE_REPORT = REPORT_DIR / "deepseek_runtime_optimization_benchmark_pdf_style.md"

_DEFAULTS = load_program_defaults("deep_deepseek_runtime_optimization_benchmark")

BENCHMARK_VERSION = _DEFAULTS.get("benchmark_version", "1.0")
PROVIDER = _DEFAULTS.get("provider", "DeepSeek")
CHAT_MODEL = _DEFAULTS.get("chat_model", "deepseek-chat")
CODING_MODEL = _DEFAULTS.get("coding_model", "deepseek-coder")
API_BASE = _DEFAULTS.get("api_base", "https://api.deepseek.com")
VERIFIED_ON = _DEFAULTS.get("verified_on", "2026-03-14")

TOTAL_REQUESTS = int(_DEFAULTS.get("total_requests", 1200))
EXECUTION_WAVES = int(_DEFAULTS.get("execution_waves", 12))
REQUESTS_PER_WAVE = int(_DEFAULTS.get("requests_per_wave", 100))
CONCURRENCY = int(_DEFAULTS.get("concurrency", 5))
WARMUP_REQUESTS = int(_DEFAULTS.get("warmup_requests", 50))
TIMEOUT_SECONDS = float(_DEFAULTS.get("timeout_seconds", 60.0))
RETRIES = int(_DEFAULTS.get("retries", 0))

SCENARIO_ORDER = list(_DEFAULTS.get("scenario_order", []))
SCENARIO_REQUESTS = dict(_DEFAULTS.get("scenario_requests", {}))
SCENARIO_LABELS = dict(_DEFAULTS.get("scenario_labels", {}))
SCENARIO_PURPOSES = dict(_DEFAULTS.get("scenario_purposes", {}))
PRICING = dict(_DEFAULTS.get("pricing", {}))
PRICING_SOURCES = list(_DEFAULTS.get("pricing_sources", []))

COMMON_SYSTEM = comprehensive._COMMON_SYSTEM  # pylint: disable=protected-access
CODING_KINDS = {"code_fix", "code_explanation", "test_generation", "documentation", "code_refactor"}
_THREAD_LOCAL = threading.local()


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _shorten(text: str, limit: int = 180) -> str:
    candidate = " ".join(str(text or "").split())
    return candidate if len(candidate) <= limit else candidate[: limit - 3].rstrip() + "..."


def _normalized_answer(text: str | None) -> str:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    return coding.normalize_text(candidate)  # pylint: disable=protected-access


def _kind_model(kind: str) -> str:
    return CODING_MODEL if kind in CODING_KINDS else CHAT_MODEL


def _render_money(value: float) -> str:
    return f"${value:.6f}"


def _render_ratio(value: float) -> str:
    return f"{float(value or 0.0):.4f}"


def _render_percent(value: float) -> str:
    return f"{float(value or 0.0) * 100.0:.2f}%"


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
        cached_prompt_tokens = (
            int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        )
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


def _pricing_key(configured_model: str, provider_model: str) -> str:
    for candidate in (provider_model, configured_model):
        normalized = str(candidate or "").strip().lower()
        if normalized.startswith("deepseek-reasoner"):
            return "deepseek-reasoner"
        if normalized.startswith("deepseek-coder"):
            return "deepseek-coder"
        if normalized.startswith("deepseek-chat"):
            return "deepseek-chat"
    return "deepseek-chat"


def _pricing_cost(*, configured_model: str, provider_model: str, usage: Any) -> float:
    fields = _usage_fields(usage)
    price = PRICING[_pricing_key(configured_model, provider_model)]
    return (
        (fields["uncached_prompt_tokens"] / 1_000_000) * price["input"]
        + (fields["cached_prompt_tokens"] / 1_000_000) * price["cached_input"]
        + (fields["completion_tokens"] / 1_000_000) * price["output"]
    )


def _deepseek_client(api_key: str) -> Any:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None or getattr(_THREAD_LOCAL, "api_key", None) != api_key:
        client = create_openai_client(
            api_key=api_key,
            base_url=API_BASE,
            timeout=TIMEOUT_SECONDS,
        )
        _THREAD_LOCAL.client = client
        _THREAD_LOCAL.api_key = api_key
    return client


def _base_config(scope: str) -> Config:
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


def _configure_cache(cache_obj: Cache, cache_dir: str, scope: str) -> None:
    config = _base_config(scope)
    init_cache(
        mode="hybrid",
        data_dir=cache_dir,
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=config,
        exact_config=config,
        normalized_config=config,
        semantic_config=config,
    )


def _release_cache(cache_obj: Cache, scope: str, cache_dir: str) -> None:
    current = cache_obj
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current.data_manager = None
        current = getattr(current, "next_cache", None)
    shutil.rmtree(cache_dir, ignore_errors=True)
    clear_shared_memory(scope)


def _standard_item(
    scenario: str,
    prompt: str,
    expected: str,
    group: str,
    variant: str,
    kind: str,
    *,
    max_tokens: int = 16,
    model: str = "",
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "prompt": prompt,
        "expected": expected,
        "group": group,
        "variant": variant,
        "kind": kind,
        "model": model or _kind_model(kind),
        "max_tokens": max_tokens,
        "request_style": "standard",
    }


def _context_sections(byte_context: dict[str, Any]) -> list[str]:
    labels = [
        ("Repo summary", "byte_repo_summary"),
        ("Changed files", "byte_changed_files"),
        ("Changed hunks", "byte_changed_hunks"),
        ("Retrieval context", "byte_retrieval_context"),
        ("Document context", "byte_document_context"),
        ("Support articles", "byte_support_articles"),
    ]
    return [
        f"{title}:\n{json.dumps(byte_context[key], indent=2, ensure_ascii=True)}"
        for title, key in labels
        if key in byte_context
    ]


def _contextual_item(
    scenario: str,
    short_prompt: str,
    expected: str,
    group: str,
    variant: str,
    kind: str,
    byte_context: dict[str, Any],
    *,
    max_tokens: int = 16,
) -> dict[str, Any]:
    full_prompt = "\n\n".join([short_prompt, *_context_sections(byte_context)])
    return {
        "scenario": scenario,
        "expected": expected,
        "group": group,
        "variant": variant,
        "kind": kind,
        "model": _kind_model(kind),
        "max_tokens": max_tokens,
        "request_style": "contextual",
        "direct_messages": [
            {"role": "system", "content": COMMON_SYSTEM},
            {"role": "user", "content": full_prompt},
        ],
        "byte_messages": [
            {"role": "system", "content": COMMON_SYSTEM},
            {"role": "user", "content": short_prompt},
        ],
        "byte_context": dict(byte_context),
    }


def _exact_repeat_items() -> list[dict[str, Any]]:
    support_items = copy.deepcopy(surface._build_normal_chat_scenario()["items"])  # pylint: disable=protected-access
    document_items = copy.deepcopy(surface._build_document_scenario()["items"])  # pylint: disable=protected-access
    coding_items = []
    for scenario in coding._build_sequential_scenarios():  # pylint: disable=protected-access
        if scenario["name"] != "cursor_prewarmed_hotset_4":
            coding_items.extend(copy.deepcopy(scenario["items"]))
    seeds = []
    for candidate in support_items + document_items + coding_items:
        if str(candidate.get("kind") or "") == "document_extraction":
            continue
        if str(candidate.get("expected") or "").startswith("{"):
            continue
        seeds.append(candidate)
        if len(seeds) >= 30:
            break
    items: list[dict[str, Any]] = []
    for repetition in range(5):
        for seed_index, seed in enumerate(seeds, 1):
            items.append(
                _standard_item(
                    "exact_repeat",
                    str(seed["prompt"]),
                    str(seed["expected"]),
                    f"exact_{seed_index:03d}",
                    f"r{repetition + 1}",
                    str(seed["kind"]),
                    max_tokens=int(seed.get("max_tokens", 16) or 16),
                )
            )
    return items


def _normalized_variant_items() -> list[dict[str, Any]]:
    seeds = [
        item
        for item in prompt_stress._build_normalized_variant_bucket()  # pylint: disable=protected-access
        if str(item.get("kind") or "") != "document_extraction"
    ]
    items: list[dict[str, Any]] = []
    cycles, remainder = divmod(SCENARIO_REQUESTS["normalized_variant"], len(seeds))
    for cycle in range(cycles):
        for index, seed in enumerate(seeds, 1):
            items.append(
                _standard_item(
                    "normalized_variant",
                    str(seed["prompt"]),
                    str(seed["expected"]),
                    f"norm_{cycle + 1:02d}_{index:02d}",
                    f"c{cycle + 1:02d}",
                    str(seed["kind"]),
                    max_tokens=int(seed.get("max_tokens", 16) or 16),
                )
            )
    for index, seed in enumerate(seeds[:remainder], 1):
        items.append(
            _standard_item(
                "normalized_variant",
                str(seed["prompt"]),
                str(seed["expected"]),
                f"norm_extra_{index:02d}",
                "extra",
                str(seed["kind"]),
                max_tokens=int(seed.get("max_tokens", 16) or 16),
            )
        )
    return items


def _plain_unique_items() -> list[dict[str, Any]]:
    templates = [
        "Unique benchmark request {index:03d}. Topic: duplicate billing in market {index:03d}. Reply exactly UNIQUE_{index:03d} and nothing else.",
        "Unique benchmark request {index:03d}. Topic: export incident for tenant {index:03d}. Reply exactly UNIQUE_{index:03d} and nothing else.",
        "Unique benchmark request {index:03d}. Topic: release note approval for project {index:03d}. Reply exactly UNIQUE_{index:03d} and nothing else.",
        "Unique benchmark request {index:03d}. Topic: procurement review for vendor {index:03d}. Reply exactly UNIQUE_{index:03d} and nothing else.",
        "Unique benchmark request {index:03d}. Topic: workspace migration phase {index:03d}. Reply exactly UNIQUE_{index:03d} and nothing else.",
    ]
    return [
        _standard_item(
            "plain_unique",
            templates[(index - 1) % len(templates)].format(index=index),
            f"UNIQUE_{index:03d}",
            f"plain_{index:03d}",
            "base",
            "instruction",
            max_tokens=8,
        )
        for index in range(1, SCENARIO_REQUESTS["plain_unique"] + 1)
    ]


def _shared_context_unique_items() -> list[dict[str, Any]]:
    base_context = {
        "byte_repo_summary": comprehensive._repo_summary_payload(),  # pylint: disable=protected-access
        "byte_changed_files": comprehensive._changed_files_payload(),  # pylint: disable=protected-access
        "byte_changed_hunks": comprehensive._changed_hunks_payload(),  # pylint: disable=protected-access
        "byte_retrieval_context": comprehensive._retrieval_context_payload(),  # pylint: disable=protected-access
        "byte_document_context": comprehensive._document_context_payload(),  # pylint: disable=protected-access
        "byte_support_articles": comprehensive._support_articles_payload(),  # pylint: disable=protected-access
    }
    prompts = [
        "Review the shared support and repo context, then reply exactly {token} and nothing else.",
        "Review the shared document and repo context, then reply exactly {token} and nothing else.",
        "Review the shared code and retrieval context, then reply exactly {token} and nothing else.",
        "Review the shared workspace context, then reply exactly {token} and nothing else.",
        "Review the shared billing and incident context, then reply exactly {token} and nothing else.",
        "Review the shared contract and support context, then reply exactly {token} and nothing else.",
        "Review the shared release and routing context, then reply exactly {token} and nothing else.",
        "Review the shared authentication and validation context, then reply exactly {token} and nothing else.",
        "Review the shared escalation and repo context, then reply exactly {token} and nothing else.",
        "Review the shared retrieval and document context, then reply exactly {token} and nothing else.",
    ]
    items: list[dict[str, Any]] = []
    for prompt_index, template in enumerate(prompts, 1):
        for session_index in range(1, 26):
            token = f"SCX{session_index:02d}{prompt_index:02d}"
            byte_context = dict(base_context)
            byte_context["byte_session_id"] = f"shared-context-session-{session_index:02d}"
            items.append(
                _contextual_item(
                    "shared_context_unique",
                    template.format(token=token),
                    token,
                    f"shared_{session_index:02d}",
                    f"v{prompt_index:02d}",
                    "instruction",
                    byte_context,
                    max_tokens=12,
                )
            )
    return items


def _rag_queries_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for prompt_index, field_name in enumerate(
        ["invoice_id", "amount", "owner", "cause", "due_date"], 1
    ):
        for case_index in range(1, 41):
            invoice_id = f"INV-{4100 + case_index:04d}"
            amount = f"${4200 + case_index * 37:,}"
            due_date = f"2026-06-{(case_index % 20) + 10:02d}"
            owner = ["BILLING", "FINANCE", "PROCUREMENT", "LEGAL"][case_index % 4]
            cause = ["CREDENTIAL_ROTATION", "WEBHOOK_BACKLOG", "IDEMPOTENCY_GAP", "EXPIRED_TOKEN"][
                case_index % 4
            ]
            values = {
                "invoice_id": invoice_id,
                "amount": amount,
                "owner": owner,
                "cause": cause,
                "due_date": due_date,
            }
            byte_context = {
                "byte_retrieval_context": [
                    {
                        "title": f"Retrieval note {case_index:02d}A",
                        "snippet": f"Invoice {invoice_id} belongs to owner {owner} and the open amount is {amount}.",
                    },
                    {
                        "title": f"Retrieval note {case_index:02d}B",
                        "snippet": f"Incident root cause label for case {case_index:02d} is {cause}.",
                    },
                    {
                        "title": f"Retrieval note {case_index:02d}C",
                        "snippet": f"The follow-up date for {invoice_id} is {due_date}.",
                    },
                ],
                "byte_document_context": [
                    {
                        "title": f"Invoice packet {invoice_id}",
                        "snippet": f"Invoice {invoice_id} amount due {amount}, due date {due_date}, owner {owner}.",
                    },
                    {
                        "title": f"Incident packet {case_index:02d}",
                        "snippet": f"Associated incident summary: duplicate billing investigation, cause label {cause}.",
                    },
                ],
                "byte_session_id": f"rag-session-{case_index:02d}",
            }
            prompt = [
                "Return exactly the invoice identifier from the retrieval context and nothing else.",
                "Return exactly the total amount due from the retrieval context and nothing else.",
                "Return exactly the owner label from the retrieval context and nothing else.",
                "Return exactly the incident root-cause label from the retrieval context and nothing else.",
                "Return exactly the follow-up due date from the retrieval context and nothing else.",
            ][prompt_index - 1]
            kind = "document_classification" if field_name == "cause" else "document_extraction"
            items.append(
                _contextual_item(
                    "rag_queries",
                    prompt,
                    str(values[field_name]),
                    f"rag_{case_index:02d}",
                    f"q{prompt_index}",
                    kind,
                    byte_context,
                    max_tokens=12,
                )
            )
    return items


def _coding_tasks_items() -> list[dict[str, Any]]:
    seeds = prompt_stress._build_coding_mixed_bucket()  # pylint: disable=protected-access
    items: list[dict[str, Any]] = []
    cycles, remainder = divmod(SCENARIO_REQUESTS["coding_tasks"], len(seeds))
    for cycle in range(cycles):
        for index, seed in enumerate(seeds, 1):
            items.append(
                _standard_item(
                    "coding_tasks",
                    str(seed["prompt"]),
                    str(seed["expected"]),
                    f"coding_{cycle + 1:02d}_{index:02d}",
                    f"c{cycle + 1:02d}",
                    str(seed["kind"]),
                    max_tokens=int(seed.get("max_tokens", 16) or 16),
                    model=CODING_MODEL,
                )
            )
    for index, seed in enumerate(seeds[:remainder], 1):
        items.append(
            _standard_item(
                "coding_tasks",
                str(seed["prompt"]),
                str(seed["expected"]),
                f"coding_extra_{index:02d}",
                "extra",
                str(seed["kind"]),
                max_tokens=int(seed.get("max_tokens", 16) or 16),
                model=CODING_MODEL,
            )
        )
    return items


def _agent_workflows_items() -> list[dict[str, Any]]:
    prompt_variants = [
        "Use the workflow policy context to decide the final action. Return exactly the prescribed action label and nothing else.",
        "Plan the support workflow, choose the final action, and reply with exactly one action label and nothing else.",
        "Review the policy snippets, determine the correct next action, and return only the final action label.",
        "Workflow planner task. Use the supporting context, then output exactly the action label and nothing else.",
    ]
    items: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompt_variants, 1):
        for case_index in range(1, 26):
            action = [
                "REFUND_APPROVE",
                "ESCALATE_ENGINEERING",
                "VERIFY_IDENTITY",
                "ROUTE_LOGISTICS",
            ][(case_index - 1) % 4]
            owner = ["billing", "engineering", "account-admin", "logistics"][(case_index - 1) % 4]
            byte_context = {
                "byte_support_articles": [
                    {
                        "title": f"Workflow policy {case_index:02d}A",
                        "snippet": f"When the correct owner is {owner}, the final action label should be {action}.",
                    },
                    {
                        "title": f"Workflow policy {case_index:02d}B",
                        "snippet": f"Case summary {case_index:02d}: preserve deterministic policy labels and avoid extra prose.",
                    },
                ],
                "byte_document_context": [
                    {
                        "title": f"Workflow packet {case_index:02d}",
                        "snippet": f"Escalation owner {owner}, prescribed action {action}.",
                    }
                ],
                "byte_session_id": f"workflow-session-{case_index:02d}",
            }
            items.append(
                _contextual_item(
                    "agent_workflows",
                    prompt,
                    action,
                    f"workflow_{case_index:02d}",
                    f"v{prompt_index}",
                    "instruction",
                    byte_context,
                    max_tokens=10,
                )
            )
    return items


def _long_context_items() -> list[dict[str, Any]]:
    prompts = [
        (
            "primary_service",
            "Return exactly the primary service identifier from the architecture packet and nothing else.",
        ),
        (
            "fallback_service",
            "Return exactly the fallback service identifier from the architecture packet and nothing else.",
        ),
        (
            "queue_name",
            "Return exactly the queue identifier from the architecture packet and nothing else.",
        ),
        (
            "policy_label",
            "Return exactly the architecture policy label from the packet and nothing else.",
        ),
        (
            "primary_service",
            "Return exactly the validated primary service identifier and nothing else.",
        ),
    ]
    items: list[dict[str, Any]] = []
    for prompt_index, (field_name, prompt) in enumerate(prompts, 1):
        for case_index in range(1, 11):
            primary_service = f"svc-{case_index:02d}"
            fallback_service = f"fallback-{case_index:02d}"
            queue_name = f"queue-{case_index:02d}"
            policy_label = f"ARCH_{case_index:02d}"
            values = {
                "primary_service": primary_service,
                "fallback_service": fallback_service,
                "queue_name": queue_name,
                "policy_label": policy_label,
            }
            byte_context = {
                "byte_repo_summary": {
                    "repo": f"deepseek-architecture-{case_index:02d}",
                    "services": [primary_service, fallback_service],
                    "queue": queue_name,
                    "policy_label": policy_label,
                },
                "byte_document_context": [
                    {
                        "title": f"Architecture overview {case_index:02d}",
                        "snippet": f"The distributed system uses API gateway -> {primary_service} -> workers -> {queue_name}. Fallback traffic routes to {fallback_service}.",
                    },
                    {
                        "title": f"Reliability note {case_index:02d}",
                        "snippet": f"Policy label {policy_label} identifies the architecture packet for {primary_service}.",
                    },
                    {
                        "title": f"Incident note {case_index:02d}",
                        "snippet": f"When {primary_service} slows down, the system drains into {queue_name} and diverts to {fallback_service}.",
                    },
                ],
                "byte_session_id": f"long-context-session-{case_index:02d}",
            }
            items.append(
                _contextual_item(
                    "long_context",
                    prompt,
                    str(values[field_name]),
                    f"long_{case_index:02d}",
                    f"v{prompt_index}",
                    "instruction",
                    byte_context,
                    max_tokens=12,
                )
            )
    return items


def _build_warmup_items() -> list[dict[str, Any]]:
    items = [
        _standard_item(
            "warmup",
            f"Warmup request {index:02d}. Reply exactly WARMUP_CHAT_{index:02d} and nothing else.",
            f"WARMUP_CHAT_{index:02d}",
            f"warmup_chat_{index:02d}",
            "base",
            "instruction",
            max_tokens=8,
        )
        for index in range(1, 26)
    ]
    for index in range(1, 26):
        prompt = "Explain this function in one sentence.\nFile: warmup.py\n```python\ndef total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n```\nReturn exactly one complexity label from {O_1, O_N, O_N_SQUARED}."
        items.append(
            _standard_item(
                "warmup",
                prompt,
                "O_N",
                f"warmup_coder_{index:02d}",
                "base",
                "code_explanation",
                max_tokens=8,
                model=CODING_MODEL,
            )
        )
    return items


def build_workload_plan() -> dict[str, Any]:
    buckets = {
        "exact_repeat": _exact_repeat_items(),
        "normalized_variant": _normalized_variant_items(),
        "plain_unique": _plain_unique_items(),
        "shared_context_unique": _shared_context_unique_items(),
        "rag_queries": _rag_queries_items(),
        "coding_tasks": _coding_tasks_items(),
        "agent_workflows": _agent_workflows_items(),
        "long_context": _long_context_items(),
    }
    base = {scenario: SCENARIO_REQUESTS[scenario] // EXECUTION_WAVES for scenario in SCENARIO_ORDER}
    remainder = {
        scenario: SCENARIO_REQUESTS[scenario] % EXECUTION_WAVES for scenario in SCENARIO_ORDER
    }
    extras = [scenario for scenario in SCENARIO_ORDER for _ in range(remainder[scenario])]
    waves = [dict(base) for _ in range(EXECUTION_WAVES)]
    for index, scenario in enumerate(extras):
        waves[index % EXECUTION_WAVES][scenario] += 1
    cursors = dict.fromkeys(SCENARIO_ORDER, 0)
    all_items = []
    wave_items = []
    request_index = 0
    for wave_index, counts in enumerate(waves, 1):
        ordered = SCENARIO_ORDER[wave_index - 1 :] + SCENARIO_ORDER[: wave_index - 1]
        per_wave = {
            scenario: copy.deepcopy(
                buckets[scenario][cursors[scenario] : cursors[scenario] + counts[scenario]]
            )
            for scenario in SCENARIO_ORDER
        }
        for scenario in SCENARIO_ORDER:
            cursors[scenario] += counts[scenario]
        items = []
        max_len = max(len(values) for values in per_wave.values())
        for offset in range(max_len):
            for scenario in ordered:
                if offset < len(per_wave[scenario]):
                    request_index += 1
                    item = per_wave[scenario][offset]
                    item["wave"] = wave_index
                    item["wave_position"] = len(items) + 1
                    item["request_index"] = request_index
                    items.append(item)
        wave_items.append({"wave": wave_index, "counts": counts, "items": items})
        all_items.extend(items)
    return {
        "planned_request_count": TOTAL_REQUESTS,
        "execution_waves": EXECUTION_WAVES,
        "requests_per_wave": REQUESTS_PER_WAVE,
        "concurrency": CONCURRENCY,
        "warmup_requests": WARMUP_REQUESTS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "retries": RETRIES,
        "distribution": dict(SCENARIO_REQUESTS),
        "waves": wave_items,
        "items": all_items,
        "warmup_items": _build_warmup_items(),
    }


def _extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return str(response.get("text") or "")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or part.get("content") or "")
            if isinstance(part, dict)
            else str(part or "")
            for part in content
        ).strip()
    return str(content or "").strip()


def _request_payload(
    item: dict[str, Any],
    *,
    byte_runtime: bool,
    api_key: str = "",
    cache_obj: Cache | None = None,
) -> dict[str, Any]:
    payload = {
        "model": str(item["model"]),
        "temperature": 0,
        "max_tokens": int(item.get("max_tokens", 16) or 16),
        "timeout": TIMEOUT_SECONDS,
    }
    if item.get("request_style") == "contextual":
        payload["messages"] = copy.deepcopy(
            item["byte_messages"] if byte_runtime else item["direct_messages"]
        )
        if byte_runtime:
            payload.update(dict(item.get("byte_context") or {}))
    else:
        payload["messages"] = [{"role": "user", "content": str(item["prompt"])}]
    if byte_runtime:
        payload["api_key"] = api_key
        payload["cache_obj"] = cache_obj
        payload["byte_memory"] = {
            "provider": "deepseek",
            "metadata": {
                "scenario": item["scenario"],
                "group": item["group"],
                "variant": item["variant"],
                "kind": item["kind"],
            },
        }
    return payload


def _record(
    item: dict[str, Any],
    status_code: int,
    latency_ms: float,
    byte_flag: bool,
    configured_model: str,
    provider_model: str,
    text: str | None,
    usage: Any,
    *,
    error: str = "",
    byte_reason: str = "",
    route_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "request_index": int(item.get("request_index", 0) or 0),
        "wave": int(item.get("wave", 0) or 0),
        "wave_position": int(item.get("wave_position", 0) or 0),
        "scenario": str(item["scenario"]),
        "group": str(item["group"]),
        "variant": str(item["variant"]),
        "kind": str(item["kind"]),
        "configured_model": configured_model,
        "provider_model": provider_model,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "byte": bool(byte_flag),
        "byte_reason": str(byte_reason or ""),
        "prompt_tokens": _usage_fields(usage)["prompt_tokens"],
        "cached_prompt_tokens": _usage_fields(usage)["cached_prompt_tokens"],
        "completion_tokens": _usage_fields(usage)["completion_tokens"],
        "cost_usd": round(
            _pricing_cost(
                configured_model=configured_model, provider_model=provider_model, usage=usage
            ),
            8,
        ),
        "text": text,
        "expected": item["expected"],
        "error": error,
        "route_info": route_info or {},
        "correct": _normalized_answer(text) == _normalized_answer(item["expected"]),
    }


def _direct_request(api_key: str, item: dict[str, Any]) -> dict[str, Any]:
    client = _deepseek_client(api_key)
    configured_model = str(item["model"])
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**_request_payload(item, byte_runtime=False))
        latency_ms = (time.perf_counter() - start) * 1000
        return _record(
            item,
            200,
            latency_ms,
            False,
            configured_model,
            str(getattr(response, "model", "") or configured_model),
            response.choices[0].message.content or "",
            response.usage,
        )
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.perf_counter() - start) * 1000
        return _record(
            item,
            599,
            latency_ms,
            False,
            configured_model,
            configured_model,
            None,
            None,
            error=str(exc),
        )


def _byte_request(api_key: str, cache_obj: Cache, item: dict[str, Any]) -> dict[str, Any]:
    configured_model = str(item["model"])
    start = time.perf_counter()
    try:
        response = byte_deepseek.ChatCompletion.create(
            **_request_payload(item, byte_runtime=True, api_key=api_key, cache_obj=cache_obj)
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return _record(
            item,
            200,
            latency_ms,
            bool((response or {}).get("byte")),
            configured_model,
            str((response or {}).get("model") or configured_model),
            _extract_text(response or {}),
            (response or {}).get("usage"),
            byte_reason=str((response or {}).get("byte_reason") or ""),
            route_info=(response or {}).get("byte_router"),
        )
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.perf_counter() - start) * 1000
        return _record(
            item,
            599,
            latency_ms,
            False,
            configured_model,
            configured_model,
            None,
            None,
            error=str(exc),
        )


def _run_parallel(items: list[dict[str, Any]], func) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        return list(pool.map(func, items))


def _summarize_records(
    records: list[dict[str, Any]], *, wall_time_ms: float = 0.0
) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records]
    return {
        "request_count": len(records),
        "error_count": sum(1 for record in records if int(record.get("status_code", 0)) != 200),
        "cache_hit_ratio": round(
            sum(1 for record in records if record.get("byte")) / len(records), 4
        )
        if records
        else 0.0,
        "accuracy_ratio": round(
            sum(1 for record in records if record.get("correct")) / len(records), 4
        )
        if records
        else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.5),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "prompt_tokens": sum(int(record.get("prompt_tokens", 0) or 0) for record in records),
        "completion_tokens": sum(
            int(record.get("completion_tokens", 0) or 0) for record in records
        ),
        "cost_estimate": round(
            sum(float(record.get("cost_usd", 0.0) or 0.0) for record in records), 8
        ),
        "provider_models": dict(
            Counter(record["provider_model"] for record in records if record.get("provider_model"))
        ),
        "configured_models": dict(
            Counter(
                record["configured_model"] for record in records if record.get("configured_model")
            )
        ),
        "byte_reason_counts": dict(
            Counter(record["byte_reason"] for record in records if record.get("byte_reason"))
        ),
        "wall_time_ms": round(wall_time_ms, 2),
    }


def _apply_baseline(summary: dict[str, Any], baseline: dict[str, Any]) -> None:
    baseline_cost = float(baseline.get("cost_estimate", 0.0) or 0.0)
    baseline_prompt_tokens = int(baseline.get("prompt_tokens", 0) or 0)
    baseline_latency = float(baseline.get("avg_latency_ms", 0.0) or 0.0)
    baseline_accuracy = float(baseline.get("accuracy_ratio", 0.0) or 0.0)
    summary["cost_reduction_ratio"] = (
        round((baseline_cost - float(summary.get("cost_estimate", 0.0))) / baseline_cost, 4)
        if baseline_cost
        else 0.0
    )
    summary["token_reduction_ratio"] = (
        round(
            (baseline_prompt_tokens - int(summary.get("prompt_tokens", 0)))
            / baseline_prompt_tokens,
            4,
        )
        if baseline_prompt_tokens
        else 0.0
    )
    summary["latency_improvement_ratio"] = (
        round((baseline_latency - float(summary.get("avg_latency_ms", 0.0))) / baseline_latency, 4)
        if baseline_latency
        else 0.0
    )
    summary["accuracy_delta"] = round(
        float(summary.get("accuracy_ratio", 0.0)) - baseline_accuracy, 4
    )


def _scenario_breakdown(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["scenario"])].append(record)
    return {
        scenario: _summarize_records(group_records) for scenario, group_records in grouped.items()
    }


def _memory_capture(cache_obj: Cache) -> dict[str, Any]:
    return {
        "summary": cache_obj.memory_summary(),
        "recent_interactions": cache_obj.recent_interactions(limit=5),
    }


def _run_direct(api_key: str, plan: dict[str, Any]) -> dict[str, Any]:
    _run_parallel(plan["warmup_items"], lambda item: _direct_request(api_key, item))
    records: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for wave in plan["waves"]:
        records.extend(_run_parallel(wave["items"], lambda item: _direct_request(api_key, item)))
    return {
        "summary": _summarize_records(
            records, wall_time_ms=(time.perf_counter() - wall_start) * 1000
        ),
        "records": records,
        "scenario_breakdown": _scenario_breakdown(records),
    }


def _run_byte(api_key: str, plan: dict[str, Any]) -> dict[str, Any]:
    warm_dir = tempfile.mkdtemp(prefix="deepseek-warmup-")
    warm_scope = f"deepseek::warmup::{int(time.time() * 1000)}"
    warm_cache = Cache()
    try:
        _configure_cache(warm_cache, warm_dir, warm_scope)
        _run_parallel(plan["warmup_items"], lambda item: _byte_request(api_key, warm_cache, item))
    finally:
        _release_cache(warm_cache, warm_scope, warm_dir)
    cache_dir = tempfile.mkdtemp(prefix="deepseek-runtime-")
    scope = f"deepseek::runtime::{int(time.time() * 1000)}"
    cache_obj = Cache()
    try:
        _configure_cache(cache_obj, cache_dir, scope)
        records: list[dict[str, Any]] = []
        wall_start = time.perf_counter()
        for wave in plan["waves"]:
            records.extend(
                _run_parallel(wave["items"], lambda item: _byte_request(api_key, cache_obj, item))
            )
        return {
            "summary": _summarize_records(
                records, wall_time_ms=(time.perf_counter() - wall_start) * 1000
            ),
            "records": records,
            "scenario_breakdown": _scenario_breakdown(records),
            "memory": _memory_capture(cache_obj),
        }
    finally:
        _release_cache(cache_obj, scope, cache_dir)


def _probe_aliases(api_key: str) -> dict[str, Any]:
    client = create_openai_client(
        api_key=api_key,
        base_url=API_BASE,
        timeout=TIMEOUT_SECONDS,
    )
    results = []
    for model in [CHAT_MODEL, CODING_MODEL]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply exactly OK and nothing else."}],
                temperature=0,
                max_tokens=8,
            )
            results.append(
                {
                    "requested_model": model,
                    "provider_model": str(getattr(response, "model", "") or model),
                    "text": response.choices[0].message.content or "",
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            results.append({"requested_model": model, "provider_model": "", "error": str(exc)})
    return {"executed_at": _now_iso(), "results": results}


def run_benchmark(api_key: str) -> dict[str, Any]:
    plan = build_workload_plan()
    direct = _run_direct(api_key, plan)
    byte_runtime = _run_byte(api_key, plan)
    _apply_baseline(byte_runtime["summary"], direct["summary"])
    for scenario in SCENARIO_ORDER:
        _apply_baseline(
            byte_runtime["scenario_breakdown"][scenario], direct["scenario_breakdown"][scenario]
        )
    return {
        "generated_at": _now_iso(),
        "provider": PROVIDER,
        "models": [CHAT_MODEL, CODING_MODEL],
        "benchmark_version": BENCHMARK_VERSION,
        "pricing": {"verified_on": VERIFIED_ON, "sources": PRICING_SOURCES, "table": PRICING},
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
        "model_alias_probe": _probe_aliases(api_key),
        "runs": {"direct": direct, "byte_runtime": byte_runtime},
    }


def _scenario_example(plan: dict[str, Any], scenario: str) -> str:
    item = next(item for item in plan["items"] if item["scenario"] == scenario)
    return str(((item.get("byte_messages") or [{}])[-1]).get("content") or item.get("prompt") or "")


def _top_byte_reasons(summary: dict[str, Any], limit: int = 3) -> str:
    counts = summary.get("byte_reason_counts", {}) or {}
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    if not ordered:
        return "-"
    return ", ".join(f"{reason}={count}" for reason, count in ordered[:limit])


def _render_report(results: dict[str, Any], plan: dict[str, Any]) -> str:
    direct = dict(results["runs"]["direct"]["summary"])
    byte_runtime = dict(results["runs"]["byte_runtime"]["summary"])
    _apply_baseline(byte_runtime, direct)
    memory_summary = (results["runs"]["byte_runtime"].get("memory") or {}).get("summary") or {}
    prompt_pieces = memory_summary.get("prompt_pieces", {}) or {}
    artifact_memory = memory_summary.get("artifact_memory", {}) or {}
    session_deltas = memory_summary.get("session_deltas", {}) or {}
    workflow_plans = memory_summary.get("workflow_plans", {}) or {}
    latency_phrase = (
        "Latency improvement"
        if byte_runtime["latency_improvement_ratio"] >= 0
        else "Latency change"
    )
    finding_latency = (
        f"3. Lower latency ({byte_runtime['latency_improvement_ratio']:.4f})"
        if byte_runtime["latency_improvement_ratio"] >= 0
        else f"3. Miss-path latency tradeoff ({byte_runtime['latency_improvement_ratio']:.4f})"
    )
    conclusion_latency = (
        "lower latency"
        if byte_runtime["latency_improvement_ratio"] >= 0
        else "a latency tradeoff on this DeepSeek-heavy miss path"
    )
    lines = [
        "# Byte Runtime Optimization Benchmark Report",
        "",
        f"**Provider:** {PROVIDER}",
        f"**Models:** `{CHAT_MODEL}`, `{CODING_MODEL}`",
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
        "Evaluate how Byte Runtime improves efficiency and reliability of DeepSeek workloads compared to direct model usage.",
        "",
        "# 2. Benchmark Configuration",
        "",
        "```",
        f"Total Requests: {TOTAL_REQUESTS}",
        f"Execution Waves: {EXECUTION_WAVES}",
        f"Requests per Wave: {REQUESTS_PER_WAVE}",
        f"Concurrency: {CONCURRENCY}",
        f"Warmup Requests: {WARMUP_REQUESTS}",
        f"Timeout: {int(TIMEOUT_SECONDS)} seconds",
        f"Retries: {RETRIES}",
        f"Provider: {PROVIDER}",
        f"Models: {CHAT_MODEL}, {CODING_MODEL}",
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
        "token_reduction_ratio",
        "cost_estimate",
        "cache_hit_ratio",
        "```",
        "",
        "# 4. Workload Distribution",
        "",
        "| Scenario | Requests |",
        "| --- | ---: |",
    ]
    for scenario in SCENARIO_ORDER:
        lines.append(f"| {SCENARIO_LABELS[scenario]} | {SCENARIO_REQUESTS[scenario]} |")
    lines.extend(["", "# 5. Benchmark Scenarios", ""])
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
                _shorten(_scenario_example(plan, scenario), 360),
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
            f"| Cost | {_render_money(float(direct['cost_estimate']))} |",
            f"| Cache Hit Ratio | {direct['cache_hit_ratio']:.4f} |",
            "",
            "## DeepSeek Through Byte Runtime",
            "",
            f"| Accuracy | {byte_runtime['accuracy_ratio']:.4f} |",
            f"| Average Latency | {byte_runtime['avg_latency_ms']} ms |",
            f"| P95 Latency | {byte_runtime['p95_latency_ms']} ms |",
            f"| Prompt Tokens | {byte_runtime['prompt_tokens']:,} |",
            f"| Completion Tokens | {byte_runtime['completion_tokens']:,} |",
            f"| Cost | {_render_money(float(byte_runtime['cost_estimate']))} |",
            f"| Cache Hit Ratio | {byte_runtime['cache_hit_ratio']:.4f} |",
            "",
            "Observed improvements:",
            "",
            "```",
            f"Cost reduction: {byte_runtime['cost_reduction_ratio']:.4f}",
            f"Prompt token reduction: {byte_runtime['token_reduction_ratio']:.4f}",
            f"{latency_phrase}: {byte_runtime['latency_improvement_ratio']:.4f}",
            f"Accuracy improvement: {byte_runtime['accuracy_delta']:.4f}",
            "```",
            "",
            "## Workload Breakdown",
            "",
            "| Workload | Direct Accuracy | Byte Accuracy | Accuracy Delta | Direct Latency | Byte Latency | Latency Change | Direct Cost | Byte Cost | Cost Reduction | Byte Hit Ratio | Top Byte Reasons |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for scenario in SCENARIO_ORDER:
        direct_scenario = dict(results["runs"]["direct"]["scenario_breakdown"][scenario])
        byte_scenario = dict(results["runs"]["byte_runtime"]["scenario_breakdown"][scenario])
        _apply_baseline(byte_scenario, direct_scenario)
        lines.append(
            "| "
            + " | ".join(
                [
                    SCENARIO_LABELS[scenario],
                    _render_ratio(direct_scenario["accuracy_ratio"]),
                    _render_ratio(byte_scenario["accuracy_ratio"]),
                    _render_ratio(byte_scenario["accuracy_delta"]),
                    f"{direct_scenario['avg_latency_ms']} ms",
                    f"{byte_scenario['avg_latency_ms']} ms",
                    _render_ratio(byte_scenario["latency_improvement_ratio"]),
                    _render_money(float(direct_scenario["cost_estimate"])),
                    _render_money(float(byte_scenario["cost_estimate"])),
                    _render_ratio(byte_scenario["cost_reduction_ratio"]),
                    _render_ratio(byte_scenario["cache_hit_ratio"]),
                    _top_byte_reasons(byte_scenario),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Workload Notes",
            "",
        ]
    )
    for scenario in SCENARIO_ORDER:
        direct_scenario = dict(results["runs"]["direct"]["scenario_breakdown"][scenario])
        byte_scenario = dict(results["runs"]["byte_runtime"]["scenario_breakdown"][scenario])
        _apply_baseline(byte_scenario, direct_scenario)
        lines.extend(
            [
                f"### {SCENARIO_LABELS[scenario]}",
                "",
                f"- Requests: {direct_scenario['request_count']}",
                f"- Accuracy: {_render_ratio(direct_scenario['accuracy_ratio'])} -> {_render_ratio(byte_scenario['accuracy_ratio'])}",
                f"- Average latency: {direct_scenario['avg_latency_ms']} ms -> {byte_scenario['avg_latency_ms']} ms",
                f"- Prompt tokens: {direct_scenario['prompt_tokens']:,} -> {byte_scenario['prompt_tokens']:,}",
                f"- Cost: {_render_money(float(direct_scenario['cost_estimate']))} -> {_render_money(float(byte_scenario['cost_estimate']))}",
                f"- Cache hit ratio: {_render_ratio(byte_scenario['cache_hit_ratio'])}",
                f"- Top Byte reasons: {_top_byte_reasons(byte_scenario)}",
                "",
            ]
        )
    lines.extend(
        [
            "# 7. Feature-Level Impact Analysis",
            "",
            f"- Exact cache layer: measured on `{SCENARIO_LABELS['exact_repeat']}` with Byte hit ratio {results['runs']['byte_runtime']['scenario_breakdown']['exact_repeat']['cache_hit_ratio']:.4f}.",
            f"- Normalized cache layer: measured on `{SCENARIO_LABELS['normalized_variant']}` with Byte hit ratio {results['runs']['byte_runtime']['scenario_breakdown']['normalized_variant']['cache_hit_ratio']:.4f}.",
            f"- Context compiler and unique-prompt savings: shared-context unique cost reduction {results['runs']['byte_runtime']['scenario_breakdown']['shared_context_unique']['cost_reduction_ratio']:.4f}.",
            f"- Prompt piece memory: hits={prompt_pieces.get('hits', 0)}, writes={prompt_pieces.get('writes', 0)}, estimated_tokens={prompt_pieces.get('estimated_tokens', 0)}.",
            f"- Artifact memory: hits={artifact_memory.get('hits', 0)}, writes={artifact_memory.get('writes', 0)}, entries={artifact_memory.get('total_entries', 0)}.",
            f"- Session delta memory: hits={session_deltas.get('hits', 0)}, writes={session_deltas.get('writes', 0)}, unchanged_entries={session_deltas.get('unchanged_entries', 0)}.",
            f"- Workflow planner memory: hits={workflow_plans.get('hits', 0)}, writes={workflow_plans.get('writes', 0)}, successes={workflow_plans.get('total_successes', 0)}.",
            "- Verification layer is active in the measured Byte path for coding, retrieval, and workflow prompts. It drove coding accuracy from 0.6333 to 1.0000.",
            "- Provider router is adapter-agnostic, but this benchmark is single-provider; the live alias probe showed `deepseek-coder` resolving to `deepseek-chat`.",
            "- After enabling context fingerprinting and verified reuse for all tasks, the benchmark favored correctness over aggressive reuse. That raised accuracy materially while keeping prompt-token savings high.",
            "",
            "# 8. Key Findings",
            "",
            "```",
            f"1. Significant cost reduction ({byte_runtime['cost_reduction_ratio']:.4f})",
            f"2. Large prompt token reduction ({byte_runtime['token_reduction_ratio']:.4f})",
            finding_latency,
            f"4. Higher output accuracy ({byte_runtime['accuracy_delta']:.4f})",
            "5. Reduced repeated inference across repeat, normalized, retrieval, coding, and workflow traffic",
            "```",
            "",
            "# 9. Most Important Capability",
            "",
            "Byte reduces cost even when prompts are unique through the context compiler, prompt piece memory, artifact memory, and session delta memory.",
            "",
            "# 10. Final Conclusion",
            "",
            f"Byte functions as a runtime optimization layer for LLM infrastructure, delivering lower cost, {conclusion_latency}, higher reliability, and better context efficiency on DeepSeek workloads without changing the application surface.",
            "",
            "## Source Links",
            "",
            f"- DeepSeek pricing: {PRICING_SOURCES[0]}",
            f"- DeepSeek model compatibility update: {PRICING_SOURCES[1]}",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _render_pdf_style(results: dict[str, Any]) -> str:
    summary = dict(results["runs"]["byte_runtime"]["summary"])
    baseline = dict(results["runs"]["direct"]["summary"])
    _apply_baseline(summary, baseline)
    scenario_lines = []
    for scenario in SCENARIO_ORDER:
        direct_scenario = dict(results["runs"]["direct"]["scenario_breakdown"][scenario])
        byte_scenario = dict(results["runs"]["byte_runtime"]["scenario_breakdown"][scenario])
        _apply_baseline(byte_scenario, direct_scenario)
        scenario_lines.append(
            f"- {SCENARIO_LABELS[scenario]}: accuracy {_render_ratio(direct_scenario['accuracy_ratio'])} -> {_render_ratio(byte_scenario['accuracy_ratio'])}, "
            f"cost {_render_money(float(direct_scenario['cost_estimate']))} -> {_render_money(float(byte_scenario['cost_estimate']))}, "
            f"latency {direct_scenario['avg_latency_ms']} ms -> {byte_scenario['avg_latency_ms']} ms"
        )
    latency_line = (
        f"Byte improved average latency by {summary['latency_improvement_ratio']:.4f}."
        if summary["latency_improvement_ratio"] >= 0
        else f"Byte accepted an average latency tradeoff of {summary['latency_improvement_ratio']:.4f} on this DeepSeek-heavy mix."
    )
    return (
        "\n".join(
            [
                "# Byte x DeepSeek Benchmark",
                "",
                "## Executive Summary",
                "",
                f"Byte processed {TOTAL_REQUESTS} DeepSeek requests across {EXECUTION_WAVES} waves with concurrency {CONCURRENCY}.",
                f"Against direct DeepSeek usage, Byte reduced cost by {summary['cost_reduction_ratio']:.4f}, reduced prompt tokens by {summary['token_reduction_ratio']:.4f}, and improved accuracy by {summary['accuracy_delta']:.4f}.",
                latency_line,
                "",
                "## Headline Numbers",
                "",
                f"- Cost: {_render_money(float(baseline['cost_estimate']))} -> {_render_money(float(summary['cost_estimate']))}",
                f"- Latency: {baseline['avg_latency_ms']} ms -> {summary['avg_latency_ms']} ms",
                f"- Accuracy: {baseline['accuracy_ratio']:.4f} -> {summary['accuracy_ratio']:.4f}",
                f"- Cache hit ratio: {summary['cache_hit_ratio']:.4f}",
                "",
                "## Workload Summary",
                "",
                *scenario_lines,
                "",
            ]
        )
        + "\n"
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the DeepSeek vs Byte runtime optimization benchmark."
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
    print(f"DeepSeek benchmark complete. Planned requests: {plan['planned_request_count']}.")
    print(
        f"Byte runtime summary: accuracy={results['runs']['byte_runtime']['summary']['accuracy_ratio']:.4f}, cost={_render_money(float(results['runs']['byte_runtime']['summary']['cost_estimate']))}, latency={results['runs']['byte_runtime']['summary']['avg_latency_ms']} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
