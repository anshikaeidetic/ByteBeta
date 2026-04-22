import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from byte import Cache, Config  # pylint: disable=wrong-import-position
from byte._backends import openai as byte_openai  # pylint: disable=wrong-import-position
from byte.adapter.api import init_cache  # pylint: disable=wrong-import-position
from byte.benchmarking._optional_runtime import create_openai_client
from byte.benchmarking._program_common import call_with_retry
from byte.benchmarking.programs import deep_openai_coding_benchmark as coding
from byte.processor.pre import (  # pylint: disable=wrong-import-position
    last_content,
    normalized_last_content,
)
from byte.processor.shared_memory import (
    clear_shared_memory,  # pylint: disable=wrong-import-position
)

REPORT_DIR = REPO_ROOT / "docs" / "reports"
DEFAULT_CODING_REPORT = REPORT_DIR / "openai_cursor_coding_benchmark.json"
DEFAULT_SURFACE_REPORT = REPORT_DIR / "openai_surface_benchmark.json"
DEFAULT_REPORT = REPORT_DIR / "openai_comprehensive_workload_benchmark.md"
DEFAULT_JSON_REPORT = REPORT_DIR / "openai_comprehensive_workload_benchmark.json"

CHAT_MODEL = coding.CHEAP_MODEL

_COMMON_SYSTEM = (
    "You are Byte's evaluation assistant. Use the supplied context, stay deterministic, "
    "and follow the final exact-output instruction exactly."
)


def _run_script(script_name: str, env: dict[str, str], *args: str) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script_name), *args]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return
    stderr = (result.stderr or "").strip().splitlines()
    stdout = (result.stdout or "").strip().splitlines()
    tail = "\n".join((stderr or stdout)[-20:])
    raise RuntimeError(f"{script_name} failed with exit code {result.returncode}\n{tail}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _base_config(scope: str, *, enable_token_counter: bool = False) -> Config:
    return Config(
        enable_token_counter=enable_token_counter,
        tiered_cache=True,
        tier1_max_size=512,
        tier1_promote_on_write=True,
        async_write_back=True,
        memory_scope=scope,
        intent_memory=True,
        execution_memory=True,
        verified_reuse_for_coding=True,
        verified_reuse_for_all=False,
        delta_generation=True,
        context_compiler=True,
        context_compiler_keep_last_messages=4,
        context_compiler_max_chars=4800,
        context_compiler_focus_distillation=True,
        context_compiler_total_aux_budget_ratio=0.58,
        context_compiler_cross_note_dedupe=True,
        ambiguity_detection=True,
        planner_enabled=True,
        failure_memory=True,
        tenant_policy_learning=True,
        native_prompt_caching=True,
        native_prompt_cache_min_chars=600,
        adaptive_threshold=True,
        cache_admission_min_score=0.1,
        semantic_allowed_categories=[
            "question_answer",
            "summarization",
            "comparison",
            "instruction",
            "classification",
            "extraction",
            "documentation",
            "code_explanation",
        ],
        task_policies={"*": {"native_prompt_cache": True}},
    )


def _configure_cache(
    cache_obj: Cache,
    cache_dir: str,
    mode: str,
    *,
    scope: str,
    enable_token_counter: bool = False,
) -> None:
    config = _base_config(scope, enable_token_counter=enable_token_counter)
    init_cache(
        mode=mode,
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


def _repo_summary_payload() -> dict[str, Any]:
    return {
        "repo": "byte-enterprise-app",
        "workspace": "monorepo",
        "language": "python",
        "framework": "fastapi",
        "files": [f"services/api/module_{i:02d}.py" for i in range(30)]
        + [f"frontend/src/components/panel_{i:02d}.tsx" for i in range(18)],
        "symbols": [f"handler_{i:02d}" for i in range(18)]
        + [f"service_{i:02d}" for i in range(12)],
    }


def _changed_files_payload() -> list[dict[str, Any]]:
    return [
        {
            "path": f"services/api/feature_{i:02d}.py",
            "summary": "Touches validation, cache invalidation, and billing edge-case handling.",
        }
        for i in range(12)
    ]


def _changed_hunks_payload() -> list[dict[str, Any]]:
    hunks = []
    for i in range(8):
        hunks.append(
            {
                "path": f"services/api/feature_{i:02d}.py",
                "hunk": (
                    "@@ -10,8 +10,14 @@\n"
                    " def handle_request(user_id, payload):\n"
                    "     normalized = payload.strip()\n"
                    f"     if not normalized or len(normalized) < {i + 2}:\n"
                    "         return {'status': 'invalid'}\n"
                    "     cache_key = f'user:{user_id}:billing'\n"
                    "     result = process(normalized)\n"
                    "     return {'status': 'ok', 'result': result}\n"
                ),
            }
        )
    return hunks


def _retrieval_context_payload() -> list[dict[str, Any]]:
    return [
        {
            "title": "Billing escalation runbook",
            "snippet": "Duplicate subscription charges route to billing operations for refund review and ledger correction.",
        },
        {
            "title": "Incident summary",
            "snippet": "Export errors are commonly caused by stale session state after rollout and usually need a retry-safe patch.",
        },
        {
            "title": "Support FAQ",
            "snippet": "Shipping delay complaints map to logistics, while payment retries and double charges map to billing.",
        },
        {
            "title": "Docs style guide",
            "snippet": "Generated docs should stay concise, deterministic, and reference the owning subsystem if relevant.",
        },
    ]


def _document_context_payload() -> list[dict[str, Any]]:
    return [
        {
            "title": "Contract clause 4.2",
            "snippet": "Auto-renewal requires 30 days written notice before term end to stop renewal.",
        },
        {
            "title": "Invoice INV-2048",
            "snippet": "Customer Northwind Labs, amount due $4,250, due date 2026-04-18.",
        },
        {
            "title": "Release note draft",
            "snippet": "CSV export launched, billing duplication bug fixed, analytics generation latency reduced by 35 percent.",
        },
    ]


def _support_articles_payload() -> list[dict[str, Any]]:
    return [
        {
            "title": "Billing article",
            "snippet": "Duplicate renewal charges should be escalated to billing and marked refund-review.",
        },
        {
            "title": "Technical article",
            "snippet": "Crashes on export are technical issues and usually need logs plus version details.",
        },
        {
            "title": "Shipping article",
            "snippet": "Transit delays beyond five business days go to logistics support.",
        },
    ]


def _build_unique_context_items() -> list[dict[str, Any]]:
    repo_summary = _repo_summary_payload()
    changed_files = _changed_files_payload()
    changed_hunks = _changed_hunks_payload()
    retrieval_context = _retrieval_context_payload()
    document_context = _document_context_payload()
    support_articles = _support_articles_payload()

    items = []
    prompt_specs = [
        (
            "UNIQUE_SUPPORT_OK",
            "support_unique",
            "Review the shared support and repo context, then reply exactly UNIQUE_SUPPORT_OK and nothing else.",
        ),
        (
            "UNIQUE_DOC_OK",
            "document_unique",
            "Review the shared document and repo context, then reply exactly UNIQUE_DOC_OK and nothing else.",
        ),
        (
            "UNIQUE_PATCH_OK",
            "coding_unique",
            "Review the shared code and retrieval context, then reply exactly UNIQUE_PATCH_OK and nothing else.",
        ),
        (
            "UNIQUE_SUMMARY_OK",
            "summary_unique",
            "Review the shared workspace context, then reply exactly UNIQUE_SUMMARY_OK and nothing else.",
        ),
    ]
    for index, (token, group, short_prompt) in enumerate(prompt_specs, 1):
        full_prompt = (
            f"{short_prompt}\n\n"
            f"Repo summary:\n{json.dumps(repo_summary, indent=2)}\n\n"
            f"Changed files:\n{json.dumps(changed_files, indent=2)}\n\n"
            f"Changed hunks:\n{json.dumps(changed_hunks, indent=2)}\n\n"
            f"Retrieval context:\n{json.dumps(retrieval_context, indent=2)}\n\n"
            f"Document context:\n{json.dumps(document_context, indent=2)}\n\n"
            f"Support articles:\n{json.dumps(support_articles, indent=2)}"
        )
        items.append(
            {
                "expected": token,
                "group": group,
                "variant": f"v{index}",
                "kind": "instruction",
                "direct_messages": [
                    {"role": "system", "content": _COMMON_SYSTEM},
                    {"role": "user", "content": full_prompt},
                ],
                "byte_messages": [
                    {"role": "system", "content": _COMMON_SYSTEM},
                    {"role": "user", "content": short_prompt},
                ],
                "byte_context": {
                    "byte_repo_summary": repo_summary,
                    "byte_changed_files": changed_files,
                    "byte_changed_hunks": changed_hunks,
                    "byte_retrieval_context": retrieval_context,
                    "byte_document_context": document_context,
                    "byte_support_articles": support_articles,
                    "byte_session_id": "unique-context-session",
                },
            }
        )
    return items


def _extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content or "")
    return str(response.get("text") or "")


def _direct_context_request(api_key: str, item: dict[str, Any]) -> dict[str, Any]:
    client = create_openai_client(api_key=api_key)
    payload = {
        "model": CHAT_MODEL,
        "messages": item["direct_messages"],
        "temperature": 0,
        "max_tokens": 8,
    }
    start = time.perf_counter()
    try:
        response = call_with_retry(lambda: client.chat.completions.create(**payload))
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = byte_openai._openai_response_to_dict(response)  # pylint: disable=protected-access
        return coding._response_record(  # pylint: disable=protected-access
            status_code=200,
            latency_ms=latency_ms,
            byte_flag=False,
            model_name=str(response_dict.get("model") or CHAT_MODEL),
            route_info=None,
            text=_extract_text(response_dict),
            usage=response_dict.get("usage"),
            item=item,
        )
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.perf_counter() - start) * 1000
        return coding._response_record(  # pylint: disable=protected-access
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


def _byte_context_request(api_key: str, cache_obj: Cache, item: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": CHAT_MODEL,
        "messages": item["byte_messages"],
        "temperature": 0,
        "max_tokens": 8,
        "api_key": api_key,
        "cache_obj": cache_obj,
        **dict(item.get("byte_context") or {}),
        "byte_memory": {
            "provider": "openai",
            "metadata": {
                "scenario": "unique_context_compiler",
                "group": item["group"],
                "variant": item["variant"],
            },
        },
    }
    start = time.perf_counter()
    try:
        response = call_with_retry(lambda: byte_openai.ChatCompletion.create(**payload))
        latency_ms = (time.perf_counter() - start) * 1000
        return coding._response_record(  # pylint: disable=protected-access
            status_code=200,
            latency_ms=latency_ms,
            byte_flag=bool(response.get("byte")),
            model_name=str(response.get("model") or CHAT_MODEL),
            route_info=response.get("byte_router"),
            text=_extract_text(response),
            usage=response.get("usage"),
            item=item,
        )
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.perf_counter() - start) * 1000
        return coding._response_record(  # pylint: disable=protected-access
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


def _run_unique_context_sequence(api_key: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    direct = coding._summarize_records([_direct_context_request(api_key, item) for item in items])  # pylint: disable=protected-access
    cache_dir = tempfile.mkdtemp(prefix="byte-unique-context-")
    scope = f"comprehensive::unique_context::{int(time.time() * 1000)}"
    clear_shared_memory(scope)
    cache_obj = Cache()
    try:
        _configure_cache(cache_obj, cache_dir, "hybrid", scope=scope)
        byte_records = [_byte_context_request(api_key, cache_obj, item) for item in items]
        byte_summary = coding._summarize_records(byte_records)  # pylint: disable=protected-access
        byte_summary["memory"] = {
            "summary": cache_obj.memory_summary(),
            "recent_interactions": cache_obj.recent_interactions(limit=4),
        }
    finally:
        _release_cache(cache_obj, scope, cache_dir)

    baseline_cost = direct["total_cost_usd"]
    baseline_latency = direct["avg_latency_ms"]
    byte_summary["saved_vs_baseline_usd"] = round(baseline_cost - byte_summary["total_cost_usd"], 8)
    byte_summary["savings_ratio"] = (
        round((baseline_cost - byte_summary["total_cost_usd"]) / baseline_cost, 4)
        if baseline_cost
        else 0.0
    )
    byte_summary["latency_delta_ms"] = round(byte_summary["avg_latency_ms"] - baseline_latency, 2)
    return {
        "name": "unique_context_compiler_session_4",
        "description": (
            "Four unique prompts in one session with large repeated repo, retrieval, document, and support context. "
            "ByteAI Cache uses the context compiler, prompt-piece memory, artifact memory, and session-delta memory to reduce "
            "prompt tokens even when the answer cache does not hit."
        ),
        "request_count": len(items),
        "runs": {
            "direct": direct,
            "byte_hybrid": byte_summary,
        },
    }


def _build_streaming_items() -> list[dict[str, Any]]:
    return [
        {
            "prompt": "Reply with exactly STREAM_ALPHA and nothing else.",
            "expected": "STREAM_ALPHA",
            "group": "stream_alpha",
            "variant": "v1",
            "kind": "stream_chat",
        },
        {
            "prompt": "Reply with exactly STREAM_ALPHA and nothing else.",
            "expected": "STREAM_ALPHA",
            "group": "stream_alpha",
            "variant": "v2",
            "kind": "stream_chat",
        },
        {
            "prompt": "Keep the answer to STREAM_BETA. Reply with exactly STREAM_BETA and nothing else.",
            "expected": "STREAM_BETA",
            "group": "stream_beta",
            "variant": "v1",
            "kind": "stream_chat",
        },
        {
            "prompt": "Byte benchmark request. Reply with exactly STREAM_BETA and nothing else. Keep the answer to STREAM_BETA.",
            "expected": "STREAM_BETA",
            "group": "stream_beta",
            "variant": "v2",
            "kind": "stream_chat",
        },
    ]


def _stream_usage_from_chunks(chunks: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    usage_payload: dict[str, Any] = {}
    byte_flag = False
    saved_token = None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        byte_flag = byte_flag or bool(chunk.get("byte"))
        if chunk.get("usage") is not None:
            usage_payload = chunk.get("usage") or {}
        if chunk.get("saved_token") is not None:
            saved_token = chunk.get("saved_token")
    if saved_token is not None and byte_flag and not usage_payload:
        usage_payload = {
            "prompt_tokens": int((saved_token or [0, 0])[0] or 0),
            "completion_tokens": int((saved_token or [0, 0])[1] or 0),
            "prompt_tokens_details": {
                "cached_tokens": int((saved_token or [0, 0])[0] or 0),
            },
        }
    return usage_payload, byte_flag


def _stream_text_from_chunks(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if text:
                            parts.append(str(text))
            elif content:
                parts.append(str(content))
            elif delta.get("text"):
                parts.append(str(delta.get("text")))
    return "".join(parts).strip()


def _stream_record(
    *,
    item: dict[str, Any],
    latency_ms: float,
    model_name: str,
    usage: dict[str, Any],
    text: str | None,
    byte_flag: bool,
    error: str = "",
) -> dict[str, Any]:
    fields = coding._usage_fields(usage)  # pylint: disable=protected-access
    if byte_flag and fields["cached_prompt_tokens"] >= fields["prompt_tokens"] > 0:
        cost_usd = 0.0
    else:
        cost_usd = round(coding._pricing_cost(model_name, usage), 8) if usage else 0.0  # pylint: disable=protected-access
    return {
        "status_code": 200 if not error else 599,
        "latency_ms": round(latency_ms, 2),
        "byte": byte_flag,
        "prompt_tokens": fields["prompt_tokens"],
        "cached_prompt_tokens": fields["cached_prompt_tokens"],
        "completion_tokens": fields["completion_tokens"],
        "cost_usd": cost_usd,
        "text": text,
        "expected": item["expected"],
        "group": item["group"],
        "variant": item["variant"],
        "kind": item["kind"],
        "model": model_name,
        "route_info": {},
        "error": error,
        "correct": coding._normalized_answer(text) == coding._normalized_answer(item["expected"]),  # pylint: disable=protected-access
    }


def _direct_stream_request(api_key: str, item: dict[str, Any]) -> dict[str, Any]:
    client = create_openai_client(api_key=api_key)
    start = time.perf_counter()
    try:
        stream = call_with_retry(
            lambda: client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": _COMMON_SYSTEM},
                    {"role": "user", "content": item["prompt"]},
                ],
                temperature=0,
                max_tokens=8,
                stream=True,
                stream_options={"include_usage": True},
            )
        )
        chunks = [byte_openai._openai_response_to_dict(chunk) for chunk in stream]  # pylint: disable=protected-access
        latency_ms = (time.perf_counter() - start) * 1000
        usage, _ = _stream_usage_from_chunks(chunks)
        return _stream_record(
            item=item,
            latency_ms=latency_ms,
            model_name=CHAT_MODEL,
            usage=usage,
            text=_stream_text_from_chunks(chunks),
            byte_flag=False,
        )
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.perf_counter() - start) * 1000
        return _stream_record(
            item=item,
            latency_ms=latency_ms,
            model_name=CHAT_MODEL,
            usage={},
            text=None,
            byte_flag=False,
            error=str(exc),
        )


def _byte_stream_request(api_key: str, cache_obj: Cache, item: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        stream = call_with_retry(
            lambda: byte_openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": _COMMON_SYSTEM},
                    {"role": "user", "content": item["prompt"]},
                ],
                temperature=0,
                max_tokens=8,
                stream=True,
                stream_options={"include_usage": True},
                api_key=api_key,
                cache_obj=cache_obj,
                byte_memory={
                    "provider": "openai",
                    "metadata": {
                        "scenario": "streaming_memory",
                        "group": item["group"],
                        "variant": item["variant"],
                    },
                },
            )
        )
        chunks = list(stream)
        latency_ms = (time.perf_counter() - start) * 1000
        usage, byte_flag = _stream_usage_from_chunks(chunks)
        return _stream_record(
            item=item,
            latency_ms=latency_ms,
            model_name=CHAT_MODEL,
            usage=usage,
            text=_stream_text_from_chunks(chunks),
            byte_flag=byte_flag,
        )
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.perf_counter() - start) * 1000
        return _stream_record(
            item=item,
            latency_ms=latency_ms,
            model_name=CHAT_MODEL,
            usage={},
            text=None,
            byte_flag=False,
            error=str(exc),
        )


def _run_stream_sequence(api_key: str, items: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"byte-stream-{mode}-")
    scope = f"comprehensive::stream::{mode}::{int(time.time() * 1000)}"
    clear_shared_memory(scope)
    cache_obj = Cache()
    try:
        _configure_cache(cache_obj, cache_dir, mode, scope=scope, enable_token_counter=True)
        records = [_byte_stream_request(api_key, cache_obj, item) for item in items]
        summary = coding._summarize_records(records)  # pylint: disable=protected-access
        summary["memory"] = {
            "summary": cache_obj.memory_summary(),
            "recent_interactions": cache_obj.recent_interactions(limit=4),
        }
        return summary
    finally:
        _release_cache(cache_obj, scope, cache_dir)


def _run_streaming_benchmark(api_key: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    direct = coding._summarize_records([_direct_stream_request(api_key, item) for item in items])  # pylint: disable=protected-access
    runs = {
        "direct": direct,
        "byte_exact": _run_stream_sequence(api_key, items, "exact"),
        "byte_normalized": _run_stream_sequence(api_key, items, "normalized"),
    }
    baseline_cost = direct["total_cost_usd"]
    baseline_latency = direct["avg_latency_ms"]
    for key, data in runs.items():
        if key == "direct":
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
    return {
        "name": "streaming_chat_reuse_4",
        "description": (
            "Streaming chat completions with one exact-duplicate pair and one canonical exact-answer pair. "
            "This exercises stream memory recording and verified reuse for cached stream responses."
        ),
        "request_count": len(items),
        "runs": runs,
    }


def _coding_headlines(coding_report: dict[str, Any]) -> list[str]:
    if not coding_report:
        return ["- Coding report was unavailable."]
    lines = []
    for scenario in (coding_report.get("sequential_scenarios") or [])[:3]:
        hybrid = (scenario.get("runs") or {}).get("hybrid", {})
        lines.append(
            f"- {scenario.get('name')}: hybrid savings_ratio={hybrid.get('savings_ratio')}, "
            f"hybrid hit_ratio={hybrid.get('hit_ratio')}, hybrid avg_latency={hybrid.get('avg_latency_ms')} ms"
        )
    routing = (coding_report.get("routing_blend") or {}).get("runs", {})
    if routing:
        lines.append(
            f"- routing_blend: byte_hybrid_routed savings_ratio={routing.get('byte_hybrid_routed', {}).get('savings_ratio')}, "
            f"route_tiers={routing.get('byte_hybrid_routed', {}).get('route_tiers', {})}"
        )
    return lines


def _surface_headlines(surface_report: dict[str, Any]) -> list[str]:
    if not surface_report:
        return ["- Surface report was unavailable."]
    lines = []
    for scenario in surface_report.get("text_scenarios") or []:
        hybrid = (scenario.get("runs") or {}).get("hybrid", {})
        lines.append(
            f"- {scenario.get('name')}: hybrid savings_ratio={hybrid.get('savings_ratio')}, "
            f"hit_ratio={hybrid.get('hit_ratio')}, accuracy={hybrid.get('accuracy_ratio')}"
        )
    for key in ("moderation", "image_generation", "speech_generation", "transcription"):
        section = surface_report.get(key) or {}
        if section.get("unavailable"):
            lines.append(f"- {key}: unavailable ({section.get('unavailable')})")
            continue
        runs = section.get("runs") or {}
        normalized = runs.get("normalized") or runs.get("exact") or {}
        lines.append(
            f"- {key}: avoided_call_ratio={normalized.get('avoided_call_ratio')}, "
            f"avg_latency={normalized.get('avg_latency_ms')} ms"
        )
    return lines


def _render_mode_line(name: str, data: dict[str, Any], *, baseline_label: str = "direct") -> str:
    return coding._render_mode_line(name, data, baseline_label=baseline_label)  # pylint: disable=protected-access


def _render_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# ByteAI Cache OpenAI Comprehensive Workload Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {results['generated_at']}")
    lines.append(f"Chat model: `{results['chat_model']}`")
    lines.append("")
    lines.append(
        "This report runs the existing live coding and surface benchmarks, then adds two deeper scenarios:"
    )
    lines.append(
        "- unique-prompt miss-path savings via the context compiler, prompt-piece memory, artifact memory, and session-delta memory"
    )
    lines.append("- streaming memory recording and cached stream reuse")
    lines.append("")
    lines.append("## Source Artifacts")
    lines.append("")
    lines.append(f"- Coding JSON: `{results['artifacts']['coding_json']}`")
    lines.append(f"- Surface JSON: `{results['artifacts']['surface_json']}`")
    lines.append(f"- Comprehensive JSON: `{results['artifacts']['comprehensive_json']}`")
    lines.append("")
    lines.append("## Coding Highlights")
    lines.append("")
    lines.extend(results["coding_headlines"])
    lines.append("")
    lines.append("## Surface Highlights")
    lines.append("")
    lines.extend(results["surface_headlines"])
    lines.append("")
    lines.append("## Unique Prompt Savings")
    lines.append("")
    unique_context = results["unique_context"]
    lines.append(f"- {unique_context['description']}")
    lines.append(_render_mode_line("Direct", unique_context["runs"]["direct"]))
    lines.append(_render_mode_line("ByteAI Cache hybrid", unique_context["runs"]["byte_hybrid"]))
    memory_summary = (unique_context["runs"]["byte_hybrid"].get("memory") or {}).get(
        "summary"
    ) or {}
    lines.append(
        f"- ByteAI Cache memory summary: prompt_pieces={memory_summary.get('prompt_pieces', {})}, "
        f"artifact_memory={memory_summary.get('artifact_memory', {})}, session_deltas={memory_summary.get('session_deltas', {})}"
    )
    lines.append("")
    lines.append("## Streaming")
    lines.append("")
    streaming = results["streaming"]
    lines.append(f"- {streaming['description']}")
    lines.append(_render_mode_line("Direct", streaming["runs"]["direct"]))
    lines.append(_render_mode_line("ByteAI Cache exact", streaming["runs"]["byte_exact"]))
    lines.append(_render_mode_line("ByteAI Cache normalized", streaming["runs"]["byte_normalized"]))
    stream_memory = (streaming["runs"]["byte_normalized"].get("memory") or {}).get("summary") or {}
    lines.append(
        f"- Streaming memory summary: ai_memory={stream_memory.get('ai_memory', {})}, "
        f"execution_memory={stream_memory.get('execution_memory', {})}, workflow_plans={stream_memory.get('workflow_plans', {})}"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Live execution used OpenAI because that is the provider key available for runtime verification."
    )
    lines.append(
        "- The optimization/runtime features exercised here are implemented in ByteAI Cache's shared stack, so the same prompt-piece, workflow, planner, memory, batching, and context-compiler behavior applies across the other adapters too."
    )
    lines.append(
        "- Media request avoidance is reported as avoided upstream calls, while text/coding/streaming scenarios report token-estimated cost and latency."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--coding-json", default=str(DEFAULT_CODING_REPORT))
    parser.add_argument("--surface-json", default=str(DEFAULT_SURFACE_REPORT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--json-report", default=str(DEFAULT_JSON_REPORT))
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("BYTE_TEST_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set BYTE_TEST_OPENAI_API_KEY or pass --api-key.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["BYTE_TEST_OPENAI_API_KEY"] = api_key
    env["PYTHONPATH"] = str(REPO_ROOT)

    coding_json_path = Path(args.coding_json)
    surface_json_path = Path(args.surface_json)

    _run_script(
        "deep_openai_coding_benchmark.py",
        env,
        "--json-report",
        str(coding_json_path),
        "--report",
        str(coding_json_path.with_suffix(".md")),
    )
    _run_script(
        "deep_openai_surface_benchmark.py",
        env,
        "--coding-json",
        str(coding_json_path),
        "--json-report",
        str(surface_json_path),
        "--report",
        str(surface_json_path.with_suffix(".md")),
    )

    coding_report = _load_json(coding_json_path)
    surface_report = _load_json(surface_json_path)
    unique_context = _run_unique_context_sequence(api_key, _build_unique_context_items())
    streaming = _run_streaming_benchmark(api_key, _build_streaming_items())

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "chat_model": CHAT_MODEL,
        "artifacts": {
            "coding_json": str(coding_json_path),
            "surface_json": str(surface_json_path),
            "comprehensive_json": str(Path(args.json_report)),
        },
        "coding_headlines": _coding_headlines(coding_report),
        "surface_headlines": _surface_headlines(surface_report),
        "coding_report": coding_report,
        "surface_report": surface_report,
        "unique_context": unique_context,
        "streaming": streaming,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results), encoding="utf-8")

    json_path = Path(args.json_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
