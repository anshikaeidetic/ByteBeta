import argparse
import base64
import binascii
import hashlib
import json
import os
import shutil
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from byte import Cache, Config  # pylint: disable=wrong-import-position
from byte._backends import openai as byte_openai  # pylint: disable=wrong-import-position
from byte.adapter.api import (  # pylint: disable=wrong-import-position
    init_cache,
    provider_capability_matrix,
)
from byte.benchmarking._optional_runtime import create_openai_client
from byte.benchmarking._program_common import (
    call_with_retry,
    make_item,
    p95,
    release_cache_tree,
)
from byte.benchmarking._program_defaults import load_program_defaults
from byte.benchmarking.programs import deep_openai_coding_benchmark as coding
from byte.processor.model_router import (
    clear_route_performance,  # pylint: disable=wrong-import-position
)
from byte.processor.pre import (  # pylint: disable=wrong-import-position
    get_file_bytes,
    get_openai_moderation_input,
    get_prompt,
    normalize_text,
    normalized_get_prompt,
)
from byte.processor.shared_memory import (
    clear_shared_memory,  # pylint: disable=wrong-import-position
)
from byte.utils.response import (  # pylint: disable=wrong-import-position
    get_audio_text_from_openai_answer,
    get_image_from_openai_b64,
    get_image_from_openai_url,
)

_DEFAULTS = load_program_defaults("deep_openai_surface_benchmark")

TEXT_MODES = list(_DEFAULTS.get("text_modes", ["exact", "normalized", "hybrid"]))
MEDIA_MODES = list(_DEFAULTS.get("media_modes", ["exact", "normalized"]))
CHAT_MODEL = coding.CHEAP_MODEL
IMAGE_MODEL_CANDIDATES = ["gpt-image-1-mini", "gpt-image-1"]
SPEECH_MODEL_CANDIDATES = ["gpt-4o-mini-tts", "tts-1"]
TRANSCRIBE_MODEL_CANDIDATES = ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"]

_make_item = make_item


def _build_normal_chat_scenario() -> dict[str, Any]:
    items = [
        _make_item(
            (
                "You are triaging a support inbox.\n"
                "Labels: BILLING, TECHNICAL, SHIPPING\n"
                'Ticket: "I was charged twice for the same subscription renewal and need the duplicate charge reversed."\n'
                "Reply with exactly one label."
            ),
            "BILLING",
            "support_billing",
            "v1",
            "support_classification",
        ),
        _make_item(
            (
                'Ticket: "I was charged twice for the same subscription renewal and need the duplicate charge reversed."\n'
                "Support categories: BILLING, TECHNICAL, SHIPPING\n"
                "Classify this request and answer with exactly one label."
            ),
            "BILLING",
            "support_billing",
            "v2",
            "support_classification",
        ),
        _make_item(
            (
                "You are triaging a support inbox.\n"
                "Labels: BILLING, TECHNICAL, SHIPPING\n"
                'Ticket: "The desktop app crashes every time I click export after the latest update."\n'
                "Reply with exactly one label."
            ),
            "TECHNICAL",
            "support_technical",
            "v1",
            "support_classification",
        ),
        _make_item(
            (
                'Ticket: "The desktop app crashes every time I click export after the latest update."\n'
                "Support categories: BILLING, TECHNICAL, SHIPPING\n"
                "Classify this request and answer with exactly one label."
            ),
            "TECHNICAL",
            "support_technical",
            "v2",
            "support_classification",
        ),
        _make_item(
            (
                "You are triaging a support inbox.\n"
                "Labels: BILLING, TECHNICAL, SHIPPING\n"
                'Ticket: "My order has shown in transit for eight days and still has not reached me."\n'
                "Reply with exactly one label."
            ),
            "SHIPPING",
            "support_shipping",
            "v1",
            "support_classification",
        ),
        _make_item(
            (
                'Ticket: "My order has shown in transit for eight days and still has not reached me."\n'
                "Support categories: BILLING, TECHNICAL, SHIPPING\n"
                "Classify this request and answer with exactly one label."
            ),
            "SHIPPING",
            "support_shipping",
            "v2",
            "support_classification",
        ),
        _make_item(
            "What is the capital of France? Reply with exactly PARIS and nothing else.",
            "PARIS",
            "general_qa_paris",
            "v1",
            "question_answer",
        ),
        _make_item(
            "Answer this concise factual question with exactly one token: France -> capital city -> PARIS.",
            "PARIS",
            "general_qa_paris",
            "v2",
            "question_answer",
        ),
    ]
    return {
        "name": "normal_chat_and_support_8",
        "description": "Short support-intent and normal chat traffic with repeated wrapper variants around the same underlying asks.",
        "items": items,
    }


def _build_document_scenario() -> dict[str, Any]:
    invoice_doc = (
        "Document: Invoice INV-2048\n"
        "Customer: Northwind Labs\n"
        "Amount Due: $4,250\n"
        "Due Date: 2026-04-18\n"
        "Notes: Annual analytics subscription renewal."
    )
    incident_doc = (
        "Incident report:\n"
        "- Service: payments-api\n"
        "- Start: 09:02 UTC\n"
        "- End: 09:47 UTC\n"
        "- Customer impact: checkout failures for card payments\n"
        "- Root cause: expired database credential after secret rotation\n"
        "- Mitigation: credential refreshed and pods restarted"
    )
    contract_doc = (
        "Clause text:\n"
        "This agreement renews automatically for successive one-year terms unless either party provides written notice at least 30 days before the renewal date."
    )
    release_bullets = (
        "Drafting notes:\n"
        "- Added CSV export for analytics dashboards\n"
        "- Reduced report generation latency by 35 percent\n"
        "- Fixed duplicate billing edge case for seat upgrades"
    )
    items = [
        _make_item(
            (
                "Read the incident report and answer with exactly one label that best matches the summary.\n"
                "Labels: PAYMENTS_OUTAGE, BILLING_ISSUE, SHIPMENT_DELAY\n"
                f"{incident_doc}"
            ),
            "PAYMENTS_OUTAGE",
            "doc_summary_incident",
            "v1",
            "document_summary",
        ),
        _make_item(
            (
                f"{incident_doc}\n"
                "Choose the best summary label from {PAYMENTS_OUTAGE, BILLING_ISSUE, SHIPMENT_DELAY} and reply with exactly one label."
            ),
            "PAYMENTS_OUTAGE",
            "doc_summary_incident",
            "v2",
            "document_summary",
        ),
        _make_item(
            (
                "Extract the invoice identifier from the following business document.\n"
                f"{invoice_doc}\n"
                "Return exactly the invoice identifier and nothing else."
            ),
            "INV-2048",
            "doc_extract_invoice",
            "v1",
            "document_extraction",
        ),
        _make_item(
            (f"{invoice_doc}\nAnswer with exactly the invoice number from this document."),
            "INV-2048",
            "doc_extract_invoice",
            "v2",
            "document_extraction",
        ),
        _make_item(
            (
                "Classify the contract clause.\n"
                "Labels: AUTO_RENEWAL, TERMINATION_FOR_CAUSE, DATA_PROCESSING\n"
                f"{contract_doc}\n"
                "Reply with exactly one label."
            ),
            "AUTO_RENEWAL",
            "doc_classify_clause",
            "v1",
            "document_classification",
        ),
        _make_item(
            (
                f"{contract_doc}\n"
                "Return exactly one label from {AUTO_RENEWAL, TERMINATION_FOR_CAUSE, DATA_PROCESSING}."
            ),
            "AUTO_RENEWAL",
            "doc_classify_clause",
            "v2",
            "document_classification",
        ),
        _make_item(
            (
                "Draft a release note headline based on the notes below, then reply with exactly RELEASE_NOTE_READY and nothing else.\n"
                f"{release_bullets}"
            ),
            "RELEASE_NOTE_READY",
            "doc_draft_release_note",
            "v1",
            "document_generation",
        ),
        _make_item(
            (
                f"{release_bullets}\n"
                "Prepare the release note title internally, but output exactly RELEASE_NOTE_READY."
            ),
            "RELEASE_NOTE_READY",
            "doc_draft_release_note",
            "v2",
            "document_generation",
        ),
    ]
    return {
        "name": "document_workflows_8",
        "description": "Document-heavy prompts covering summary, extraction, clause classification, and draft-generation shells.",
        "items": items,
    }


def _input_text(data: dict[str, Any], **_: Any) -> str:
    return str(data.get("input") or "")


def _normalized_input_text(data: dict[str, Any], **_: Any) -> str:
    return normalize_text(_input_text(data))


def _base_cache_config(scope: str) -> Config:
    return Config(
        enable_token_counter=False,
        tiered_cache=True,
        tier1_max_size=256,
        tier1_promote_on_write=True,
        async_write_back=True,
        memory_scope=scope,
        intent_memory=True,
        execution_memory=True,
        verified_reuse_for_all=False,
        verified_reuse_for_coding=False,
        delta_generation=True,
        context_compiler=True,
        ambiguity_detection=True,
        planner_enabled=True,
        failure_memory=True,
        tenant_policy_learning=True,
        budget_strategy="balanced",
    )


def _configure_media_cache(
    cache_obj: Cache,
    cache_dir: str,
    mode: str,
    *,
    scope: str,
    pre_func,
    normalized_pre_func=None,
) -> None:
    config = _base_cache_config(scope)
    init_cache(
        mode=mode,
        data_dir=cache_dir,
        cache_obj=cache_obj,
        pre_func=pre_func,
        normalized_pre_func=normalized_pre_func,
        config=config,
        exact_config=config,
        normalized_config=config,
    )


def _media_digest(data: bytes) -> dict[str, Any]:
    payload = data or b""
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest()[:16] if payload else "",
    }


def _media_text_valid(text: str | None, expectations: list[str] | None = None) -> bool:
    candidate = (text or "").strip().lower()
    if not candidate:
        return False
    if not expectations:
        return True
    return all(token.lower() in candidate for token in expectations)


def _media_record(
    *,
    status_code: int,
    latency_ms: float,
    byte_flag: bool,
    model_name: str,
    output_bytes: int = 0,
    output_sha: str = "",
    valid_output: bool = True,
    text: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "byte": byte_flag,
        "model": model_name,
        "output_bytes": output_bytes,
        "output_sha": output_sha,
        "valid_output": valid_output,
        "text": text,
        "error": error,
    }


def _summarize_media_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records]
    return {
        "request_count": len(records),
        "cached_count": sum(1 for record in records if record["byte"]),
        "miss_count": sum(1 for record in records if not record["byte"]),
        "error_count": sum(1 for record in records if record["status_code"] != 200),
        "valid_ratio": round(
            sum(1 for record in records if record["valid_output"]) / len(records),
            4,
        )
        if records
        else 0.0,
        "hit_ratio": round(
            sum(1 for record in records if record["byte"]) / len(records),
            4,
        )
        if records
        else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": p95(latencies),
        "avg_output_bytes": round(statistics.mean(record["output_bytes"] for record in records), 2)
        if records
        else 0.0,
        "models": {
            model: sum(1 for record in records if record["model"] == model)
            for model in sorted({record["model"] for record in records if record["model"]})
        },
        "sample": records[:3],
    }


def _media_scenario_summary(
    name: str, description: str, runs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    direct = runs["direct"]
    baseline_requests = direct["request_count"]
    baseline_latency = direct["avg_latency_ms"]
    for key, data in runs.items():
        if key == "direct":
            data["avoided_upstream_calls"] = 0
            data["avoided_call_ratio"] = 0.0
            data["latency_delta_ms"] = 0.0
            continue
        data["avoided_upstream_calls"] = max(baseline_requests - data["miss_count"], 0)
        data["avoided_call_ratio"] = (
            round(
                max(baseline_requests - data["miss_count"], 0) / baseline_requests,
                4,
            )
            if baseline_requests
            else 0.0
        )
        data["latency_delta_ms"] = round(data["avg_latency_ms"] - baseline_latency, 2)
    return {
        "name": name,
        "description": description,
        "runs": runs,
    }


def _client(api_key: str) -> Any:
    return create_openai_client(api_key=api_key)


def _extract_image_bytes(response: Any) -> bytes:
    response_dict = byte_openai._openai_response_to_dict(response)  # pylint: disable=protected-access
    try:
        img_b64 = get_image_from_openai_b64(response_dict)
        if isinstance(img_b64, str):
            return base64.b64decode(img_b64)
        return base64.b64decode(img_b64.decode("ascii"))
    except (TypeError, ValueError, binascii.Error):
        img_b64 = get_image_from_openai_url(response_dict)
        return base64.b64decode(img_b64)


def _first_working_candidate(candidates: list[str], probe) -> tuple[str | None, str]:
    """Return the first candidate accepted by a live provider probe."""

    last_error = ""
    for model in candidates:
        try:  # provider boundary probe
            if probe(model):
                return model, ""
        except Exception as exc:
            last_error = str(exc)
    return None, last_error


def _choose_working_image_model(api_key: str, prompt: str) -> tuple[str | None, str]:
    client = _client(api_key)
    return _first_working_candidate(
        IMAGE_MODEL_CANDIDATES,
        lambda model: bool(
            _extract_image_bytes(
                client.images.generate(
                    model=model,
                    prompt=prompt,
                    size="1024x1024",
                    quality="low",
                )
            )
        ),
    )


def _choose_working_speech_model(api_key: str, text: str) -> tuple[str | None, str]:
    client = _client(api_key)
    return _first_working_candidate(
        SPEECH_MODEL_CANDIDATES,
        lambda model: bool(
            byte_openai._extract_audio_bytes(  # pylint: disable=protected-access
                client.audio.speech.create(
                    model=model,
                    voice="alloy",
                    input=text,
                    response_format="mp3",
                )
            )
        ),
    )


def _choose_working_transcribe_model(api_key: str, audio_path: str) -> tuple[str | None, str]:
    client = _client(api_key)

    def probe(model: str) -> bool:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
            )
        response_dict = byte_openai._openai_response_to_dict(response)  # pylint: disable=protected-access
        return bool(get_audio_text_from_openai_answer(response_dict))

    return _first_working_candidate(TRANSCRIBE_MODEL_CANDIDATES, probe)


def _capture_media_request(request, *, model: str, byte_flag: bool, on_success) -> dict[str, Any]:
    """Wrap one provider request and normalize success and failure output."""

    start = time.perf_counter()
    try:  # provider boundary request
        response = call_with_retry(request)
        latency_ms = (time.perf_counter() - start) * 1000
        return on_success(response, latency_ms)
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return _media_record(
            status_code=599,
            latency_ms=latency_ms,
            byte_flag=byte_flag,
            model_name=model,
            valid_output=False,
            error=str(exc),
        )


def _direct_image_request(api_key: str, prompt: str, model: str) -> dict[str, Any]:
    client = _client(api_key)
    return _capture_media_request(
        lambda: client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
            quality="low",
        ),
        model=model,
        byte_flag=False,
        on_success=lambda response, latency_ms: (
            lambda digest: _media_record(
                status_code=200,
                latency_ms=latency_ms,
                byte_flag=False,
                model_name=model,
                output_bytes=digest["bytes"],
                output_sha=digest["sha256"],
            )
        )(_media_digest(_extract_image_bytes(response))),
    )


def _byte_image_request(api_key: str, cache_obj: Cache, prompt: str, model: str) -> dict[str, Any]:
    return _capture_media_request(
        lambda: byte_openai.Image.create(
            model=model,
            prompt=prompt,
            size="1024x1024",
            quality="low",
            response_format="b64_json",
            api_key=api_key,
            cache_obj=cache_obj,
        ),
        model=model,
        byte_flag=False,
        on_success=lambda response, latency_ms: (
            lambda digest: _media_record(
                status_code=200,
                latency_ms=latency_ms,
                byte_flag=bool(response.get("byte")),
                model_name=model,
                output_bytes=digest["bytes"],
                output_sha=digest["sha256"],
            )
        )(
            _media_digest(
                base64.b64decode(
                    (lambda img_b64: img_b64 if isinstance(img_b64, str) else img_b64.decode("ascii"))(
                        get_image_from_openai_b64(response)
                    )
                )
            )
        ),
    )


def _direct_speech_request(api_key: str, text: str, model: str) -> dict[str, Any]:
    client = _client(api_key)
    return _capture_media_request(
        lambda: client.audio.speech.create(
            model=model,
            voice="alloy",
            input=text,
            response_format="mp3",
        ),
        model=model,
        byte_flag=False,
        on_success=lambda response, latency_ms: (
            lambda digest: _media_record(
                status_code=200,
                latency_ms=latency_ms,
                byte_flag=False,
                model_name=model,
                output_bytes=digest["bytes"],
                output_sha=digest["sha256"],
            )
        )(_media_digest(byte_openai._extract_audio_bytes(response))),  # pylint: disable=protected-access
    )


def _byte_speech_request(api_key: str, cache_obj: Cache, text: str, model: str) -> dict[str, Any]:
    return _capture_media_request(
        lambda: byte_openai.Speech.create(
            model=model,
            input=text,
            voice="alloy",
            response_format="mp3",
            api_key=api_key,
            cache_obj=cache_obj,
        ),
        model=model,
        byte_flag=False,
        on_success=lambda response, latency_ms: (
            lambda digest: _media_record(
                status_code=200,
                latency_ms=latency_ms,
                byte_flag=bool(response.get("byte")),
                model_name=model,
                output_bytes=digest["bytes"],
                output_sha=digest["sha256"],
            )
        )(_media_digest(response.get("audio", b""))),
    )


def _write_audio_fixture(api_key: str, speech_model: str, text: str) -> tuple[str | None, str]:
    client = _client(api_key)
    temp_dir = tempfile.mkdtemp(prefix="byte-audio-fixture-")
    audio_path = os.path.join(temp_dir, "fixture.mp3")
    try:  # provider boundary fixture generation
        response = client.audio.speech.create(
            model=speech_model,
            voice="alloy",
            input=text,
            response_format="mp3",
        )
        audio_bytes = byte_openai._extract_audio_bytes(response)  # pylint: disable=protected-access
        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_bytes)
        return audio_path, ""
    except Exception as exc:  # pylint: disable=broad-except
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, str(exc)


def _direct_transcribe_request(api_key: str, audio_path: str, model: str) -> dict[str, Any]:
    client = _client(api_key)
    def request() -> Any:
        with open(audio_path, "rb") as audio_file:
            return client.audio.transcriptions.create(model=model, file=audio_file)

    return _capture_media_request(
        request,
        model=model,
        byte_flag=False,
        on_success=lambda response, latency_ms: (
            lambda response_dict, text: _media_record(
                status_code=200,
                latency_ms=latency_ms,
                byte_flag=False,
                model_name=model,
                valid_output=_media_text_valid(text, ["byte", "requests"]),
                text=text,
            )
        )(
            byte_openai._openai_response_to_dict(response),  # pylint: disable=protected-access
            get_audio_text_from_openai_answer(
                byte_openai._openai_response_to_dict(response)  # pylint: disable=protected-access
            ),
        ),
    )


def _byte_transcribe_request(
    api_key: str, cache_obj: Cache, audio_path: str, model: str
) -> dict[str, Any]:
    def request() -> Any:
        with open(audio_path, "rb") as audio_file:
            return byte_openai.Audio.transcribe(
                model,
                audio_file,
                api_key=api_key,
                cache_obj=cache_obj,
            )

    return _capture_media_request(
        request,
        model=model,
        byte_flag=False,
        on_success=lambda response, latency_ms: (
            lambda text: _media_record(
                status_code=200,
                latency_ms=latency_ms,
                byte_flag=bool(response.get("byte")),
                model_name=model,
                valid_output=_media_text_valid(text, ["byte", "requests"]),
                text=text,
            )
        )(get_audio_text_from_openai_answer(response)),
    )


def _direct_moderation_request(api_key: str, moderation_input: str) -> dict[str, Any]:
    client = _client(api_key)
    return _capture_media_request(
        lambda: client.moderations.create(input=moderation_input),
        model="omni-moderation-latest",
        byte_flag=False,
        on_success=lambda response, latency_ms: (
            lambda response_dict, flagged: _media_record(
                status_code=200,
                latency_ms=latency_ms,
                byte_flag=False,
                model_name=str(response_dict.get("model") or "omni-moderation-latest"),
                valid_output=flagged,
                text=str(flagged),
            )
        )(
            byte_openai._openai_response_to_dict(response),  # pylint: disable=protected-access
            bool(
                (
                    (
                        byte_openai._openai_response_to_dict(response)  # pylint: disable=protected-access
                    ).get("results")
                    or [{}]
                )[0].get("flagged")
            ),
        ),
    )


def _byte_moderation_request(
    api_key: str, cache_obj: Cache, moderation_input: str
) -> dict[str, Any]:
    return _capture_media_request(
        lambda: byte_openai.Moderation.create(
            input=[moderation_input],
            api_key=api_key,
            cache_obj=cache_obj,
        ),
        model="omni-moderation-latest",
        byte_flag=False,
        on_success=lambda response, latency_ms: (
            lambda flagged: _media_record(
                status_code=200,
                latency_ms=latency_ms,
                byte_flag=bool(response.get("byte")),
                model_name=str(response.get("model") or "omni-moderation-latest"),
                valid_output=flagged,
                text=str(flagged),
            )
        )(bool(((response.get("results") or [{}])[0]).get("flagged"))),
    )


def _release_cache_tree(cache_obj: Cache) -> None:
    release_cache_tree(cache_obj)


def _run_byte_media_sequence(
    prompts: list[str],
    *,
    mode: str,
    scenario_name: str,
    pre_func,
    normalized_pre_func,
    run_request,
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"surface-{scenario_name}-{mode}-")
    scope = f"surface::{scenario_name}::{mode}::{int(time.time() * 1000)}"
    clear_shared_memory(scope)
    cache_obj = Cache()
    try:
        clear_route_performance()
        _configure_media_cache(
            cache_obj,
            cache_dir,
            mode,
            scope=scope,
            pre_func=pre_func,
            normalized_pre_func=normalized_pre_func,
        )
        records = [run_request(cache_obj, prompt) for prompt in prompts]
        return _summarize_media_records(records)
    finally:
        _release_cache_tree(cache_obj)
        shutil.rmtree(cache_dir, ignore_errors=True)
        clear_shared_memory(scope)


def _load_coding_report(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _coding_highlights(report: dict[str, Any]) -> list[dict[str, Any]]:
    if not report:
        return []
    highlights = []
    for scenario in report.get("sequential_scenarios", [])[:5]:
        hybrid = (scenario.get("runs") or {}).get("hybrid", {})
        highlights.append(
            {
                "name": scenario.get("name"),
                "requests": scenario.get("request_count"),
                "hybrid_hit_ratio": hybrid.get("hit_ratio"),
                "hybrid_savings_ratio": hybrid.get("savings_ratio"),
                "hybrid_accuracy": hybrid.get("accuracy_ratio"),
                "hybrid_avg_latency_ms": hybrid.get("avg_latency_ms"),
            }
        )
    routing = (report.get("routing_blend") or {}).get("runs", {})
    if routing:
        routed = routing.get("byte_hybrid_routed", {})
        highlights.append(
            {
                "name": "routing_blend",
                "requests": (report.get("routing_blend") or {}).get("request_count"),
                "hybrid_hit_ratio": routed.get("hit_ratio"),
                "hybrid_savings_ratio": routed.get("savings_ratio"),
                "hybrid_accuracy": routed.get("accuracy_ratio"),
                "hybrid_avg_latency_ms": routed.get("avg_latency_ms"),
            }
        )
    return highlights


def _render_text_mode_line(name: str, data: dict[str, Any]) -> str:
    return coding._render_mode_line(name, data)  # pylint: disable=protected-access


def _render_media_mode_line(
    name: str, data: dict[str, Any], *, baseline_label: str = "direct"
) -> str:
    models = ", ".join(
        f"{model}:{count}" for model, count in sorted((data.get("models") or {}).items())
    )
    suffix = f", models=({models})" if models else ""
    return (
        f"- {name}: hit_ratio={data['hit_ratio']}, valid_ratio={data['valid_ratio']}, "
        f"avg_latency={data['avg_latency_ms']} ms, p95_latency={data['p95_latency_ms']} ms, "
        f"avoided_{baseline_label}_calls={data.get('avoided_upstream_calls', 0)}, "
        f"avoided_call_ratio={data.get('avoided_call_ratio', 0.0)}, avg_output_bytes={data['avg_output_bytes']}{suffix}"
    )


def _render_capability_matrix(matrix: dict[str, dict[str, Any]]) -> list[str]:
    lines = []
    preferred = [
        "openai",
        "deepseek",
        "anthropic",
        "gemini",
        "groq",
        "openrouter",
        "ollama",
        "mistral",
        "cohere",
        "bedrock",
        "llama_cpp",
    ]
    ordered = [provider for provider in preferred if provider in matrix]
    ordered.extend(sorted(provider for provider in matrix if provider not in ordered))
    for provider in ordered:
        caps = matrix.get(provider, {})
        media = []
        for name in (
            "image_generation",
            "audio_transcription",
            "audio_translation",
            "speech_generation",
            "moderation",
        ):
            if caps.get(name):
                media.append(name)
        runtime = []
        for name in (
            "smart_model_routing",
            "execution_verified_memory",
            "delta_generation",
            "context_compiler",
            "workflow_planner",
            "budget_aware_serving",
            "cheap_consensus_verification",
            "negative_context_memory",
            "counterfactual_workflow_memory",
            "uncertainty_conditioned_context_budget",
            "evidence_aware_verification",
            "source_context_gap_detection",
            "strict_cache_revalidation",
        ):
            if caps.get(name):
                runtime.append(name)
        inputs = []
        for name in ("vision_inputs", "document_inputs", "audio_inputs"):
            if caps.get(name):
                inputs.append(name)
        lines.append(
            f"- `{provider}`: chat={caps.get('chat_completion', False)}, docs={caps.get('document_tasks', False)}, "
            f"coding={caps.get('coding_tasks', False)}, inputs={inputs}, media={media}, shared_runtime={runtime}"
        )
    return lines


def _render_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# ByteAI Cache OpenAI Surface Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {results['generated_at']}")
    lines.append(f"Chat benchmark model: `{results['chat_model']}`")
    lines.append("")
    lines.append(
        "This report combines the Cursor-style coding benchmark with broader OpenAI coverage for normal chat, document workflows, moderation, image generation, speech generation, and transcription."
    )
    lines.append(
        "The runtime features underneath the benchmark are shared across all adapters. The capability matrix below shows which provider adapters currently expose which endpoint surfaces inside ByteAI Cache."
    )
    lines.append("")
    lines.append("## Adapter Capability Matrix")
    lines.append("")
    lines.extend(_render_capability_matrix(results["capability_matrix"]))
    lines.append("")
    lines.append("## Coding Highlights")
    lines.append("")
    if results["coding_report"]:
        lines.append(
            f"- Source report: `{results['coding_report'].get('path', 'docs/reports/openai_cursor_coding_benchmark.json')}`"
        )
        for highlight in results["coding_highlights"]:
            lines.append(
                f"- {highlight['name']}: requests={highlight['requests']}, hybrid_hit_ratio={highlight['hybrid_hit_ratio']}, "
                f"hybrid_savings_ratio={highlight['hybrid_savings_ratio']}, hybrid_accuracy={highlight['hybrid_accuracy']}, "
                f"hybrid_avg_latency={highlight['hybrid_avg_latency_ms']} ms"
            )
    else:
        lines.append("- Coding report was not available.")
    lines.append("")
    lines.append("## Text Benchmarks")
    lines.append("")
    for scenario in results["text_scenarios"]:
        lines.append(f"### {scenario['name']}")
        lines.append(f"- {scenario['description']}")
        lines.append(_render_text_mode_line("Direct", scenario["runs"]["direct"]))
        for mode in TEXT_MODES:
            lines.append(_render_text_mode_line(f"ByteAI Cache {mode}", scenario["runs"][mode]))
        lines.append("")
    lines.append("## Moderation")
    lines.append("")
    moderation = results["moderation"]
    lines.append(f"- {moderation['description']}")
    lines.append(_render_media_mode_line("Direct", moderation["runs"]["direct"]))
    for mode in ("exact", "normalized"):
        lines.append(_render_media_mode_line(f"ByteAI Cache {mode}", moderation["runs"][mode]))
    lines.append("")
    lines.append("## Image Generation")
    lines.append("")
    image = results["image_generation"]
    lines.append(f"- {image['description']}")
    if image.get("unavailable"):
        lines.append(f"- Unavailable in this run: {image['unavailable']}")
    else:
        lines.append(f"- Working model: `{image['model']}`")
        lines.append(_render_media_mode_line("Direct", image["runs"]["direct"]))
        for mode in ("exact", "normalized"):
            lines.append(_render_media_mode_line(f"ByteAI Cache {mode}", image["runs"][mode]))
    lines.append("")
    lines.append("## Speech Generation")
    lines.append("")
    speech = results["speech_generation"]
    lines.append(f"- {speech['description']}")
    if speech.get("unavailable"):
        lines.append(f"- Unavailable in this run: {speech['unavailable']}")
    else:
        lines.append(f"- Working model: `{speech['model']}`")
        lines.append(_render_media_mode_line("Direct", speech["runs"]["direct"]))
        for mode in ("exact", "normalized"):
            lines.append(_render_media_mode_line(f"ByteAI Cache {mode}", speech["runs"][mode]))
    lines.append("")
    lines.append("## Audio Transcription")
    lines.append("")
    transcription = results["transcription"]
    lines.append(f"- {transcription['description']}")
    if transcription.get("unavailable"):
        lines.append(f"- Unavailable in this run: {transcription['unavailable']}")
    else:
        lines.append(f"- Working model: `{transcription['model']}`")
        lines.append(_render_media_mode_line("Direct", transcription["runs"]["direct"]))
        lines.append(_render_media_mode_line("ByteAI Cache exact", transcription["runs"]["exact"]))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Text and coding sections report token-estimated cost and latency because those endpoints expose usage clearly."
    )
    lines.append(
        "- Media and moderation sections report avoided upstream calls and latency because those endpoint pricing units differ from chat and are better compared as request avoidance in this benchmark."
    )
    lines.append(
        "- OpenAI is the only provider exercised live in this benchmark run, but the shared runtime features listed above are implemented centrally and available across ByteAI Cache's built-in adapters. Rich endpoint coverage varies by provider and is reflected in the capability matrix."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--coding-json", default="docs/reports/openai_cursor_coding_benchmark.json")
    parser.add_argument("--report", default="docs/reports/openai_surface_benchmark.md")
    parser.add_argument("--json-report", default="docs/reports/openai_surface_benchmark.json")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("BYTE_TEST_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set BYTE_TEST_OPENAI_API_KEY or pass --api-key.")

    results: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "chat_model": CHAT_MODEL,
        "capability_matrix": provider_capability_matrix(),
        "text_scenarios": [],
    }

    coding_report = _load_coding_report(args.coding_json)
    if coding_report:
        coding_report["path"] = args.coding_json
    results["coding_report"] = coding_report
    results["coding_highlights"] = _coding_highlights(coding_report)

    for scenario in (_build_normal_chat_scenario(), _build_document_scenario()):
        runs = {"direct": coding._run_direct_sequence(api_key, scenario["items"], model=CHAT_MODEL)}  # pylint: disable=protected-access
        for mode in TEXT_MODES:
            runs[mode] = coding._run_byte_sequence(  # pylint: disable=protected-access
                api_key,
                scenario["items"],
                mode=mode,
                model=CHAT_MODEL,
                routed=False,
                scenario_name=scenario["name"],
                warm_data=None,
            )
        results["text_scenarios"].append(
            coding._scenario_summary(  # pylint: disable=protected-access
                scenario["name"],
                scenario["description"],
                scenario["items"],
                runs,
            )
        )

    moderation_inputs = [
        "I want to kill them.",
        "I want to kill them!",
    ]
    moderation_runs = {
        "direct": _summarize_media_records(
            [_direct_moderation_request(api_key, item) for item in moderation_inputs]
        ),
    }
    for mode in MEDIA_MODES:
        moderation_runs[mode] = _run_byte_media_sequence(
            moderation_inputs,
            mode=mode,
            scenario_name="moderation",
            pre_func=get_openai_moderation_input,
            normalized_pre_func=_normalized_input_text,
            run_request=lambda cache_obj, prompt: _byte_moderation_request(
                api_key, cache_obj, prompt
            ),
        )
    results["moderation"] = _media_scenario_summary(
        "moderation_variants_2",
        "Repeated moderation of a harmful input with punctuation-only variation to show exact versus normalized reuse.",
        moderation_runs,
    )

    image_prompts = [
        "Create a clean flat icon of a blue whale on a white background.",
        "Create a clean flat icon of a blue whale, on a white background.",
    ]
    image_model, image_error = _choose_working_image_model(api_key, image_prompts[0])
    if not image_model:
        results["image_generation"] = {
            "description": "Image generation benchmark using prompt-variant reuse.",
            "unavailable": image_error,
        }
    else:
        image_runs = {
            "direct": _summarize_media_records(
                [_direct_image_request(api_key, prompt, image_model) for prompt in image_prompts]
            ),
        }
        for mode in MEDIA_MODES:
            image_runs[mode] = _run_byte_media_sequence(
                image_prompts,
                mode=mode,
                scenario_name="image_generation",
                pre_func=get_prompt,
                normalized_pre_func=normalized_get_prompt,
                run_request=lambda cache_obj, prompt: _byte_image_request(
                    api_key, cache_obj, prompt, image_model
                ),
            )
        image_summary = _media_scenario_summary(
            "image_generation_variants_2",
            "Image generation with punctuation-only prompt variation to test exact versus normalized media reuse.",
            image_runs,
        )
        image_summary["model"] = image_model
        results["image_generation"] = image_summary

    speech_inputs = [
        "Byte speeds up repeated AI requests for support, coding, and docs.",
        "Byte speeds up repeated AI requests for support, coding and docs.",
    ]
    speech_model, speech_error = _choose_working_speech_model(api_key, speech_inputs[0])
    if not speech_model:
        results["speech_generation"] = {
            "description": "Speech generation benchmark using punctuation-only prompt variation.",
            "unavailable": speech_error,
        }
        results["transcription"] = {
            "description": "Audio transcription of a generated speech fixture.",
            "unavailable": f"Speech model unavailable, so no audio fixture could be generated. {speech_error}",
        }
    else:
        speech_runs = {
            "direct": _summarize_media_records(
                [_direct_speech_request(api_key, text, speech_model) for text in speech_inputs]
            ),
        }
        for mode in MEDIA_MODES:
            speech_runs[mode] = _run_byte_media_sequence(
                speech_inputs,
                mode=mode,
                scenario_name="speech_generation",
                pre_func=_input_text,
                normalized_pre_func=_normalized_input_text,
                run_request=lambda cache_obj, prompt: _byte_speech_request(
                    api_key, cache_obj, prompt, speech_model
                ),
            )
        speech_summary = _media_scenario_summary(
            "speech_generation_variants_2",
            "Speech generation with punctuation-only input variation to test exact versus normalized reuse.",
            speech_runs,
        )
        speech_summary["model"] = speech_model
        results["speech_generation"] = speech_summary

        audio_path, audio_error = _write_audio_fixture(api_key, speech_model, speech_inputs[0])
        if not audio_path:
            results["transcription"] = {
                "description": "Audio transcription of a generated speech fixture.",
                "unavailable": audio_error,
            }
        else:
            try:
                transcribe_model, transcribe_error = _choose_working_transcribe_model(
                    api_key, audio_path
                )
                if not transcribe_model:
                    results["transcription"] = {
                        "description": "Audio transcription of a generated speech fixture.",
                        "unavailable": transcribe_error,
                    }
                else:
                    transcription_runs = {
                        "direct": _summarize_media_records(
                            [
                                _direct_transcribe_request(api_key, audio_path, transcribe_model),
                                _direct_transcribe_request(api_key, audio_path, transcribe_model),
                            ]
                        ),
                        "exact": _run_byte_media_sequence(
                            [audio_path, audio_path],
                            mode="exact",
                            scenario_name="audio_transcription",
                            pre_func=get_file_bytes,
                            normalized_pre_func=None,
                            run_request=lambda cache_obj, path: _byte_transcribe_request(
                                api_key, cache_obj, path, transcribe_model
                            ),
                        ),
                    }
                    transcription_summary = _media_scenario_summary(
                        "audio_transcription_exact_2",
                        "Transcription of the same generated speech fixture twice to measure exact audio-cache reuse.",
                        transcription_runs,
                    )
                    transcription_summary["model"] = transcribe_model
                    results["transcription"] = transcription_summary
            finally:
                shutil.rmtree(os.path.dirname(audio_path), ignore_errors=True)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results), encoding="utf-8")

    json_path = Path(args.json_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
