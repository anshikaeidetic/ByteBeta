import argparse
import copy
import json
import os
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from byte.adapter.api import provider_capability_matrix  # pylint: disable=wrong-import-position
from byte.benchmarking._program_defaults import load_program_defaults
from byte.benchmarking.programs import deep_openai_coding_benchmark as coding
from byte.benchmarking.programs import deep_openai_prompt_stress_benchmark as prompt_stress
from byte.benchmarking.programs import deep_openai_surface_benchmark as surface
from byte.processor.pre import (  # pylint: disable=wrong-import-position
    get_file_bytes,
    get_openai_moderation_input,
    get_prompt,
    normalized_get_prompt,
)

REPORT_DIR = REPO_ROOT / "docs" / "reports"
DEFAULT_REPORT = REPORT_DIR / "openai_1000_request_mixed_benchmark.md"
DEFAULT_JSON_REPORT = REPORT_DIR / "openai_1000_request_mixed_benchmark.json"

_DEFAULTS = load_program_defaults("deep_openai_1000_request_mixed_benchmark")

TEXT_WAVES = int(_DEFAULTS.get("text_waves", 8))
TEXT_REQUESTS_PER_WAVE = int(_DEFAULTS.get("text_requests_per_wave", 100))
TEXT_REQUEST_COUNT = TEXT_WAVES * TEXT_REQUESTS_PER_WAVE
MODERATION_REQUEST_COUNT = int(_DEFAULTS.get("moderation_request_count", 80))
IMAGE_REQUEST_COUNT = int(_DEFAULTS.get("image_request_count", 40))
SPEECH_REQUEST_COUNT = int(_DEFAULTS.get("speech_request_count", 40))
TRANSCRIPTION_REQUEST_COUNT = int(_DEFAULTS.get("transcription_request_count", 40))
PLANNED_REQUEST_COUNT = (
    TEXT_REQUEST_COUNT
    + MODERATION_REQUEST_COUNT
    + IMAGE_REQUEST_COUNT
    + SPEECH_REQUEST_COUNT
    + TRANSCRIPTION_REQUEST_COUNT
)
CHAT_MODEL = prompt_stress.CHAT_MODEL

VIDEO_SUPPORT = dict(_DEFAULTS.get("video_support", {}))

PLANNED_SEGMENTS = {
    "text_core": TEXT_REQUEST_COUNT,
    "moderation": MODERATION_REQUEST_COUNT,
    "image_generation": IMAGE_REQUEST_COUNT,
    "speech_generation": SPEECH_REQUEST_COUNT,
    "audio_transcription": TRANSCRIPTION_REQUEST_COUNT,
    "video": 0,
}

PLAIN_UNIQUE_TEMPLATES = [
    "Single-token benchmark request {index:04d}. Topic: duplicate billing in region {index:04d}. Reply exactly STRESS_UNIQUE_{index:04d} and nothing else.",
    "Single-token benchmark request {index:04d}. Topic: isolated export incident {index:04d}. Reply exactly STRESS_UNIQUE_{index:04d} and nothing else.",
    "Single-token benchmark request {index:04d}. Topic: release note draft {index:04d}. Reply exactly STRESS_UNIQUE_{index:04d} and nothing else.",
    "Single-token benchmark request {index:04d}. Topic: logistics escalation {index:04d}. Reply exactly STRESS_UNIQUE_{index:04d} and nothing else.",
    "Single-token benchmark request {index:04d}. Topic: code review follow-up {index:04d}. Reply exactly STRESS_UNIQUE_{index:04d} and nothing else.",
]

SHARED_CONTEXT_PROMPTS = [
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

IMAGE_PROMPT_PAIRS = [
    (
        "Create a clean flat icon of a blue whale on a white background.",
        "Create a clean flat icon of a blue whale, on a white background.",
    ),
    (
        "Create a clean flat icon of a red lighthouse on a white background.",
        "Create a clean flat icon of a red lighthouse, on a white background.",
    ),
    (
        "Create a clean flat icon of a green cactus on a white background.",
        "Create a clean flat icon of a green cactus, on a white background.",
    ),
    (
        "Create a clean flat icon of a yellow submarine on a white background.",
        "Create a clean flat icon of a yellow submarine, on a white background.",
    ),
    (
        "Create a clean flat icon of an orange fox on a white background.",
        "Create a clean flat icon of an orange fox, on a white background.",
    ),
]

SPEECH_INPUT_PAIRS = [
    (
        "Byte speeds up repeated AI requests for support, coding, and docs.",
        "Byte speeds up repeated AI requests for support, coding and docs.",
    ),
    (
        "Byte keeps prompt reuse deterministic for support and workflow automation.",
        "Byte keeps prompt reuse deterministic for support and workflow automation!",
    ),
    (
        "Byte reduces repeated coding assistant traffic across large support queues.",
        "Byte reduces repeated coding assistant traffic across large support queues!",
    ),
    (
        "Byte compiles shared context to keep document and coding prompts smaller.",
        "Byte compiles shared context to keep document and coding prompts smaller!",
    ),
    (
        "Byte remembers reusable artifacts for docs, support, and code reviews.",
        "Byte remembers reusable artifacts for docs, support, and code reviews!",
    ),
]

MODERATION_INPUT_PAIR = (
    "I want to kill them.",
    "I want to kill them!",
)


def _clone_item(item: dict[str, Any], *, wave_index: int, item_index: int) -> dict[str, Any]:
    cloned = copy.deepcopy(item)
    cloned["wave"] = wave_index
    cloned["variant"] = f"{item.get('variant', 'base')}::w{wave_index:02d}::{item_index:03d}"
    return cloned


def _build_shared_context_wave(wave_index: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for session_index in range(5):
        session_id = f"mega-session-{wave_index:02d}-{session_index + 1:02d}"
        for prompt_index, (group_stub, prompt_template) in enumerate(SHARED_CONTEXT_PROMPTS, 1):
            token = f"CTX_{group_stub.upper()}_{wave_index:02d}_{session_index + 1:02d}"
            items.append(
                prompt_stress._build_contextual_item(  # pylint: disable=protected-access
                    session_id,
                    token,
                    group=f"mega_context_{group_stub}_{wave_index:02d}_{session_index + 1:02d}",
                    short_prompt=prompt_template.format(token=token),
                    variant=f"w{wave_index:02d}_v{prompt_index}",
                )
            )
    return items


def _build_plain_unique_wave(wave_index: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    base_index = (wave_index - 1) * 20
    for local_index in range(20):
        absolute_index = base_index + local_index + 1
        token = f"STRESS_UNIQUE_{absolute_index:04d}"
        template = PLAIN_UNIQUE_TEMPLATES[local_index % len(PLAIN_UNIQUE_TEMPLATES)]
        item = coding._make_item(  # pylint: disable=protected-access
            template.format(index=absolute_index),
            token,
            f"mega_unique_{absolute_index:04d}",
            f"w{wave_index:02d}",
            "instruction",
            max_tokens=12,
        )
        item["bucket"] = "plain_unique"
        item["request_style"] = "standard"
        items.append(item)
    return items


def _build_text_wave(wave_index: int) -> list[dict[str, Any]]:
    builders = (
        prompt_stress._build_exact_repeat_bucket,  # pylint: disable=protected-access
        prompt_stress._build_normalized_variant_bucket,  # pylint: disable=protected-access
        prompt_stress._build_coding_mixed_bucket,  # pylint: disable=protected-access
    )
    items: list[dict[str, Any]] = []
    item_index = 0
    for builder in builders:
        for item in builder():
            item_index += 1
            items.append(_clone_item(item, wave_index=wave_index, item_index=item_index))
    for item in _build_shared_context_wave(wave_index):
        item_index += 1
        items.append(_clone_item(item, wave_index=wave_index, item_index=item_index))
    for item in _build_plain_unique_wave(wave_index):
        item_index += 1
        items.append(_clone_item(item, wave_index=wave_index, item_index=item_index))
    if len(items) != TEXT_REQUESTS_PER_WAVE:
        raise ValueError(
            f"Expected {TEXT_REQUESTS_PER_WAVE} text items in wave {wave_index}, found {len(items)}"
        )
    return items


def _build_text_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for wave_index in range(1, TEXT_WAVES + 1):
        items.extend(_build_text_wave(wave_index))
    if len(items) != TEXT_REQUEST_COUNT:
        raise ValueError(f"Expected {TEXT_REQUEST_COUNT} text items, found {len(items)}")
    return items


def _repeat_pair(pair: tuple[str, str], repetitions: int) -> list[str]:
    items: list[str] = []
    for _ in range(repetitions):
        items.extend(pair)
    return items


def _build_moderation_inputs() -> list[str]:
    items = _repeat_pair(MODERATION_INPUT_PAIR, MODERATION_REQUEST_COUNT // 2)
    if len(items) != MODERATION_REQUEST_COUNT:
        raise ValueError(
            f"Expected {MODERATION_REQUEST_COUNT} moderation inputs, found {len(items)}"
        )
    return items


def _build_image_prompts() -> list[str]:
    items: list[str] = []
    for pair in IMAGE_PROMPT_PAIRS:
        items.extend(_repeat_pair(pair, 4))
    if len(items) != IMAGE_REQUEST_COUNT:
        raise ValueError(f"Expected {IMAGE_REQUEST_COUNT} image prompts, found {len(items)}")
    return items


def _build_speech_inputs() -> list[str]:
    items: list[str] = []
    for pair in SPEECH_INPUT_PAIRS:
        items.extend(_repeat_pair(pair, 4))
    if len(items) != SPEECH_REQUEST_COUNT:
        raise ValueError(f"Expected {SPEECH_REQUEST_COUNT} speech inputs, found {len(items)}")
    return items


def _build_transcription_inputs(audio_path: str) -> list[str]:
    items = [audio_path for _ in range(TRANSCRIPTION_REQUEST_COUNT)]
    if len(items) != TRANSCRIPTION_REQUEST_COUNT:
        raise ValueError(
            f"Expected {TRANSCRIPTION_REQUEST_COUNT} transcription inputs, found {len(items)}"
        )
    return items


def build_workload_plan() -> dict[str, Any]:
    text_items = _build_text_items()
    moderation_inputs = _build_moderation_inputs()
    image_prompts = _build_image_prompts()
    speech_inputs = _build_speech_inputs()
    return {
        "planned_request_count": PLANNED_REQUEST_COUNT,
        "segments": dict(PLANNED_SEGMENTS),
        "video_support": dict(VIDEO_SUPPORT),
        "text_items": text_items,
        "moderation_inputs": moderation_inputs,
        "image_prompts": image_prompts,
        "speech_inputs": speech_inputs,
    }


def _provider_coverage() -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    matrix = provider_capability_matrix()
    for provider, caps in sorted(matrix.items()):
        segment_support = {
            "text_core": bool(
                caps.get("chat_completion")
                and caps.get("coding_tasks")
                and caps.get("document_tasks")
            ),
            "moderation": bool(caps.get("moderation")),
            "image_generation": bool(caps.get("image_generation")),
            "speech_generation": bool(caps.get("speech_generation")),
            "audio_transcription": bool(caps.get("audio_transcription")),
            "video": False,
        }
        coverage[provider] = {
            "segments": segment_support,
            "live_supported_request_count": sum(
                PLANNED_SEGMENTS[name] for name, enabled in segment_support.items() if enabled
            ),
        }
    return coverage


def _run_text_workload(api_key: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    _, direct_summary = prompt_stress._run_direct_sequence(api_key, items)  # pylint: disable=protected-access
    prompt_stress._apply_baseline(direct_summary, direct_summary)  # pylint: disable=protected-access
    runs = {"direct": direct_summary}
    for mode in prompt_stress.BYTE_MODES:
        _, summary = prompt_stress._run_byte_sequence(api_key, items, mode=mode)  # pylint: disable=protected-access
        prompt_stress._apply_baseline(summary, direct_summary)  # pylint: disable=protected-access
        runs[mode] = summary
    return {
        "request_count": len(items),
        "waves": TEXT_WAVES,
        "composition": prompt_stress._bucket_counts(items),  # pylint: disable=protected-access
        "runs": runs,
    }


def _run_moderation_workload(api_key: str, inputs: list[str]) -> dict[str, Any]:
    runs = {
        "direct": surface._summarize_media_records(  # pylint: disable=protected-access
            [surface._direct_moderation_request(api_key, item) for item in inputs]  # pylint: disable=protected-access
        )
    }
    for mode in surface.MEDIA_MODES:
        runs[mode] = surface._run_byte_media_sequence(  # pylint: disable=protected-access
            inputs,
            mode=mode,
            scenario_name="mixed_1000_moderation",
            pre_func=get_openai_moderation_input,
            normalized_pre_func=surface._normalized_input_text,  # pylint: disable=protected-access
            run_request=lambda cache_obj, prompt: surface._byte_moderation_request(
                api_key, cache_obj, prompt
            ),  # pylint: disable=protected-access
        )
    summary = surface._media_scenario_summary(  # pylint: disable=protected-access
        "moderation_variants_80",
        "Repeated harmful moderation traffic with punctuation-only variation to validate exact and normalized reuse at larger scale.",
        runs,
    )
    summary["request_count"] = len(inputs)
    summary["executed_request_count"] = len(inputs)
    return summary


def _run_image_workload(api_key: str, prompts: list[str]) -> dict[str, Any]:
    model, error = surface._choose_working_image_model(api_key, prompts[0])  # pylint: disable=protected-access
    if not model:
        return {
            "name": "image_generation_variants_40",
            "description": "Image generation benchmark using punctuation-only prompt variation.",
            "request_count": len(prompts),
            "executed_request_count": 0,
            "unavailable": error,
        }

    runs = {
        "direct": surface._summarize_media_records(  # pylint: disable=protected-access
            [surface._direct_image_request(api_key, prompt, model) for prompt in prompts]  # pylint: disable=protected-access
        )
    }
    for mode in surface.MEDIA_MODES:
        runs[mode] = surface._run_byte_media_sequence(  # pylint: disable=protected-access
            prompts,
            mode=mode,
            scenario_name="mixed_1000_image_generation",
            pre_func=get_prompt,
            normalized_pre_func=normalized_get_prompt,
            run_request=lambda cache_obj, prompt: surface._byte_image_request(
                api_key, cache_obj, prompt, model
            ),  # pylint: disable=protected-access
        )
    summary = surface._media_scenario_summary(  # pylint: disable=protected-access
        "image_generation_variants_40",
        "Image generation with prompt-variant reuse extended to forty requests across multiple icon prompts.",
        runs,
    )
    summary["model"] = model
    summary["request_count"] = len(prompts)
    summary["executed_request_count"] = len(prompts)
    return summary


def _run_speech_and_transcription_workloads(
    api_key: str, inputs: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    speech_model, speech_error = surface._choose_working_speech_model(api_key, inputs[0])  # pylint: disable=protected-access
    if not speech_model:
        speech_summary = {
            "name": "speech_generation_variants_40",
            "description": "Speech generation benchmark using punctuation-only text variation.",
            "request_count": len(inputs),
            "executed_request_count": 0,
            "unavailable": speech_error,
        }
        transcription_summary = {
            "name": "audio_transcription_exact_40",
            "description": "Audio transcription of a generated speech fixture repeated forty times.",
            "request_count": TRANSCRIPTION_REQUEST_COUNT,
            "executed_request_count": 0,
            "unavailable": f"Speech model unavailable, so no audio fixture could be generated. {speech_error}",
        }
        return speech_summary, transcription_summary

    speech_runs = {
        "direct": surface._summarize_media_records(  # pylint: disable=protected-access
            [surface._direct_speech_request(api_key, text, speech_model) for text in inputs]  # pylint: disable=protected-access
        )
    }
    for mode in surface.MEDIA_MODES:
        speech_runs[mode] = surface._run_byte_media_sequence(  # pylint: disable=protected-access
            inputs,
            mode=mode,
            scenario_name="mixed_1000_speech_generation",
            pre_func=surface._input_text,  # pylint: disable=protected-access
            normalized_pre_func=surface._normalized_input_text,  # pylint: disable=protected-access
            run_request=lambda cache_obj, prompt: surface._byte_speech_request(
                api_key, cache_obj, prompt, speech_model
            ),  # pylint: disable=protected-access
        )
    speech_summary = surface._media_scenario_summary(  # pylint: disable=protected-access
        "speech_generation_variants_40",
        "Speech generation with punctuation-only text variation scaled to forty requests.",
        speech_runs,
    )
    speech_summary["model"] = speech_model
    speech_summary["request_count"] = len(inputs)
    speech_summary["executed_request_count"] = len(inputs)

    audio_path, audio_error = surface._write_audio_fixture(api_key, speech_model, inputs[0])  # pylint: disable=protected-access
    if not audio_path:
        transcription_summary = {
            "name": "audio_transcription_exact_40",
            "description": "Audio transcription of a generated speech fixture repeated forty times.",
            "request_count": TRANSCRIPTION_REQUEST_COUNT,
            "executed_request_count": 0,
            "unavailable": audio_error,
        }
        return speech_summary, transcription_summary

    try:
        transcribe_model, transcribe_error = surface._choose_working_transcribe_model(
            api_key, audio_path
        )  # pylint: disable=protected-access
        if not transcribe_model:
            transcription_summary = {
                "name": "audio_transcription_exact_40",
                "description": "Audio transcription of a generated speech fixture repeated forty times.",
                "request_count": TRANSCRIPTION_REQUEST_COUNT,
                "executed_request_count": 0,
                "unavailable": transcribe_error,
            }
            return speech_summary, transcription_summary

        transcription_inputs = _build_transcription_inputs(audio_path)
        transcription_runs = {
            "direct": surface._summarize_media_records(  # pylint: disable=protected-access
                [
                    surface._direct_transcribe_request(api_key, path, transcribe_model)
                    for path in transcription_inputs
                ]  # pylint: disable=protected-access
            ),
            "exact": surface._run_byte_media_sequence(  # pylint: disable=protected-access
                transcription_inputs,
                mode="exact",
                scenario_name="mixed_1000_audio_transcription",
                pre_func=get_file_bytes,
                normalized_pre_func=None,
                run_request=lambda cache_obj, path: surface._byte_transcribe_request(
                    api_key, cache_obj, path, transcribe_model
                ),  # pylint: disable=protected-access
            ),
        }
        transcription_summary = surface._media_scenario_summary(  # pylint: disable=protected-access
            "audio_transcription_exact_40",
            "Transcription of the same generated speech fixture forty times to measure large-scale exact audio reuse.",
            transcription_runs,
        )
        transcription_summary["model"] = transcribe_model
        transcription_summary["request_count"] = len(transcription_inputs)
        transcription_summary["executed_request_count"] = len(transcription_inputs)
        return speech_summary, transcription_summary
    finally:
        with suppress(OSError):
            os.remove(audio_path)
        with suppress(OSError):
            os.rmdir(os.path.dirname(audio_path))


def _executed_request_count(results: dict[str, Any]) -> int:
    executed = int(results["text_workload"]["request_count"])
    for media in results["media_workloads"].values():
        executed += int(media.get("executed_request_count", 0) or 0)
    return executed


def _render_provider_coverage(coverage: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    ordered = [
        provider
        for provider in (
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
        )
        if provider in coverage
    ]
    ordered.extend(sorted(provider for provider in coverage if provider not in ordered))
    for provider in ordered:
        data = coverage[provider]
        segments = data["segments"]
        lines.append(
            f"- `{provider}`: live_supported_requests={data['live_supported_request_count']}, "
            f"text_core={segments['text_core']}, moderation={segments['moderation']}, "
            f"image_generation={segments['image_generation']}, speech_generation={segments['speech_generation']}, "
            f"audio_transcription={segments['audio_transcription']}, video={segments['video']}"
        )
    return lines


def _render_media_section(title: str, summary: dict[str, Any], *, modes: list[str]) -> list[str]:
    lines = [f"## {title}", "", f"- {summary['description']}"]
    if summary.get("unavailable"):
        lines.append(f"- Unavailable in this run: {summary['unavailable']}")
        lines.append("")
        return lines
    if summary.get("model"):
        lines.append(f"- Working model: `{summary['model']}`")
    lines.append(surface._render_media_mode_line("Direct", summary["runs"]["direct"]))  # pylint: disable=protected-access
    for mode in modes:
        if mode in summary["runs"]:
            lines.append(
                surface._render_media_mode_line(f"ByteAI Cache {mode}", summary["runs"][mode])
            )  # pylint: disable=protected-access
    lines.append("")
    return lines


def _render_bucket_mode_line(label: str, data: dict[str, Any]) -> str:
    return (
        f"- {label}: cost={prompt_stress._render_money(float(data.get('total_cost_usd', 0.0) or 0.0))}, "  # pylint: disable=protected-access
        f"hit_ratio={data.get('hit_ratio', 0.0)}, accuracy={data.get('accuracy_ratio', 0.0)}, "
        f"avg_latency={data.get('avg_latency_ms', 0.0)} ms, p95={data.get('p95_latency_ms', 0.0)} ms"
    )


def _render_report(results: dict[str, Any]) -> str:
    text_workload = results["text_workload"]
    lines: list[str] = [
        "# ByteAI Cache OpenAI 1000-Request Mixed Benchmark",
        "",
        f"Generated: {results['generated_at']}",
        f"Planned requests: {results['planned_request_count']}",
        f"Executed requests: {results['executed_request_count']}",
        f"Chat model: `{results['chat_model']}`",
        "",
        "## Workload Composition",
        "",
        f"- Text core requests: {TEXT_REQUEST_COUNT} across {TEXT_WAVES} waves of the existing 100-request support/document/coding/context stress mix.",
        f"- Moderation requests: {MODERATION_REQUEST_COUNT}",
        f"- Image generation requests: {IMAGE_REQUEST_COUNT}",
        f"- Speech generation requests: {SPEECH_REQUEST_COUNT}",
        f"- Audio transcription requests: {TRANSCRIPTION_REQUEST_COUNT}",
        f"- Video: {results['video_support']['status']} ({results['video_support']['reason']})",
        f"- Text bucket composition: {text_workload['composition']}",
        "",
        "## Adapter Coverage",
        "",
        "The workload plan is shared across adapters. Live execution below uses OpenAI only, but the coverage table shows how much of the same 1000-request plan each built-in adapter can exercise with its current surface support.",
        "",
    ]
    lines.extend(_render_provider_coverage(results["provider_coverage"]))
    lines.extend(
        [
            "",
            "## Text/Core Workload",
            "",
        ]
    )
    lines.append(prompt_stress._render_mode_line("Direct", text_workload["runs"]["direct"]))  # pylint: disable=protected-access
    for mode in prompt_stress.BYTE_MODES:
        lines.append(
            prompt_stress._render_mode_line(f"ByteAI Cache {mode}", text_workload["runs"][mode])
        )  # pylint: disable=protected-access
    lines.extend(
        [
            "",
            "## Text Detail Table",
            "",
            "| Mode | Cost (USD) | Savings | Hit Ratio | Accuracy | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Prompt Tokens | Prompt Reduction | Throughput (RPS) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode_name in ["direct", *prompt_stress.BYTE_MODES]:
        data = text_workload["runs"][mode_name]
        lines.append(
            f"| {mode_name} | {prompt_stress._render_money(float(data.get('total_cost_usd', 0.0) or 0.0))} | {data.get('savings_ratio', 0.0)} | "  # pylint: disable=protected-access
            f"{data.get('hit_ratio', 0.0)} | {data.get('accuracy_ratio', 0.0)} | {data.get('avg_latency_ms', 0.0)} | "
            f"{data.get('p50_latency_ms', 0.0)} | {data.get('p95_latency_ms', 0.0)} | {data.get('p99_latency_ms', 0.0)} | "
            f"{data.get('total_prompt_tokens', 0)} | {data.get('prompt_token_reduction_ratio', 0.0)} | {data.get('throughput_rps', 0.0)} |"
        )
    plain_unique = {
        mode: text_workload["runs"][mode]["bucket_breakdown"]["plain_unique"]
        for mode in ["direct", *prompt_stress.BYTE_MODES]
    }
    shared_context_unique = {
        mode: text_workload["runs"][mode]["bucket_breakdown"]["shared_context_unique"]
        for mode in ["direct", *prompt_stress.BYTE_MODES]
    }
    shared_context_hybrid_savings = 0.0
    shared_direct_cost = float(shared_context_unique["direct"].get("total_cost_usd", 0.0) or 0.0)
    shared_hybrid_cost = float(shared_context_unique["hybrid"].get("total_cost_usd", 0.0) or 0.0)
    if shared_direct_cost:
        shared_context_hybrid_savings = round(
            (shared_direct_cost - shared_hybrid_cost) / shared_direct_cost, 4
        )
    lines.extend(
        [
            "",
            "## Unique Prompt Savings",
            "",
            "### plain_unique",
            "",
            "- One-off exact-output prompts keep simple unique requests in the benchmark so savings are not driven only by repeated coding or support traffic.",
            _render_bucket_mode_line("Direct", plain_unique["direct"]),
            _render_bucket_mode_line("ByteAI Cache exact", plain_unique["exact"]),
            _render_bucket_mode_line("ByteAI Cache normalized", plain_unique["normalized"]),
            _render_bucket_mode_line("ByteAI Cache hybrid", plain_unique["hybrid"]),
            "",
            "### shared_context_unique",
            "",
            "- This is the explicit unique-prompt savings feature: every request asks for a new token, but the repo, retrieval, document, and support context repeats so ByteAI can save prompt cost without answer-cache hits.",
            _render_bucket_mode_line("Direct", shared_context_unique["direct"]),
            _render_bucket_mode_line("ByteAI Cache exact", shared_context_unique["exact"]),
            _render_bucket_mode_line(
                "ByteAI Cache normalized", shared_context_unique["normalized"]
            ),
            _render_bucket_mode_line("ByteAI Cache hybrid", shared_context_unique["hybrid"]),
            f"- Hybrid shared-context savings vs direct: {shared_context_hybrid_savings}",
            "",
            "## Text Bucket Breakdown",
            "",
        ]
    )
    lines.extend(
        prompt_stress._render_breakdown_table(
            "Direct By Bucket", text_workload["runs"]["direct"]["bucket_breakdown"]
        )
    )  # pylint: disable=protected-access
    lines.extend(
        prompt_stress._render_breakdown_table(
            "Exact By Bucket", text_workload["runs"]["exact"]["bucket_breakdown"]
        )
    )  # pylint: disable=protected-access
    lines.extend(
        prompt_stress._render_breakdown_table(
            "Normalized By Bucket", text_workload["runs"]["normalized"]["bucket_breakdown"]
        )
    )  # pylint: disable=protected-access
    lines.extend(
        prompt_stress._render_breakdown_table(
            "Hybrid By Bucket", text_workload["runs"]["hybrid"]["bucket_breakdown"]
        )
    )  # pylint: disable=protected-access
    lines.extend(
        prompt_stress._render_breakdown_table(
            "Hybrid By Kind", text_workload["runs"]["hybrid"]["kind_breakdown"]
        )
    )  # pylint: disable=protected-access
    lines.extend(
        [
            "## Cache Behavior",
            "",
            f"- Hybrid hit reasons: {text_workload['runs']['hybrid'].get('byte_reason_counts', {})}",
            f"- Normalized hit reasons: {text_workload['runs']['normalized'].get('byte_reason_counts', {})}",
            f"- Exact hit reasons: {text_workload['runs']['exact'].get('byte_reason_counts', {})}",
            "",
        ]
    )
    lines.extend(
        _render_media_section(
            "Moderation", results["media_workloads"]["moderation"], modes=["exact", "normalized"]
        )
    )
    lines.extend(
        _render_media_section(
            "Image Generation",
            results["media_workloads"]["image_generation"],
            modes=["exact", "normalized"],
        )
    )
    lines.extend(
        _render_media_section(
            "Speech Generation",
            results["media_workloads"]["speech_generation"],
            modes=["exact", "normalized"],
        )
    )
    lines.extend(
        _render_media_section(
            "Audio Transcription", results["media_workloads"]["transcription"], modes=["exact"]
        )
    )
    lines.extend(
        [
            "## Notes",
            "",
            "- The 800-request text core intentionally extends the prior 100-request prompt stress benchmark into a longer-lived session. Exact and normalized reuse therefore compound across waves instead of resetting every 100 requests.",
            "- The shared workload plan is adapter-aware through the capability matrix, but only OpenAI is executed live in this run because it is the only provider key available.",
            "- Video remains a tracked gap rather than a silently skipped benchmark surface. The parser can normalize video parts, but there is no live cross-adapter runner to execute yet.",
            "- Image, speech, and transcription model probing happens before the measured workload starts so the measured request counts remain exactly 1000 when the surfaces are available.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a shared-plan 1000-request mixed benchmark and execute it live with OpenAI."
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key. Falls back to BYTE_TEST_OPENAI_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--json-report", default=str(DEFAULT_JSON_REPORT))
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("BYTE_TEST_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set BYTE_TEST_OPENAI_API_KEY or pass --api-key.")

    plan = build_workload_plan()
    text_workload = _run_text_workload(api_key, plan["text_items"])
    moderation_summary = _run_moderation_workload(api_key, plan["moderation_inputs"])
    image_summary = _run_image_workload(api_key, plan["image_prompts"])
    speech_summary, transcription_summary = _run_speech_and_transcription_workloads(
        api_key, plan["speech_inputs"]
    )

    results = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "planned_request_count": plan["planned_request_count"],
        "executed_request_count": 0,
        "chat_model": CHAT_MODEL,
        "video_support": plan["video_support"],
        "provider_coverage": _provider_coverage(),
        "artifacts": {
            "report": str(Path(args.report).resolve()),
            "json_report": str(Path(args.json_report).resolve()),
        },
        "text_workload": text_workload,
        "media_workloads": {
            "moderation": moderation_summary,
            "image_generation": image_summary,
            "speech_generation": speech_summary,
            "transcription": transcription_summary,
        },
    }
    results["executed_request_count"] = _executed_request_count(results)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results), encoding="utf-8")

    json_path = Path(args.json_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
