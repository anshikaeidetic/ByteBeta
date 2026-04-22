import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from byte.adapter.api import provider_capability_matrix  # pylint: disable=wrong-import-position
from byte.benchmarking._program_defaults import load_program_defaults
from byte.benchmarking.programs import deep_openai_prompt_stress_benchmark as prompt_stress
from byte.benchmarking.programs import deep_openai_surface_benchmark as surface
from byte.processor.pre import (  # pylint: disable=wrong-import-position
    get_file_bytes,
    get_openai_moderation_input,
    get_prompt,
    normalized_get_prompt,
)

REPORT_DIR = REPO_ROOT / "docs" / "reports"
DEFAULT_REPORT = REPORT_DIR / "openai_100_request_mixed_benchmark.md"
DEFAULT_JSON_REPORT = REPORT_DIR / "openai_100_request_mixed_benchmark.json"

_DEFAULTS = load_program_defaults("deep_openai_100_request_mixed_benchmark")

TEXT_REQUEST_COUNT = int(_DEFAULTS.get("text_request_count", 60))
MODERATION_REQUEST_COUNT = int(_DEFAULTS.get("moderation_request_count", 10))
IMAGE_REQUEST_COUNT = int(_DEFAULTS.get("image_request_count", 10))
SPEECH_REQUEST_COUNT = int(_DEFAULTS.get("speech_request_count", 10))
TRANSCRIPTION_REQUEST_COUNT = int(_DEFAULTS.get("transcription_request_count", 10))
PLANNED_REQUEST_COUNT = (
    TEXT_REQUEST_COUNT
    + MODERATION_REQUEST_COUNT
    + IMAGE_REQUEST_COUNT
    + SPEECH_REQUEST_COUNT
    + TRANSCRIPTION_REQUEST_COUNT
)

VIDEO_SUPPORT = dict(_DEFAULTS.get("video_support", {}))

PLANNED_SEGMENTS = {
    "text_core": TEXT_REQUEST_COUNT,
    "moderation": MODERATION_REQUEST_COUNT,
    "image_generation": IMAGE_REQUEST_COUNT,
    "speech_generation": SPEECH_REQUEST_COUNT,
    "audio_transcription": TRANSCRIPTION_REQUEST_COUNT,
    "video": 0,
}


def _clone_item(item: dict[str, Any], *, variant: str) -> dict[str, Any]:
    cloned = copy.deepcopy(item)
    cloned["variant"] = variant
    return cloned


def _take_items(items: list[dict[str, Any]], bucket_name: str, count: int) -> list[dict[str, Any]]:
    selected = []
    for index, item in enumerate(items[:count], 1):
        selected.append(_clone_item(item, variant=f"{bucket_name}::{index:02d}"))
    return selected


def _build_text_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    items.extend(_take_items(prompt_stress._build_exact_repeat_bucket(), "exact_repeat", 12))  # pylint: disable=protected-access
    items.extend(
        _take_items(prompt_stress._build_normalized_variant_bucket(), "normalized_variant", 12)
    )  # pylint: disable=protected-access
    items.extend(_take_items(prompt_stress._build_coding_mixed_bucket(), "coding_mixed", 12))  # pylint: disable=protected-access
    items.extend(_take_items(prompt_stress._build_plain_unique_bucket(), "plain_unique", 12))  # pylint: disable=protected-access
    items.extend(
        _take_items(
            prompt_stress._build_shared_context_unique_bucket(), "shared_context_unique", 12
        )
    )  # pylint: disable=protected-access
    if len(items) != TEXT_REQUEST_COUNT:
        raise ValueError(f"Expected {TEXT_REQUEST_COUNT} text items, found {len(items)}")
    return items


def _build_moderation_inputs() -> list[str]:
    inputs = []
    pair = ("I want to kill them.", "I want to kill them!")
    for _ in range(MODERATION_REQUEST_COUNT // 2):
        inputs.extend(pair)
    return inputs


def _build_image_prompts() -> list[str]:
    prompts = [
        "Create a clean flat icon of a blue whale on a white background.",
        "Create a clean flat icon of a blue whale, on a white background.",
        "Create a clean flat icon of a red lighthouse on a white background.",
        "Create a clean flat icon of a red lighthouse, on a white background.",
        "Create a clean flat icon of a green cactus on a white background.",
        "Create a clean flat icon of a green cactus, on a white background.",
        "Create a clean flat icon of a yellow submarine on a white background.",
        "Create a clean flat icon of a yellow submarine, on a white background.",
        "Create a clean flat icon of an orange fox on a white background.",
        "Create a clean flat icon of an orange fox, on a white background.",
    ]
    return prompts


def _build_speech_inputs() -> list[str]:
    return [
        "Byte speeds up repeated AI requests for support, coding, and docs.",
        "Byte speeds up repeated AI requests for support, coding and docs.",
        "Byte keeps prompt reuse deterministic for support and workflow automation.",
        "Byte keeps prompt reuse deterministic for support and workflow automation!",
        "Byte reduces repeated coding assistant traffic across large support queues.",
        "Byte reduces repeated coding assistant traffic across large support queues!",
        "Byte compiles shared context to keep document and coding prompts smaller.",
        "Byte compiles shared context to keep document and coding prompts smaller!",
        "Byte remembers reusable artifacts for docs, support, and code reviews.",
        "Byte remembers reusable artifacts for docs, support, and code reviews!",
    ]


def _build_transcription_inputs(audio_path: str) -> list[str]:
    return [audio_path for _ in range(TRANSCRIPTION_REQUEST_COUNT)]


def build_workload_plan() -> dict[str, Any]:
    return {
        "planned_request_count": PLANNED_REQUEST_COUNT,
        "segments": dict(PLANNED_SEGMENTS),
        "video_support": dict(VIDEO_SUPPORT),
        "text_items": _build_text_items(),
        "moderation_inputs": _build_moderation_inputs(),
        "image_prompts": _build_image_prompts(),
        "speech_inputs": _build_speech_inputs(),
    }


def provider_coverage() -> dict[str, dict[str, Any]]:
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
            scenario_name="mixed_100_moderation",
            pre_func=get_openai_moderation_input,
            normalized_pre_func=surface._normalized_input_text,  # pylint: disable=protected-access
            run_request=lambda cache_obj, prompt: surface._byte_moderation_request(
                api_key, cache_obj, prompt
            ),  # pylint: disable=protected-access
        )
    summary = surface._media_scenario_summary(  # pylint: disable=protected-access
        "moderation_variants_10",
        "Repeated harmful moderation traffic with punctuation-only variation.",
        runs,
    )
    summary["request_count"] = len(inputs)
    summary["executed_request_count"] = len(inputs)
    return summary


def _run_image_workload(api_key: str, prompts: list[str]) -> dict[str, Any]:
    model, error = surface._choose_working_image_model(api_key, prompts[0])  # pylint: disable=protected-access
    if not model:
        return {
            "name": "image_generation_variants_10",
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
            scenario_name="mixed_100_image",
            pre_func=get_prompt,
            normalized_pre_func=normalized_get_prompt,
            run_request=lambda cache_obj, prompt: surface._byte_image_request(
                api_key, cache_obj, prompt, model
            ),  # pylint: disable=protected-access
        )
    summary = surface._media_scenario_summary(  # pylint: disable=protected-access
        "image_generation_variants_10",
        "Image generation benchmark using punctuation-only prompt variation.",
        runs,
    )
    summary["request_count"] = len(prompts)
    summary["executed_request_count"] = len(prompts)
    summary["model"] = model
    return summary


def _run_speech_workload(api_key: str, inputs: list[str]) -> dict[str, Any]:
    model, error = surface._choose_working_speech_model(api_key, inputs[0])  # pylint: disable=protected-access
    if not model:
        return {
            "name": "speech_generation_variants_10",
            "description": "Speech generation benchmark using punctuation-only text variation.",
            "request_count": len(inputs),
            "executed_request_count": 0,
            "unavailable": error,
        }
    runs = {
        "direct": surface._summarize_media_records(  # pylint: disable=protected-access
            [surface._direct_speech_request(api_key, text, model) for text in inputs]  # pylint: disable=protected-access
        )
    }
    for mode in surface.MEDIA_MODES:
        runs[mode] = surface._run_byte_media_sequence(  # pylint: disable=protected-access
            inputs,
            mode=mode,
            scenario_name="mixed_100_speech",
            pre_func=surface._input_text,  # pylint: disable=protected-access
            normalized_pre_func=surface._normalized_input_text,  # pylint: disable=protected-access
            run_request=lambda cache_obj, text: surface._byte_speech_request(
                api_key, cache_obj, text, model
            ),  # pylint: disable=protected-access
        )
    summary = surface._media_scenario_summary(  # pylint: disable=protected-access
        "speech_generation_variants_10",
        "Speech generation benchmark using punctuation-only text variation.",
        runs,
    )
    summary["model"] = model
    summary["request_count"] = len(inputs)
    summary["executed_request_count"] = len(inputs)
    return summary


def _run_transcription_workload(api_key: str, audio_path: str) -> dict[str, Any]:
    inputs = _build_transcription_inputs(audio_path)
    model, error = surface._choose_working_transcribe_model(api_key, audio_path)  # pylint: disable=protected-access
    if not model:
        return {
            "name": "audio_transcription_exact_10",
            "description": "Audio transcription benchmark using repeated local audio inputs.",
            "request_count": len(inputs),
            "executed_request_count": 0,
            "unavailable": error,
        }
    runs = {
        "direct": surface._summarize_media_records(  # pylint: disable=protected-access
            [surface._direct_transcribe_request(api_key, item, model) for item in inputs]  # pylint: disable=protected-access
        )
    }
    runs["exact"] = surface._run_byte_media_sequence(  # pylint: disable=protected-access
        inputs,
        mode="exact",
        scenario_name="mixed_100_transcription",
        pre_func=get_file_bytes,
        normalized_pre_func=None,
        run_request=lambda cache_obj, item: surface._byte_transcribe_request(
            api_key, cache_obj, item, model
        ),  # pylint: disable=protected-access
    )
    summary = surface._media_scenario_summary(  # pylint: disable=protected-access
        "audio_transcription_variants_10",
        "Audio transcription benchmark using repeated local audio inputs.",
        runs,
    )
    summary["model"] = model
    summary["request_count"] = len(inputs)
    summary["executed_request_count"] = len(inputs)
    return summary


def run_benchmark(api_key: str, *, audio_path: str) -> dict[str, Any]:
    plan = build_workload_plan()
    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "planned_request_count": plan["planned_request_count"],
        "segments": plan["segments"],
        "video_support": plan["video_support"],
        "provider_coverage": provider_coverage(),
        "text_core": _run_text_workload(api_key, plan["text_items"]),
        "moderation": _run_moderation_workload(api_key, plan["moderation_inputs"]),
        "image_generation": _run_image_workload(api_key, plan["image_prompts"]),
        "speech_generation": _run_speech_workload(api_key, plan["speech_inputs"]),
        "audio_transcription": _run_transcription_workload(api_key, audio_path),
    }
    result["executed_request_count"] = sum(
        int(result[name].get("executed_request_count", result[name].get("request_count", 0)) or 0)
        for name in (
            "text_core",
            "moderation",
            "image_generation",
            "speech_generation",
            "audio_transcription",
        )
    )
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator or 0.0)
    if denominator == 0.0:
        return 0.0
    return float(numerator or 0.0) / denominator


def _render_report(result: dict[str, Any]) -> str:
    text_direct = result["text_core"]["runs"]["direct"]
    text_hybrid = result["text_core"]["runs"]["hybrid"]
    text_savings = _safe_ratio(
        float(text_direct.get("total_cost_usd", 0.0) or 0.0)
        - float(text_hybrid.get("total_cost_usd", 0.0) or 0.0),
        float(text_direct.get("total_cost_usd", 0.0) or 0.0),
    )
    text_prompt_reduction = _safe_ratio(
        float(text_direct.get("total_prompt_tokens", 0.0) or 0.0)
        - float(text_hybrid.get("total_prompt_tokens", 0.0) or 0.0),
        float(text_direct.get("total_prompt_tokens", 0.0) or 0.0),
    )
    shared_context_direct = text_direct["bucket_breakdown"]["shared_context_unique"]
    shared_context_hybrid = text_hybrid["bucket_breakdown"]["shared_context_unique"]
    plain_unique_hybrid = text_hybrid["bucket_breakdown"]["plain_unique"]
    shared_context_savings = _safe_ratio(
        float(shared_context_direct.get("total_cost_usd", 0.0) or 0.0)
        - float(shared_context_hybrid.get("total_cost_usd", 0.0) or 0.0),
        float(shared_context_direct.get("total_cost_usd", 0.0) or 0.0),
    )
    lines = [
        "# OpenAI 100-Request Mixed Benchmark",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Planned requests: `{result['planned_request_count']}`",
        f"- Executed requests: `{result['executed_request_count']}`",
        f"- Hybrid text-core savings vs direct: `{text_savings:.4f}`",
        f"- Hybrid text-core accuracy: `{float(text_hybrid.get('accuracy_ratio', 0.0) or 0.0):.4f}`",
        f"- Hybrid text-core prompt-token reduction: `{text_prompt_reduction:.4f}`",
        f"- Hybrid text-core average latency: `{float(text_hybrid.get('avg_latency_ms', 0.0) or 0.0):.2f} ms`",
        "",
        "## Segment Plan",
        "",
    ]
    for name, count in result["segments"].items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(
        [
            "",
            "## Video",
            "",
            f"- Status: `{result['video_support']['status']}`",
            f"- Reason: {result['video_support']['reason']}",
            "",
            "## Provider Coverage",
            "",
        ]
    )
    for provider, coverage in sorted(result["provider_coverage"].items()):
        lines.append(
            f"- `{provider}`: `{coverage['live_supported_request_count']}` planned live requests"
        )
    lines.extend(
        [
            "",
            "## Unique Prompt Savings",
            "",
            f"- `shared_context_unique` hybrid savings vs direct: `{shared_context_savings:.4f}`",
            f"- `shared_context_unique` hybrid hit ratio: `{float(shared_context_hybrid.get('hit_ratio', 0.0) or 0.0):.4f}`",
            f"- `plain_unique` hybrid accuracy: `{float(plain_unique_hybrid.get('accuracy_ratio', 0.0) or 0.0):.4f}`",
            f"- `plain_unique` hybrid hit ratio: `{float(plain_unique_hybrid.get('hit_ratio', 0.0) or 0.0):.4f}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-key", default=os.getenv("BYTE_TEST_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--json-report", default=str(DEFAULT_JSON_REPORT))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "OpenAI API key required via --api-key or OPENAI_API_KEY/BYTE_TEST_OPENAI_API_KEY."
        )

    result = run_benchmark(args.api_key, audio_path=args.audio_path)
    report_path = Path(args.report)
    json_path = Path(args.json_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(str(report_path))
    print(str(json_path))


if __name__ == "__main__":
    main()
