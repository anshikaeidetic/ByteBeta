"""Context compilation facade for Byte preprocessing."""

from byte.processor._optimization_summary import (
    compact_text,
    distill_artifact_for_focus,
    extract_prompt_pieces,
    stable_digest,
    summarize_artifact_payload,
    summarize_artifact_sketch,
)
from byte.processor._pre_context_runtime import compile_request_context
from byte.prompt_distillation import (
    PromptDistillationResult,
    distill_request_payload,
    measure_request_prompt,
    verify_request_faithfulness,
)
from byte.utils.multimodal import content_signature

__all__ = [
    "PromptDistillationResult",
    "compact_text",
    "compile_request_context",
    "content_signature",
    "distill_artifact_for_focus",
    "distill_request_payload",
    "extract_prompt_pieces",
    "measure_request_prompt",
    "stable_digest",
    "summarize_artifact_payload",
    "summarize_artifact_sketch",
    "verify_request_faithfulness",
]
