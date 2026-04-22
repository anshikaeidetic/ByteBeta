"""Public prompt-distillation facade."""

from byte.prompt_distillation._distillation_common import (
    PromptDistillationResult,
    PromptModuleRegistry,
    normalize_text,
)
from byte.prompt_distillation._distillation_engine import (
    distill_request_payload,
    export_prompt_distillation_manifest,
)
from byte.prompt_distillation._distillation_faithfulness import verify_request_faithfulness
from byte.prompt_distillation._distillation_measure import measure_request_prompt

__all__ = [
    "PromptDistillationResult",
    "PromptModuleRegistry",
    "distill_request_payload",
    "export_prompt_distillation_manifest",
    "measure_request_prompt",
    "normalize_text",
    "verify_request_faithfulness",
]
