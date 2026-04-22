
"""Runtime object construction for Hugging Face H2O execution."""

from __future__ import annotations

from typing import Any

from byte.h2o._runtime_common import _torch, _transformers
from byte.h2o._runtime_generation import H2OGenerationMixin
from byte.h2o._runtime_tokens import (
    _infer_model_family,
    _resolve_model_device,
    _resolve_torch_dtype,
)


class H2ORuntime(H2OGenerationMixin):
    def __init__(
        self,
        *,
        model_name: str,
        tokenizer_name: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        device: str | None = None,
        device_map: Any | None = None,
        torch_dtype: Any | None = None,
        local_files_only: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        torch = _torch()
        AutoModelForCausalLM, AutoTokenizer = _transformers()
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        dtype = _resolve_torch_dtype(torch_dtype)
        model_kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if device_map not in (None, ""):
            model_kwargs["device_map"] = device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device and device_map in (None, ""):
            model = model.to(device)
        model.eval()
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.tokenizer = tokenizer
        self.model = model
        self.model_family = _infer_model_family(model)
        self.device = _resolve_model_device(model, torch)

__all__ = ["H2ORuntime"]
