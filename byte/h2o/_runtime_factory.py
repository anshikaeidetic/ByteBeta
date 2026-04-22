
"""Cached factory for Hugging Face H2O runtime instances."""

from __future__ import annotations

from typing import Any

from byte.adapter.client_pool import get_sync_client
from byte.h2o._runtime_runtime import H2ORuntime


def get_huggingface_runtime(
    model_name: str,
    *,
    tokenizer_name: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    device: str | None = None,
    device_map: Any | None = None,
    torch_dtype: Any | None = None,
    local_files_only: bool = False,
    attn_implementation: str | None = None,
) -> H2ORuntime:
    return get_sync_client(
        "huggingface_runtime",
        _create_runtime,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )


def _create_runtime(**kwargs) -> H2ORuntime:
    return H2ORuntime(**kwargs)

__all__ = ["get_huggingface_runtime"]
