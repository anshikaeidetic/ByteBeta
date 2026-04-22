"""Worker routing helpers for the Byte control plane."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


def _short_digest(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]


def _model_family(model_name: str) -> str:
    lowered = str(model_name or "").strip().lower()
    if "/" in lowered:
        _, _, lowered = lowered.partition("/")
    for token in ("llama", "mistral", "gemma", "qwen", "phi", "gpt", "claude"):
        if token in lowered:
            return token
    return lowered.split("-", 1)[0] if lowered else ""


def _compression_profile(request_kwargs: dict[str, Any] | None) -> str:
    request_kwargs = request_kwargs or {}
    codec = str(
        request_kwargs.get("byte_kv_codec")
        or request_kwargs.get("kv_codec")
        or request_kwargs.get("byte_compression_mode")
        or "disabled"
    ).strip()
    bits = int(request_kwargs.get("byte_kv_bits") or request_kwargs.get("kv_bits") or 8)
    h2o = bool(
        request_kwargs.get("byte_h2o_enabled")
        or request_kwargs.get("h2o_enabled")
        or False
    )
    return f"{codec}:{bits}:h2o={int(h2o)}"


def _worker_supports_model(worker: dict[str, Any], model_name: str) -> bool:
    inventory = [
        str(item).strip()
        for item in (worker.get("model_inventory") or [])
        if str(item).strip()
    ]
    if not inventory:
        return True
    normalized_model = str(model_name or "").strip().lower()
    for item in inventory:
        token = item.strip().lower()
        if token == normalized_model:
            return True
        if token.endswith("/*") and normalized_model.startswith(token[:-2] + "/"):
            return True
        if token in normalized_model:
            return True
    return False

@dataclass(frozen=True)
class WorkerSelection:
    worker_id: str
    url: str
    source: str
    model_name: str
    score: float
    queue_depth: int = 0
    free_vram_gb: float = 0.0
    model_inventory: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
