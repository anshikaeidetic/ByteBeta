"""Shared descriptors and low-level helpers for the H2O runtime."""

from __future__ import annotations

# ruff: noqa: F401
import itertools
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from byte.h2o.policy import (
    H2OSequenceCache,
    H2OSettings,
    normalize_model_family,
    resolve_h2o_settings,
)
from byte.quantization.backend import resolve_compression_backend
from byte.quantization.polar import PolarQuantCodec
from byte.quantization.qjl import QJLCodec
from byte.quantization.turbo import TurboQuantCodec
from byte.utils.error import CacheError


def _torch() -> Any:
    from byte.utils import import_torch

    import_torch()
    import torch

    return torch


def _transformers() -> tuple[Any, ...]:
    from byte.utils import import_huggingface

    import_huggingface()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer


_RUNTIME_COUNTER = itertools.count(1)
_SUPPORTED_COMPRESSION_FAMILIES = {
    "llama",
    "mistral",
    "gemma",
    "qwen2",
    "phi3",
    "opt",
    "gpt_neox",
}


@dataclass
class CompressionDescriptor:
    requested: bool
    applied: bool
    requested_codec: str
    applied_codec: str
    bits: int
    hot_window_ratio: float
    mode: str
    backend_policy: str
    backend: str
    verify_shadow_rate: float
    fallback_reason: str = ""
    raw_bytes: int = 0
    compressed_bytes: int = 0
    compression_ratio: float = 0.0
    distortion_mean: float = 0.0
    layer_count: int = 0
    compressed_layers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "applied": self.applied,
            "requested_codec": self.requested_codec,
            "applied_codec": self.applied_codec,
            "bits": self.bits,
            "hot_window_ratio": self.hot_window_ratio,
            "mode": self.mode,
            "backend_policy": self.backend_policy,
            "backend": self.backend,
            "verify_shadow_rate": self.verify_shadow_rate,
            "fallback_reason": self.fallback_reason,
            "raw_bytes": self.raw_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "distortion_mean": self.distortion_mean,
            "layer_count": self.layer_count,
            "compressed_layers": self.compressed_layers,
        }


@dataclass
class RuntimeDescriptor:
    provider: str
    model_name: str
    model_family: str
    requested_h2o: bool
    applied_h2o: bool
    heavy_ratio: float
    recent_ratio: float
    heavy_budget: int
    recent_budget: int
    cache_budget: int
    fallback_reason: str
    prompt_tokens: int
    cache_hit: bool = False
    compression: CompressionDescriptor | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "h2o_requested": self.requested_h2o,
            "h2o_applied": self.applied_h2o,
            "h2o_heavy_ratio": self.heavy_ratio,
            "h2o_recent_ratio": self.recent_ratio,
            "h2o_heavy_budget": self.heavy_budget,
            "h2o_recent_budget": self.recent_budget,
            "h2o_cache_budget": self.cache_budget,
            "h2o_fallback_reason": self.fallback_reason,
            "prompt_tokens": self.prompt_tokens,
            "cache_hit": self.cache_hit,
        }
        payload["byte_compression"] = (
            self.compression.to_dict() if self.compression is not None else {}
        )
        return payload


class _RuntimeStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.applied = 0
        self.fallbacks = 0
        self.evictions = 0
        self.retained_fraction_sum = 0.0
        self.retained_fraction_samples = 0
        self.compression_requests = 0
        self.compression_applied = 0
        self.compression_fallbacks = 0
        self.compression_ratio_sum = 0.0
        self.compression_ratio_samples = 0
        self.distortion_sum = 0.0
        self.distortion_samples = 0

    def record(
        self,
        descriptor: RuntimeDescriptor,
        eviction_stats: dict[str, Any] | None,
    ) -> None:
        with self._lock:
            self.requests += 1
            if descriptor.applied_h2o:
                self.applied += 1
            if descriptor.requested_h2o and not descriptor.applied_h2o:
                self.fallbacks += 1
            compression = descriptor.compression
            if compression is not None and compression.requested:
                self.compression_requests += 1
                if compression.applied:
                    self.compression_applied += 1
                if compression.fallback_reason and not compression.applied:
                    self.compression_fallbacks += 1
                if compression.compression_ratio > 0:
                    self.compression_ratio_sum += float(compression.compression_ratio)
                    self.compression_ratio_samples += 1
                if compression.distortion_mean > 0:
                    self.distortion_sum += float(compression.distortion_mean)
                    self.distortion_samples += 1
            if eviction_stats:
                self.evictions += int(eviction_stats.get("evicted_tokens", 0) or 0)
                if eviction_stats.get("original_tokens", 0):
                    self.retained_fraction_sum += float(
                        eviction_stats.get("retained_fraction", 0.0) or 0.0
                    )
                    self.retained_fraction_samples += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            avg_retained = (
                self.retained_fraction_sum / self.retained_fraction_samples
                if self.retained_fraction_samples
                else 1.0
            )
            return {
                "requests": self.requests,
                "applied": self.applied,
                "fallbacks": self.fallbacks,
                "evictions": self.evictions,
                "avg_retained_fraction": round(avg_retained, 6),
                "compression_requests": self.compression_requests,
                "compression_applied": self.compression_applied,
                "compression_fallbacks": self.compression_fallbacks,
                "avg_compression_ratio": round(
                    self.compression_ratio_sum / self.compression_ratio_samples, 6
                )
                if self.compression_ratio_samples
                else 0.0,
                "avg_distortion_mean": round(self.distortion_sum / self.distortion_samples, 6)
                if self.distortion_samples
                else 0.0,
            }


_STATS = _RuntimeStats()


def h2o_runtime_stats() -> dict[str, Any]:
    return _STATS.snapshot()


def _normalize_kv_codec(codec_name: str) -> str:
    normalized = str(codec_name or "").strip().lower()
    if normalized in {"", "disabled", "none"}:
        return "disabled"
    if normalized in {"pq", "polar"}:
        return "polarquant"
    if normalized in {"tq", "turbo"}:
        return "turboquant"
    return normalized


def _resolve_compression_descriptor(
    *,
    model_family: str,
    kv_codec: str,
    kv_bits: int,
    kv_hot_window_ratio: float,
    compression_mode: str,
    compression_backend_policy: str,
    compression_verify_shadow_rate: float,
) -> CompressionDescriptor:
    normalized_family = normalize_model_family(model_family)
    normalized_codec = _normalize_kv_codec(kv_codec)
    normalized_mode = str(compression_mode or "shadow").strip().lower() or "shadow"
    normalized_backend_policy = (
        str(compression_backend_policy or "auto").strip().lower() or "auto"
    )
    requested = normalized_codec not in {"disabled", "h2o"}
    backend = resolve_compression_backend(normalized_backend_policy).resolved
    if not requested:
        return CompressionDescriptor(
            requested=False,
            applied=False,
            requested_codec=normalized_codec,
            applied_codec="disabled",
            bits=max(1, int(kv_bits or 1)),
            hot_window_ratio=max(0.0, min(float(kv_hot_window_ratio or 0.0), 1.0)),
            mode=normalized_mode,
            backend_policy=normalized_backend_policy,
            backend=backend,
            verify_shadow_rate=max(0.0, min(float(compression_verify_shadow_rate or 0.0), 1.0)),
        )
    if normalized_family not in _SUPPORTED_COMPRESSION_FAMILIES:
        return CompressionDescriptor(
            requested=True,
            applied=False,
            requested_codec=normalized_codec,
            applied_codec="disabled",
            bits=max(1, int(kv_bits or 1)),
            hot_window_ratio=max(0.0, min(float(kv_hot_window_ratio or 0.0), 1.0)),
            mode=normalized_mode,
            backend_policy=normalized_backend_policy,
            backend=backend,
            verify_shadow_rate=max(0.0, min(float(compression_verify_shadow_rate or 0.0), 1.0)),
            fallback_reason=f"unsupported_model_family:{normalized_family or 'unknown'}",
        )
    return CompressionDescriptor(
        requested=True,
        applied=True,
        requested_codec=normalized_codec,
        applied_codec=normalized_codec,
        bits=max(1, int(kv_bits or 1)),
        hot_window_ratio=max(0.0, min(float(kv_hot_window_ratio or 0.0), 1.0)),
        mode=normalized_mode,
        backend_policy=normalized_backend_policy,
        backend=backend,
        verify_shadow_rate=max(0.0, min(float(compression_verify_shadow_rate or 0.0), 1.0)),
    )


def describe_huggingface_runtime(
    *,
    model_name: str,
    model_family: str,
    prompt_tokens: int,
    cache_hit: bool,
    h2o_enabled: bool,
    h2o_heavy_ratio: float,
    h2o_recent_ratio: float,
    kv_codec: str = "disabled",
    kv_bits: int = 8,
    kv_hot_window_ratio: float = 0.25,
    compression_mode: str = "shadow",
    compression_backend_policy: str = "auto",
    compression_verify_shadow_rate: float = 0.1,
) -> RuntimeDescriptor:
    settings = resolve_h2o_settings(
        enabled=h2o_enabled,
        prompt_tokens=prompt_tokens,
        model_family=model_family,
        heavy_ratio=h2o_heavy_ratio,
        recent_ratio=h2o_recent_ratio,
    )
    compression = _resolve_compression_descriptor(
        model_family=model_family,
        kv_codec=kv_codec,
        kv_bits=kv_bits,
        kv_hot_window_ratio=kv_hot_window_ratio,
        compression_mode=compression_mode,
        compression_backend_policy=compression_backend_policy,
        compression_verify_shadow_rate=compression_verify_shadow_rate,
    )
    return RuntimeDescriptor(
        provider="huggingface",
        model_name=model_name,
        model_family=normalize_model_family(model_family),
        requested_h2o=settings.requested,
        applied_h2o=settings.applied,
        heavy_ratio=settings.heavy_ratio,
        recent_ratio=settings.recent_ratio,
        heavy_budget=settings.heavy_budget,
        recent_budget=settings.recent_budget,
        cache_budget=settings.cache_budget,
        fallback_reason=settings.fallback_reason,
        prompt_tokens=prompt_tokens,
        cache_hit=cache_hit,
        compression=compression,
    )


def _codec_for_tensor(codec_name: str, bits: int, *, is_key: bool) -> Any:
    normalized = _normalize_kv_codec(codec_name)
    if normalized == "polarquant":
        return PolarQuantCodec(bits=bits)
    if normalized == "turboquant":
        return TurboQuantCodec(bits=bits)
    if normalized == "qjl":
        return QJLCodec(sketch_dim=max(32, min(256, bits * 16)))
    if normalized == "hybrid":
        if is_key:
            return QJLCodec(sketch_dim=max(32, min(256, bits * 16)))
        return TurboQuantCodec(bits=bits)
    raise CacheError(f"Unsupported KV codec: {codec_name}")


def _tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _distortion_threshold(codec_name: str) -> float:
    normalized = _normalize_kv_codec(codec_name)
    if normalized == "qjl":
        return 0.45
    if normalized == "polarquant":
        return 0.28
    if normalized == "turboquant":
        return 0.18
    if normalized == "hybrid":
        return 0.22
    return 0.25

def _past_seq_len(past_key_values: Any) -> int:
    if not past_key_values:
        return 0
    key_states = past_key_values[0][0]
    return int(key_states.shape[-2])


def _build_response(
    *,
    response_mode: str,
    model_name: str,
    content: str,
    usage: dict[str, int],
    descriptor: RuntimeDescriptor,
) -> dict[str, Any]:
    created = int(time.time())
    payload = {
        "id": f"byte-hf-{next(_RUNTIME_COUNTER)}",
        "created": created,
        "model": model_name,
        "usage": usage,
        "byte_provider": "huggingface",
        "byte_runtime": descriptor.to_dict(),
    }
    if response_mode == "chat":
        payload["object"] = "chat.completion"
        payload["choices"] = [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ]
        return payload
    payload["object"] = "text_completion"
    payload["choices"] = [
        {
            "index": 0,
            "finish_reason": "stop",
            "text": content,
        }
    ]
    return payload


__all__ = [
    "_RUNTIME_COUNTER",
    "_STATS",
    "CompressionDescriptor",
    "RuntimeDescriptor",
    "_build_response",
    "_codec_for_tensor",
    "_distortion_threshold",
    "_normalize_kv_codec",
    "_past_seq_len",
    "_resolve_compression_descriptor",
    "_tensor_to_numpy",
    "_torch",
    "_transformers",
    "describe_huggingface_runtime",
    "h2o_runtime_stats",
]
