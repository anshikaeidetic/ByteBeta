from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompressionBackend:
    requested: str
    resolved: str
    device: str
    has_cuda: bool
    has_triton: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "device": self.device,
            "has_cuda": self.has_cuda,
            "has_triton": self.has_triton,
            "reason": self.reason,
        }


def resolve_compression_backend(
    policy: str = "auto",
    *,
    tensor: Any = None,
    torch_module: Any = None,
) -> CompressionBackend:
    requested = str(policy or "auto").strip().lower() or "auto"
    if requested not in {"auto", "cuda", "triton", "torch"}:
        requested = "auto"
    has_cuda = bool(torch_module is not None and getattr(torch_module.cuda, "is_available", lambda: False)())
    has_triton = bool(importlib.util.find_spec("triton") is not None and has_cuda)
    device = "cpu"
    if tensor is not None:
        raw_device = str(getattr(tensor, "device", "cpu"))
        device = "cuda" if raw_device.startswith("cuda") else "cpu"
    elif has_cuda:
        device = "cuda"
    if requested == "torch":
        return CompressionBackend(
            requested=requested,
            resolved="torch",
            device=device,
            has_cuda=has_cuda,
            has_triton=has_triton,
        )
    if requested == "cuda":
        if has_cuda:
            return CompressionBackend(
                requested=requested,
                resolved="cuda",
                device="cuda",
                has_cuda=has_cuda,
                has_triton=has_triton,
            )
        return CompressionBackend(
            requested=requested,
            resolved="torch",
            device=device,
            has_cuda=has_cuda,
            has_triton=has_triton,
            reason="cuda_unavailable",
        )
    if requested == "triton":
        if has_triton:
            return CompressionBackend(
                requested=requested,
                resolved="triton",
                device="cuda",
                has_cuda=has_cuda,
                has_triton=has_triton,
            )
        fallback = "cuda" if has_cuda else "torch"
        return CompressionBackend(
            requested=requested,
            resolved=fallback,
            device="cuda" if has_cuda else device,
            has_cuda=has_cuda,
            has_triton=has_triton,
            reason="triton_unavailable",
        )
    if has_cuda:
        return CompressionBackend(
            requested=requested,
            resolved="cuda",
            device="cuda",
            has_cuda=has_cuda,
            has_triton=has_triton,
        )
    return CompressionBackend(
        requested=requested,
        resolved="torch",
        device=device,
        has_cuda=has_cuda,
        has_triton=has_triton,
        reason="cpu_fallback",
    )
