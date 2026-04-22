from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from byte.quantization.bitpacking import pack_unsigned, packed_nbytes, unpack_unsigned
from byte.quantization.qjl import QJLCodec, QJLPayload


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class TurboQuantPayload:
    codec: str
    version: int
    shape: tuple[int, ...]
    bits: int
    mean: float
    scale: float
    quantized: bytes
    count: int
    residual: QJLPayload | None
    raw_nbytes: int
    compressed_nbytes: int

    def summary(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "version": self.version,
            "shape": list(self.shape),
            "bits": self.bits,
            "mean": round(self.mean, 6),
            "scale": round(self.scale, 6),
            "count": self.count,
            "residual_codec": self.residual.codec if self.residual is not None else "",
            "raw_nbytes": self.raw_nbytes,
            "compressed_nbytes": self.compressed_nbytes,
        }


class TurboQuantCodec:
    def __init__(self, *, bits: int = 8, residual_sketch_dim: int = 64, seed: int = 19) -> None:
        self.bits = max(2, min(int(bits or 8), 12))
        self.residual_codec = QJLCodec(sketch_dim=max(int(residual_sketch_dim or 0), 8), seed=seed)

    def encode(self, value: Any) -> TurboQuantPayload:
        original = _to_numpy(value)
        vector = original.reshape(-1).astype(np.float32)
        if vector.size == 0:
            return TurboQuantPayload(
                codec="turboquant",
                version=1,
                shape=tuple(original.shape),
                bits=self.bits,
                mean=0.0,
                scale=1.0,
                quantized=b"",
                count=0,
                residual=None,
                raw_nbytes=0,
                compressed_nbytes=0,
            )
        mean = float(np.mean(vector))
        centered = vector - mean
        max_abs = float(np.max(np.abs(centered))) if centered.size else 0.0
        qmax = (1 << self.bits) - 1
        scale = max(max_abs / max(qmax // 2, 1), 1e-8)
        shifted = np.clip(np.round(centered / scale) + (qmax // 2), 0, qmax).astype(np.int32)
        decoded = (shifted.astype(np.float32) - (qmax // 2)) * scale + mean
        residual_vec = vector - decoded
        residual = None
        if float(np.linalg.norm(residual_vec)) > 1e-6:
            residual = self.residual_codec.encode(residual_vec.astype(np.float32))
        compressed_nbytes = packed_nbytes(len(shifted), self.bits) + 16
        if residual is not None:
            compressed_nbytes += int(residual.compressed_nbytes)
        return TurboQuantPayload(
            codec="turboquant",
            version=1,
            shape=tuple(original.shape),
            bits=self.bits,
            mean=mean,
            scale=scale,
            quantized=pack_unsigned(shifted.tolist(), self.bits),
            count=len(shifted),
            residual=residual,
            raw_nbytes=int(vector.nbytes),
            compressed_nbytes=compressed_nbytes,
        )

    def decode(self, payload: TurboQuantPayload) -> np.ndarray:
        if payload.count <= 0:
            return np.zeros(payload.shape, dtype=np.float32)
        qmax = (1 << payload.bits) - 1
        shifted = unpack_unsigned(payload.quantized, count=payload.count, bits=payload.bits).astype(np.float32)
        decoded = (shifted - (qmax // 2)) * float(payload.scale) + float(payload.mean)
        if payload.residual is not None:
            decoded = decoded + self.residual_codec.decode(payload.residual).reshape(-1)
        return decoded.reshape(payload.shape).astype(np.float32)

    def approximate_inner_product(self, query: Any, payload: TurboQuantPayload) -> float:
        query_vec = _to_numpy(query).reshape(-1).astype(np.float32)
        recon = self.decode(payload).reshape(-1)
        return float(np.dot(query_vec, recon))

    def similarity(self, query: Any, payload: TurboQuantPayload) -> float:
        query_vec = _to_numpy(query).reshape(-1).astype(np.float32)
        recon = self.decode(payload).reshape(-1)
        query_norm = float(np.linalg.norm(query_vec))
        recon_norm = float(np.linalg.norm(recon))
        if query_norm <= 0 or recon_norm <= 0:
            return 0.0
        return float(np.dot(query_vec, recon) / (query_norm * recon_norm))
