from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from byte.quantization.bitpacking import pack_unsigned, packed_nbytes, unpack_unsigned


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


@lru_cache(maxsize=128)
def _preconditioner(dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed or 0))
    perm = rng.permutation(dim)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dim)
    return perm.astype(np.int32), signs.astype(np.float32)


def _apply_preconditioner(vector: np.ndarray, seed: int) -> np.ndarray:
    perm, signs = _preconditioner(vector.shape[0], seed)
    return (vector[perm] * signs).astype(np.float32)


def _inverse_preconditioner(vector: np.ndarray, seed: int) -> np.ndarray:
    perm, signs = _preconditioner(vector.shape[0], seed)
    restored = np.zeros_like(vector, dtype=np.float32)
    restored[perm] = vector * signs
    return restored


def _angles_from_vector(vector: np.ndarray) -> tuple[float, np.ndarray]:
    dim = vector.shape[0]
    radius = float(np.linalg.norm(vector))
    if radius <= 0 or dim <= 1:
        return radius, np.zeros(max(dim - 1, 0), dtype=np.float32)
    normalized = vector / radius
    angles = []
    for index in range(dim - 1):
        tail_norm = float(np.linalg.norm(normalized[index:]))
        if tail_norm <= 0:
            angles.append(0.0)
            continue
        if index == dim - 2:
            angle = math.atan2(float(normalized[index + 1]), float(normalized[index]))
            if angle < 0:
                angle += 2 * math.pi
        else:
            value = float(np.clip(normalized[index] / tail_norm, -1.0, 1.0))
            angle = math.acos(value)
        angles.append(angle)
    return radius, np.asarray(angles, dtype=np.float32)


def _vector_from_angles(radius: float, angles: np.ndarray) -> np.ndarray:
    dim = int(len(angles) + 1)
    if dim <= 1:
        return np.asarray([radius], dtype=np.float32)
    coords = np.zeros(dim, dtype=np.float32)
    sin_prod = 1.0
    for index in range(dim):
        if index == 0:
            coords[index] = radius * math.cos(float(angles[0]))
            sin_prod = math.sin(float(angles[0]))
            continue
        if index < dim - 1:
            coords[index] = radius * sin_prod * math.cos(float(angles[index]))
            sin_prod *= math.sin(float(angles[index]))
            continue
        coords[index] = radius * sin_prod
    if dim >= 2:
        prefix = np.prod(np.sin(angles[:-1])) if len(angles) > 1 else 1.0
        final = float(angles[-1])
        coords[-2] = radius * prefix * math.cos(final)
        coords[-1] = radius * prefix * math.sin(final)
    return coords.astype(np.float32)


@dataclass(frozen=True)
class PolarQuantPayload:
    codec: str
    version: int
    shape: tuple[int, ...]
    seed: int
    bits: int
    radius: float
    angles: bytes
    angle_count: int
    raw_nbytes: int
    compressed_nbytes: int

    def summary(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "version": self.version,
            "shape": list(self.shape),
            "seed": self.seed,
            "bits": self.bits,
            "radius": round(self.radius, 6),
            "angle_count": self.angle_count,
            "raw_nbytes": self.raw_nbytes,
            "compressed_nbytes": self.compressed_nbytes,
        }


class PolarQuantCodec:
    def __init__(self, *, bits: int = 8, seed: int = 11) -> None:
        self.bits = max(2, min(int(bits or 8), 12))
        self.seed = int(seed or 0)

    def encode(self, value: Any) -> PolarQuantPayload:
        original = _to_numpy(value)
        vector = original.reshape(-1).astype(np.float32)
        if vector.size == 0:
            return PolarQuantPayload(
                codec="polarquant",
                version=1,
                shape=tuple(original.shape),
                seed=self.seed,
                bits=self.bits,
                radius=0.0,
                angles=b"",
                angle_count=0,
                raw_nbytes=0,
                compressed_nbytes=0,
            )
        preconditioned = _apply_preconditioner(vector, self.seed)
        radius, angles = _angles_from_vector(preconditioned)
        qmax = (1 << self.bits) - 1
        quantized = []
        for index, angle in enumerate(angles.tolist()):
            max_angle = 2.0 * math.pi if index == len(angles) - 1 else math.pi
            scaled = int(round(np.clip(angle / max_angle, 0.0, 1.0) * qmax))
            quantized.append(scaled)
        return PolarQuantPayload(
            codec="polarquant",
            version=1,
            shape=tuple(original.shape),
            seed=self.seed,
            bits=self.bits,
            radius=radius,
            angles=pack_unsigned(quantized, self.bits),
            angle_count=len(quantized),
            raw_nbytes=int(vector.nbytes),
            compressed_nbytes=packed_nbytes(len(quantized), self.bits) + 16,
        )

    def decode(self, payload: PolarQuantPayload) -> np.ndarray:
        if payload.angle_count <= 0:
            return np.zeros(payload.shape, dtype=np.float32)
        qmax = (1 << payload.bits) - 1
        quantized = unpack_unsigned(payload.angles, count=payload.angle_count, bits=payload.bits)
        angles = []
        for index, value in enumerate(quantized.tolist()):
            max_angle = 2.0 * math.pi if index == payload.angle_count - 1 else math.pi
            angles.append((float(value) / max(qmax, 1)) * max_angle)
        preconditioned = _vector_from_angles(payload.radius, np.asarray(angles, dtype=np.float32))
        restored = _inverse_preconditioner(preconditioned, payload.seed)
        return restored.reshape(payload.shape).astype(np.float32)

    def similarity(self, query: Any, payload: PolarQuantPayload) -> float:
        query_vec = _to_numpy(query).reshape(-1).astype(np.float32)
        recon = self.decode(payload).reshape(-1)
        query_norm = float(np.linalg.norm(query_vec))
        recon_norm = float(np.linalg.norm(recon))
        if query_norm <= 0 or recon_norm <= 0:
            return 0.0
        return float(np.dot(query_vec, recon) / (query_norm * recon_norm))
