from __future__ import annotations

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
def _projection_matrix(input_dim: int, sketch_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed or 0))
    matrix = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(sketch_dim, input_dim))
    return matrix / np.sqrt(float(sketch_dim))


@dataclass(frozen=True)
class QJLPayload:
    codec: str
    version: int
    shape: tuple[int, ...]
    sketch_dim: int
    seed: int
    norm: float
    signs: bytes
    raw_nbytes: int
    compressed_nbytes: int

    def summary(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "version": self.version,
            "shape": list(self.shape),
            "sketch_dim": self.sketch_dim,
            "seed": self.seed,
            "norm": round(self.norm, 6),
            "raw_nbytes": self.raw_nbytes,
            "compressed_nbytes": self.compressed_nbytes,
        }


class QJLCodec:
    def __init__(self, *, sketch_dim: int = 128, seed: int = 7) -> None:
        self.sketch_dim = max(int(sketch_dim or 0), 8)
        self.seed = int(seed or 0)

    def encode(self, value: Any) -> QJLPayload:
        original = _to_numpy(value)
        vector = original.reshape(-1).astype(np.float32)
        projection = _projection_matrix(vector.shape[0], self.sketch_dim, self.seed)
        projected = projection @ vector
        signs = (projected >= 0).astype(np.uint8)
        return QJLPayload(
            codec="qjl",
            version=1,
            shape=tuple(original.shape),
            sketch_dim=self.sketch_dim,
            seed=self.seed,
            norm=float(np.linalg.norm(vector)),
            signs=pack_unsigned(signs.tolist(), 1),
            raw_nbytes=int(vector.nbytes),
            compressed_nbytes=packed_nbytes(self.sketch_dim, 1) + 16,
        )

    def decode(self, payload: QJLPayload) -> np.ndarray:
        flat_dim = int(np.prod(payload.shape))
        if flat_dim <= 0:
            return np.zeros(payload.shape, dtype=np.float32)
        projection = _projection_matrix(flat_dim, payload.sketch_dim, payload.seed)
        signs = unpack_unsigned(payload.signs, count=payload.sketch_dim, bits=1).astype(np.float32)
        signs = np.where(signs > 0.0, 1.0, -1.0)
        recon = (projection.T @ signs) / max(float(payload.sketch_dim), 1.0)
        norm = float(np.linalg.norm(recon))
        if payload.norm > 0 and norm > 0:
            recon = recon * (payload.norm / norm)
        return recon.reshape(payload.shape).astype(np.float32)

    def approximate_inner_product(self, query: Any, payload: QJLPayload) -> float:
        query_vec = _to_numpy(query).reshape(-1).astype(np.float32)
        recon = self.decode(payload).reshape(-1)
        return float(np.dot(query_vec, recon))

    def similarity(self, query: Any, payload: QJLPayload) -> float:
        query_vec = _to_numpy(query).reshape(-1).astype(np.float32)
        recon = self.decode(payload).reshape(-1)
        query_norm = float(np.linalg.norm(query_vec))
        recon_norm = float(np.linalg.norm(recon))
        if query_norm <= 0 or recon_norm <= 0:
            return 0.0
        return float(np.dot(query_vec, recon) / (query_norm * recon_norm))
