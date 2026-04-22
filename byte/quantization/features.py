from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_:-]{2,}")


def lexical_tokens(value: object) -> list[str]:
    if value in (None, ""):
        return []
    return _TOKEN_RE.findall(str(value).lower())


def hashed_text_features(
    value: object,
    *,
    dimension: int = 256,
    signed: bool = True,
) -> np.ndarray:
    dims = max(int(dimension or 0), 8)
    vector = np.zeros(dims, dtype=np.float32)
    tokens = lexical_tokens(value)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        bucket = int.from_bytes(digest[:8], "little") % dims
        sign = 1.0 if not signed or digest[8] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def blend_token_streams(values: Iterable[object]) -> np.ndarray:
    vectors = [hashed_text_features(value) for value in values if value not in (None, "")]
    if not vectors:
        return np.zeros(256, dtype=np.float32)
    merged = np.sum(np.stack(vectors, axis=0), axis=0)
    norm = float(np.linalg.norm(merged))
    if norm > 0:
        merged /= norm
    return merged.astype(np.float32)
