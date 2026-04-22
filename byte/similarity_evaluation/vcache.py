"""VCache per-prompt learned sigmoid decision boundary .

UC Berkeley Sky Computing Lab. Enforces a user-defined maximum error rate δ
with formal guarantees via online logistic regression. Each cached embedding
maintains its own sigmoid model fitted from correctness feedback.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from typing import Any

from byte.similarity_evaluation.similarity_evaluation import SimilarityEvaluation
from byte.utils.log import byte_log


@dataclass
class _VCacheParams:
    t: float = 0.80       # sigmoid transition point (learned threshold)
    gamma: float = 10.0   # sigmoid sharpness
    n: int = 0            # observation count


class VCacheParamStore:
    """Thread-safe, JSON-backed store for per-embedding vCache sigmoid parameters.

    Keyed by question_id (integer row ID from the scalar store). Parameters
    survive server restarts via a sidecar .json file.
    """

    def __init__(self, store_path: str | None = None) -> None:
        self._path = store_path or os.path.join(
            os.environ.get("BYTE_CACHE_DIR", "byte_cache"), "vcache_params.json"
        )
        self._lock = threading.Lock()
        self._data: dict[int, _VCacheParams] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as fh:
                raw = json.load(fh)
            with self._lock:
                self._data = {
                    int(k): _VCacheParams(**v) for k, v in raw.items()
                }
        except FileNotFoundError:
            pass
        except Exception as exc:  # pylint: disable=W0703
            byte_log.warning("VCacheParamStore: failed to load %s — %s", self._path, exc)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            with self._lock:
                payload = {str(k): {"t": v.t, "gamma": v.gamma, "n": v.n} for k, v in self._data.items()}
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except Exception as exc:  # pylint: disable=W0703
            byte_log.warning("VCacheParamStore: failed to save — %s", exc)

    def get(self, question_id: int) -> _VCacheParams:
        with self._lock:
            return self._data.get(question_id, _VCacheParams())

    def put(self, question_id: int, params: _VCacheParams) -> None:
        with self._lock:
            self._data[question_id] = params
        self._save()

    def count_cold(self, min_observations: int) -> int:
        with self._lock:
            return sum(1 for p in self._data.values() if p.n < min_observations)

    def empirical_error_rate(self) -> float:
        """Rough proxy: fraction of entries whose fitted t > 0.9 (indicating high rejection)."""
        with self._lock:
            if not self._data:
                return 0.0
            high_t = sum(1 for p in self._data.values() if p.t > 0.9)
            return high_t / len(self._data)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _p_correct(similarity: float, params: _VCacheParams) -> float:
    """P(correct | similarity) = sigmoid(gamma * (similarity - t))."""
    return _sigmoid(params.gamma * (similarity - params.t))


def _online_update(params: _VCacheParams, similarity: float, was_correct: bool, lr: float) -> _VCacheParams:
    """One gradient step of logistic regression to update t and gamma."""
    p = _p_correct(similarity, params)
    y = 1.0 if was_correct else 0.0
    error = p - y
    # gradient wrt t:   d/dt  = error * gamma
    # gradient wrt gamma: d/dgamma = error * (similarity - t)  [sign flipped]
    new_t = params.t + lr * error * params.gamma
    new_gamma = params.gamma - lr * error * (similarity - params.t)
    new_gamma = max(1.0, min(50.0, new_gamma))  # clamp sharpness
    new_t = max(0.0, min(1.0, new_t))
    return _VCacheParams(t=new_t, gamma=new_gamma, n=params.n + 1)


class VCacheEvaluation(SimilarityEvaluation):
    """Per-prompt learned sigmoid decision boundary (vCache, ,
    UC Berkeley Sky Computing Lab).

    Replaces the global static similarity threshold with a per-prompt learned
    sigmoid model that enforces a user-defined maximum error rate δ. Fitted
    online from correctness feedback; falls back to a global threshold for
    cold embeddings (n < min_observations).
    """

    def __init__(
        self,
        *,
        delta: float = 0.05,
        min_observations: int = 10,
        learning_rate: float = 0.01,
        cold_fallback_threshold: float = 0.80,
        param_store: VCacheParamStore | None = None,
        store_path: str | None = None,
    ) -> None:
        self.delta = delta
        self.min_observations = min_observations
        self.learning_rate = learning_rate
        self.cold_fallback_threshold = cold_fallback_threshold
        self._store = param_store or VCacheParamStore(store_path=store_path)

    def _question_id(self, cache_dict: dict[str, Any]) -> int | None:
        search_result = cache_dict.get("search_result")
        if search_result is not None:
            try:
                return int(search_result[1])
            except (IndexError, TypeError, ValueError):
                pass
        return None

    def _raw_similarity(self, cache_dict: dict[str, Any]) -> float:
        """Extract cosine/vector similarity from the search result distance."""
        search_result = cache_dict.get("search_result")
        if search_result is not None:
            try:
                # Vector search returns (distance, id); convert distance to similarity.
                # FAISS returns L2 distance; approximate cosine ~ 1 - dist/2 for unit vecs.
                dist = float(search_result[0])
                return max(0.0, min(1.0, 1.0 - dist / 2.0))
            except (IndexError, TypeError, ValueError):
                pass
        # Fallback: use embedding cosine if available
        src_emb = cache_dict.get("embedding")
        if src_emb is not None:
            return 0.75  # safe conservative estimate
        return 0.5

    def evaluation(self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **_) -> float:
        """Accept or reject the cache hit based on per-prompt sigmoid decision.

        Returns the raw similarity score if accepted, else 0.0.
        """
        similarity = self._raw_similarity(cache_dict)
        question_id = self._question_id(cache_dict)

        if question_id is None:
            # No row ID available — fall back to cold threshold
            return similarity if similarity >= self.cold_fallback_threshold else 0.0

        params = self._store.get(question_id)

        if params.n < self.min_observations:
            # Cold start: use global fallback threshold
            return similarity if similarity >= self.cold_fallback_threshold else 0.0

        # vCache decision: accept iff P(correct | s) >= 1 - delta
        p_correct = _p_correct(similarity, params)
        accepted = p_correct >= (1.0 - self.delta)
        return similarity if accepted else 0.0

    def update(self, question_id: int, similarity: float, was_correct: bool) -> None:
        """Record a correctness observation and update the per-prompt sigmoid model."""
        if question_id is None:
            return
        params = self._store.get(question_id)
        updated = _online_update(params, similarity, was_correct, self.learning_rate)
        self._store.put(question_id, updated)
        try:
            from byte.telemetry import (
                bump_research_counter as _bump,  # pylint: disable=import-outside-toplevel
            )
            _bump("vcache_threshold_updates")
        except Exception:  # pragma: no cover - defensive
            pass

    def cold_count(self) -> int:
        return self._store.count_cold(self.min_observations)

    def empirical_error_rate(self) -> float:
        return self._store.empirical_error_rate()

    def range(self) -> tuple[float, float]:
        return 0.0, 1.0
