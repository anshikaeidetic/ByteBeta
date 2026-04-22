"""LSH-based approximate cache prefilter .

Based on "Leveraging Approximate Caching for Faster Retrieval-Augmented
Generation" (March 2025). The idea: a locality-sensitive hash provides a
near-O(1) near-duplicate lookup that can gate the expensive vector similarity
search. For high-skew workloads (77.2% DB-call reduction on MedRAG), most
repeat queries are served entirely from the LSH tier, skipping embedding +
vector search altogether.

This module wraps `datasketch`'s MinHash-LSH into a simple Tier-0 gate:

    prefilter = LSHPrefilter(num_perm=128, threshold=0.6)
    prefilter.index(question_id="42", text="What is Byte?")
    hits = prefilter.query("What's Byte?")         # list of prior question_ids

`hits` is either empty (no near-duplicates known → fall through to vector
search) or a small list of previously-seen near-duplicate keys. The caller
can then attempt a direct lookup on those keys and skip the expensive search
when one is present.

The prefilter emits Prometheus counters via the telemetry registry:
    - byteai_lsh_prefilter_lookups_total
    - byteai_lsh_prefilter_tier0_hits_total
    - byteai_lsh_prefilter_skipped_searches_total
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from typing import Any

from byte.utils.log import byte_log

try:
    from datasketch import MinHash, MinHashLSH  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    MinHash = None  # type: ignore
    MinHashLSH = None  # type: ignore


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _shingles(text: str, k: int = 5) -> Iterable[bytes]:
    """Yield k-word shingles as encoded tokens. Fast and language-agnostic."""
    tokens = [t.lower() for t in _WORD_RE.findall(text or "") if t]
    if not tokens:
        return
    if len(tokens) <= k:
        yield " ".join(tokens).encode("utf-8")
        return
    for i in range(len(tokens) - k + 1):
        yield " ".join(tokens[i:i + k]).encode("utf-8")


class LSHPrefilter:
    """Thread-safe MinHash-LSH gate in front of the vector store."""

    def __init__(
        self,
        *,
        num_perm: int = 128,
        threshold: float = 0.6,
        shingle_k: int = 5,
    ) -> None:
        if MinHashLSH is None:
            raise RuntimeError(
                "LSHPrefilter requires the 'datasketch' package. "
                "Install it with `pip install datasketch`."
            )
        self._num_perm = int(num_perm)
        self._threshold = float(threshold)
        self._shingle_k = max(1, int(shingle_k))
        self._lsh = MinHashLSH(threshold=self._threshold, num_perm=self._num_perm)
        self._lock = threading.Lock()
        self._signatures: dict[Any, Any] = {}  # key -> MinHash, retained for updates

    # ── introspection ──

    @property
    def size(self) -> int:
        return len(self._signatures)

    # ── indexing ──

    def _minhash(self, text: str) -> Any:
        m = MinHash(num_perm=self._num_perm)
        for shingle in _shingles(text, self._shingle_k):
            m.update(shingle)
        return m

    def index(self, key: Any, text: str) -> None:
        """Add or update an entry in the LSH. Safe for concurrent callers."""
        if not text:
            return
        m = self._minhash(text)
        with self._lock:
            if key in self._signatures:
                try:
                    self._lsh.remove(key)
                except Exception:  # pragma: no cover - defensive; remove missing
                    pass
            try:
                self._lsh.insert(key, m)
                self._signatures[key] = m
            except Exception as exc:  # pragma: no cover - defensive
                byte_log.warning("LSHPrefilter insert failed for key=%s: %s", key, exc)

    def remove(self, key: Any) -> None:
        with self._lock:
            try:
                self._lsh.remove(key)
            except Exception:  # pragma: no cover - defensive
                pass
            self._signatures.pop(key, None)

    # ── querying ──

    def query(self, text: str) -> list[Any]:
        """Return keys of near-duplicate entries (Jaccard ≥ threshold)."""
        try:
            from byte.telemetry import (
                bump_research_counter as _bump,  # pylint: disable=import-outside-toplevel
            )
            _bump("lsh_prefilter_lookups")
        except Exception:  # pragma: no cover - defensive
            pass

        if not text or not self._signatures:
            return []
        m = self._minhash(text)
        with self._lock:
            try:
                results = list(self._lsh.query(m))
            except Exception:  # pragma: no cover - defensive
                return []

        if results:
            try:
                from byte.telemetry import (
                    bump_research_counter as _bump,  # pylint: disable=import-outside-toplevel
                )
                _bump("lsh_prefilter_tier0_hits")
                _bump("lsh_prefilter_skipped_searches")
            except Exception:  # pragma: no cover - defensive
                pass
        return results

    # ── stats ──

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "num_perm": self._num_perm,
            "threshold": self._threshold,
            "shingle_k": self._shingle_k,
        }


# ─── Process-wide instance tied to the active cache ─────────────────────


_INSTANCE_LOCK = threading.Lock()
_INSTANCE: LSHPrefilter | None = None


def get_lsh_prefilter(
    *,
    num_perm: int = 128,
    threshold: float = 0.6,
    shingle_k: int = 5,
) -> LSHPrefilter | None:
    """Return the process-wide prefilter, creating on first call. None if datasketch is missing."""
    global _INSTANCE
    if MinHashLSH is None:
        return None
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = LSHPrefilter(
                num_perm=num_perm,
                threshold=threshold,
                shingle_k=shingle_k,
            )
        return _INSTANCE


def reset_lsh_prefilter() -> None:
    """Drop the process-wide prefilter (useful for tests + config reloads)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None


__all__ = ["LSHPrefilter", "get_lsh_prefilter", "reset_lsh_prefilter"]
