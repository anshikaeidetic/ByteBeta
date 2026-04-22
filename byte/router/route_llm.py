"""Byte Smart Router — multi-signal complexity analysis for model tier selection.

Per-query cheap/strong classifier that emits a routing score in [0, 1];
queries scoring below a tunable threshold are served by the cheap tier, the
rest are escalated to the strong tier.

This module provides:

1. `ByteRouterScorer` — a no-dependency heuristic scorer that extracts features
   (length, reasoning/code/math markers, token counts) and composes them into
   a routing score. Works out of the box with no training data.

2. Optional KNN mode — when the operator provides a JSON file of labelled
   (query, label) examples, the scorer embeds the incoming query and votes
   among the k-nearest labelled neighbours.

3. `route_decision()` — returns the selected model, score, and a rationale
   suitable for audit/metric emission.

The scorer is stateless per call and safe to invoke from any pipeline stage.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

from byte.utils.log import byte_log

# ─── Heuristic feature extraction ─────────────────────────────────────────

# Patterns in a query that typically indicate the strong model is needed.
# Weights sum to > 1 when multiple fire, but `score` is clipped to [0, 1].
_STRONG_SIGNALS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\b(why|prove|proof|derive|explain why|step by step|chain of thought)\b", re.I), 0.32),
    (re.compile(r"\b(analy[sz]e|compare|contrast|synthesi[sz]e|evaluate)\b", re.I), 0.24),
    (re.compile(r"\b(write|implement|refactor|debug|fix|build|create)\s+(code|function|class|algorithm|solution|program|script)\b", re.I), 0.34),
    (re.compile(r"\bsolve\s+(this|the)\s+(problem|puzzle|equation|question)\b", re.I), 0.24),
    (re.compile(r"```[\s\S]+```"), 0.24),                                # fenced code block
    (re.compile(r"\$[^$]{4,}\$|\\\w+\{[^}]+\}"), 0.22),                  # latex math
    (re.compile(r"\b(plan|outline|design|architecture|tradeoff|tradeoffs?)\b", re.I), 0.16),
    (re.compile(r"\b(summari[sz]e|translate|paraphrase)\s+the\s+(following|text|document)", re.I), 0.12),
    (re.compile(r"\b(multi[- ]step|nested|recursive|dynamic programming|algorithm|complexity)\b", re.I), 0.30),
    (re.compile(r"\b\d+\s*(balloons?|balls?|coins?|integers?|items?)\b.*\b(maximum|minimum|optimal)\b", re.I), 0.35),
    (re.compile(r"\b(maximum|minimum|optimal|optimi[sz]e)\s+\w*\s*(number|count|score|cost|coins?)\b", re.I), 0.22),
    (re.compile(r"\b(return\s+the|output\s+the|find\s+the)\s+(maximum|minimum|optimal|best)\b", re.I), 0.20),
    (re.compile(r"\b(example|examples|test case|input|output)\s*[:=]", re.I), 0.08),
)

# Patterns that indicate the cheap model suffices.
_CHEAP_SIGNALS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"^\s*(hi|hello|hey|thanks|thank you|bye|ok|okay)\b", re.I), 0.40),
    (re.compile(r"^\s*(what is|who is|when (is|was)|where is|how old)\b", re.I), 0.20),
    (re.compile(r"^\s*(capital of|define|meaning of)\b", re.I), 0.25),
    (re.compile(r"^\s*\d+\s*[+\-*/×÷]\s*\d+\s*=?\s*$"), 0.45),          # arithmetic
    (re.compile(r"^.{0,48}\?\s*$"), 0.12),                              # short question
)


def _extract_features(query: str) -> dict[str, float]:
    """Return raw numeric features for a query — pure function, no I/O."""
    q = str(query or "")
    length = len(q)
    # Approximate token count using the "4 chars per token" heuristic.
    approx_tokens = max(1, length // 4)
    line_count = q.count("\n") + 1
    uppercase_ratio = (sum(1 for c in q if c.isupper()) / length) if length else 0.0
    punctuation_ratio = (sum(1 for c in q if c in ".!?,;:") / length) if length else 0.0
    has_code_fence = "```" in q
    has_inline_code = "`" in q and not has_code_fence
    has_math = bool(re.search(r"\$[^$]{2,}\$|\\[a-zA-Z]+\{", q))
    has_numbered_list = bool(re.search(r"^\s*\d+\.\s", q, re.M))
    return {
        "length": float(length),
        "tokens": float(approx_tokens),
        "lines": float(line_count),
        "uppercase_ratio": uppercase_ratio,
        "punctuation_ratio": punctuation_ratio,
        "has_code_fence": 1.0 if has_code_fence else 0.0,
        "has_inline_code": 1.0 if has_inline_code else 0.0,
        "has_math": 1.0 if has_math else 0.0,
        "has_numbered_list": 1.0 if has_numbered_list else 0.0,
    }


def _length_pressure(approx_tokens: float) -> float:
    """Map token count to [0, 0.35]: longer → more likely strong."""
    # Logistic curve centred at ~250 tokens (typical short query)
    return 0.35 / (1.0 + math.exp(-(approx_tokens - 250.0) / 120.0))


def _heuristic_score(query: str) -> tuple[float, list[str]]:
    """Compute heuristic routing score in [0, 1] plus a list of signals that fired."""
    score = 0.0
    signals: list[str] = []

    for pattern, weight in _STRONG_SIGNALS:
        if pattern.search(query):
            score += weight
            signals.append(f"+strong:{pattern.pattern[:32]}")

    for pattern, weight in _CHEAP_SIGNALS:
        if pattern.search(query):
            score -= weight
            signals.append(f"-cheap:{pattern.pattern[:32]}")

    features = _extract_features(query)
    score += _length_pressure(features["tokens"])
    if features["has_code_fence"]:
        score += 0.10
        signals.append("+code_fence")
    if features["has_math"]:
        score += 0.08
        signals.append("+math")
    if features["has_numbered_list"] and features["length"] > 200:
        score += 0.06
        signals.append("+structured_multi_step")

    # Clip to [0, 1] and return signals.
    return max(0.0, min(1.0, score)), signals


# ─── KNN scorer (optional, data-driven) ─────────────────────────────────


@dataclass
class _KNNSample:
    query: str
    label: float   # 0 = cheap sufficed, 1 = strong needed
    embedding: list[float]


class _KNNScorer:
    """k-nearest-neighbour scorer over a labelled seed dataset.

    Loaded lazily from a JSON file specified via `ByteRouterScorer(seed_path=...)`.
    Each entry in the file must have the shape:
        {"query": "<text>", "label": 0 or 1, "embedding": [...]}
    If embeddings are absent, set the `embedding_fn` on ByteRouterScorer and this
    class will compute them on load.
    """

    def __init__(self, samples: list[_KNNSample], k: int = 5) -> None:
        self._samples = samples
        self._k = max(1, int(k))

    def score(self, query_embedding: list[float]) -> tuple[float, list[str]]:
        if not self._samples:
            return 0.5, ["knn:empty_dataset"]
        # Cosine similarity to every sample.
        def cos(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        ranked = sorted(
            ((cos(query_embedding, s.embedding), s) for s in self._samples),
            key=lambda t: -t[0],
        )[: self._k]
        # Similarity-weighted label average.
        sims = [max(0.0, sim) for sim, _ in ranked]
        labels = [s.label for _, s in ranked]
        total = sum(sims) or 1.0
        weighted = sum(sim * label for sim, label in zip(sims, labels)) / total
        return max(0.0, min(1.0, float(weighted))), [
            f"knn:{len(ranked)}_neighbours",
            f"knn:top_sim={ranked[0][0]:.3f}",
        ]


# ─── Public scorer ────────────────────────────────────────────────────────


_SCORER_LOCK = threading.Lock()
_SCORER_CACHE: dict[str, ByteRouterScorer] = {}


@dataclass
class ByteRouterDecision:
    selected_model: str
    score: float
    tier: str                  # "cheap" | "strong"
    reason: str
    signals: list[str]


class ByteRouterScorer:
    """Per-query cheap/strong classifier for tier selection."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        seed_path: str | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
        knn_k: int = 5,
    ) -> None:
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self._seed_path = seed_path or os.environ.get("BYTE_ROUTE_LLM_SEED_PATH", "")
        self._embedding_fn = embedding_fn
        self._knn: _KNNScorer | None = None
        if self._seed_path and os.path.exists(self._seed_path):
            self._load_seed()

    # ── loading ──
    def _load_seed(self) -> None:
        try:
            with open(self._seed_path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as exc:  # pylint: disable=W0703
            byte_log.warning("Byte Router seed load failed (%s): %s", self._seed_path, exc)
            return

        samples: list[_KNNSample] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            q = str(item.get("query", "") or "")
            if not q:
                continue
            label = float(item.get("label", 0.5))
            emb = item.get("embedding") or None
            if emb is None and self._embedding_fn is not None:
                try:
                    emb = list(self._embedding_fn(q))
                except Exception:  # pylint: disable=W0703
                    emb = None
            if emb is None:
                # Skip samples without embeddings we can compute.
                continue
            samples.append(_KNNSample(query=q, label=label, embedding=list(emb)))
        if samples:
            self._knn = _KNNScorer(samples, k=5)

    # ── scoring ──
    def score(self, query: str) -> tuple[float, list[str]]:
        """Return (score in [0, 1], list-of-signals)."""
        heuristic, heur_signals = _heuristic_score(query)
        # Blend with KNN if available: 60% heuristic + 40% KNN.
        if self._knn is not None and self._embedding_fn is not None:
            try:
                emb = list(self._embedding_fn(query))
                knn_score, knn_signals = self._knn.score(emb)
                combined = 0.6 * heuristic + 0.4 * knn_score
                return combined, heur_signals + knn_signals
            except Exception as exc:  # pylint: disable=W0703
                byte_log.debug("Byte Router KNN inference failed, falling back to heuristic: %s", exc)
        return heuristic, heur_signals

    # ── full decision ──
    def decide(
        self,
        query: str,
        *,
        cheap_model: str,
        strong_model: str,
        default_model: str = "",
    ) -> ByteRouterDecision:
        score, signals = self.score(query)
        if score < self.threshold and cheap_model:
            return ByteRouterDecision(
                selected_model=cheap_model,
                score=score,
                tier="cheap",
                reason=f"Byte Router: score {score:.3f} < threshold {self.threshold:.3f}",
                signals=signals,
            )
        if score >= self.threshold and strong_model:
            return ByteRouterDecision(
                selected_model=strong_model,
                score=score,
                tier="strong",
                reason=f"Byte Router: score {score:.3f} >= threshold {self.threshold:.3f}",
                signals=signals,
            )
        # Either cheap or strong not configured — fall through.
        return ByteRouterDecision(
            selected_model=default_model or strong_model or cheap_model,
            score=score,
            tier="cheap" if score < self.threshold else "strong",
            reason="Byte Router: cheap/strong not configured, falling back",
            signals=signals,
        )


# ─── High-level helper for pipeline integration ───────────────────────────


def _get_scorer(
    threshold: float,
    seed_path: str,
    embedding_fn: Callable[[str], list[float]] | None,
) -> ByteRouterScorer:
    """Memoised scorer keyed by (threshold, seed_path) — avoids reloading seeds on every request."""
    key = f"{threshold:.4f}|{seed_path}"
    with _SCORER_LOCK:
        scorer = _SCORER_CACHE.get(key)
        if scorer is None:
            scorer = ByteRouterScorer(
                threshold=threshold,
                seed_path=seed_path or None,
                embedding_fn=embedding_fn,
            )
            _SCORER_CACHE[key] = scorer
        return scorer


def route_decision(
    query: str,
    *,
    cheap_model: str,
    strong_model: str,
    threshold: float = 0.5,
    seed_path: str = "",
    embedding_fn: Callable[[str], list[float]] | None = None,
    default_model: str = "",
) -> ByteRouterDecision:
    """Convenience entry point for pipeline hooks."""
    scorer = _get_scorer(threshold, seed_path, embedding_fn)
    return scorer.decide(
        query,
        cheap_model=cheap_model,
        strong_model=strong_model,
        default_model=default_model,
    )


def reset_scorer_cache() -> None:
    """Clear the memoised scorer cache (useful for tests and config reloads)."""
    with _SCORER_LOCK:
        _SCORER_CACHE.clear()


__all__ = [
    "ByteRouterDecision",
    "ByteRouterScorer",
    "reset_scorer_cache",
    "route_decision",
]
