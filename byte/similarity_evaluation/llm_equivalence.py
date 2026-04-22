"""LLM-based equivalence detection for ambiguous cache similarity scores.

Implements the LLM-based equivalence check from 
(Harmohit Singh, January 2026, "Semantic Caching and Intent-Driven Context
Optimization for Multi-Agent Natural Language to Code Systems").

Activates only when vector similarity falls in the configurable ambiguity
band (default: 0.70 – 0.85). Also returns structured adaptation_hints
describing how the cached answer should be adapted for the new query.
"""

from __future__ import annotations

import json
from typing import Any

from byte.similarity_evaluation.similarity_evaluation import SimilarityEvaluation
from byte.utils.log import byte_log

_EQUIVALENCE_PROMPT = """\
You are a cache equivalence judge for a code generation system.

Given two queries, determine if they are semantically equivalent — meaning
they request the same operation and would produce the same correct answer.

Query A (new): {query_a}
Query B (cached): {query_b}
Cached answer snippet: {answer_snippet}

Respond with a JSON object containing:
  "equivalent": true or false
  "confidence": float 0.0 to 1.0
  "adaptation_hints": string describing how to adapt B's answer for A (or "" if equivalent)

JSON only, no other text."""


def _call_llm_equivalence(
    query_a: str,
    query_b: str,
    answer_snippet: str,
    model: str,
    provider_key: str | None,
) -> dict[str, Any]:
    """Call a cheap LLM to determine equivalence. Returns parsed JSON or error dict."""
    try:
        import httpx  # pylint: disable=import-outside-toplevel

        prompt = _EQUIVALENCE_PROMPT.format(
            query_a=str(query_a)[:400],
            query_b=str(query_b)[:400],
            answer_snippet=str(answer_snippet)[:200],
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120,
            "temperature": 0.0,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if provider_key:
            headers["Authorization"] = f"Bearer {provider_key}"

        resp = httpx.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=6.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content.strip())
    except Exception as exc:  # pylint: disable=W0703
        byte_log.debug("LLMEquivalenceEvaluation: call failed — %s", exc)
        return {"equivalent": False, "confidence": 0.0, "adaptation_hints": ""}


class LLMEquivalenceEvaluation(SimilarityEvaluation):
    """LLM-based equivalence check for ambiguous similarity scores.

    Implements  . Wraps a base
    evaluator. When the base score falls in the ambiguity band, an LLM call
    determines true equivalence and populates adaptation_hints in src_dict.

    Usage: wrap your base evaluator:
        evaluator = LLMEquivalenceEvaluation(
            base=NumpyNormEvaluation(),
            equivalence_model="gpt-4o-mini",
        )
    """

    def __init__(
        self,
        base: SimilarityEvaluation,
        *,
        ambiguity_band_low: float = 0.70,
        ambiguity_band_high: float = 0.85,
        equivalence_model: str = "",
        provider_key: str | None = None,
    ) -> None:
        self.base = base
        self.ambiguity_band_low = ambiguity_band_low
        self.ambiguity_band_high = ambiguity_band_high
        self.equivalence_model = equivalence_model or "gpt-4o-mini"
        self.provider_key = provider_key

    def _in_band(self, score: float) -> bool:
        min_r, max_r = self.base.range()
        span = max_r - min_r or 1.0
        normalized = (score - min_r) / span
        return self.ambiguity_band_low <= normalized < self.ambiguity_band_high

    def evaluation(self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **_) -> float:
        """Evaluate similarity. For ambiguous scores, invokes LLM equivalence check.

        Sets src_dict["adaptation_hints"] when an LLM call is made.
        """
        base_score = self.base.evaluation(src_dict, cache_dict)

        if not self._in_band(base_score):
            return base_score

        # Ambiguous band — invoke LLM
        query_a = str(src_dict.get("question", ""))
        query_b = str(cache_dict.get("question", ""))
        answer_snippet = str(cache_dict.get("answer", ""))[:200]

        try:
            from byte.telemetry import (
                bump_research_counter as _bump,  # pylint: disable=import-outside-toplevel
            )
            _bump("llm_equivalence_calls")
        except Exception:  # pragma: no cover - defensive
            pass

        result = _call_llm_equivalence(
            query_a, query_b, answer_snippet,
            model=self.equivalence_model,
            provider_key=self.provider_key,
        )

        src_dict["adaptation_hints"] = result.get("adaptation_hints", "")

        if result.get("equivalent") and float(result.get("confidence", 0.0)) >= 0.6:
            min_r, max_r = self.base.range()
            return max_r  # treat as full match

        return self.base.range()[0]  # reject

    def range(self) -> tuple[float, float]:
        return self.base.range()
