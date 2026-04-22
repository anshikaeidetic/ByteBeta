"""Quality scorer implementation and response-repair orchestration."""

import hashlib
import threading
import time
from typing import Any

from byte.processor._quality_contracts import (
    _answer_matches_reference,
    _contains_token,
    _expected_structured_format,
    _extract_exact_token,
    _extract_label_candidates,
    _extract_request_text,
    _extract_structured_payload,
    _fields_from_slots,
    _is_generic_exact_token_candidate,
    _labels_from_slots,
    _match_best_label,
    _repair_exact_token_from_task_output,
    _semantic_label_from_request,
)
from byte.processor._quality_evidence import (
    _assess_evidence_support,
    _assess_structured_evidence,
    _assess_style_constraint,
    _collect_evidence_text,
    _evidence_threshold,
    _merge_evidence_assessment,
    _structured_evidence_threshold,
    _summary_evidence_threshold,
)
from byte.processor._quality_models import EvidenceAssessment, ResponseAssessment
from byte.processor.intent import extract_request_intent
from byte.processor.reasoning_reuse import assess_reasoning_answer
from byte.trust import deterministic_reference_answer


class QualityScorer:
    """Scores and tracks the quality of cache entries.

    Quality is computed from:
    - similarity_score: how close the embedding match was (0-1)
    - length_ratio: ratio of cached answer length to expected length
    - user_feedback: thumbs up/down adjustments

    :param auto_evict_threshold: entries below this score are auto-evicted (default 0.2)
    :param feedback_weight: how much a single feedback event adjusts score (default 0.1)
    """

    def __init__(
        self,
        auto_evict_threshold: float = 0.2,
        feedback_weight: float = 0.1,
    ) -> None:
        self._auto_evict_threshold = auto_evict_threshold
        self._feedback_weight = feedback_weight
        self._lock = threading.Lock()

        # query_hash -> {score, hits, feedbacks_up, feedbacks_down, last_access}
        self._entries: dict[str, dict[str, Any]] = {}

        # Stats
        self._total_scored = 0
        self._total_evicted = 0
        self._total_feedback = 0
        self._repair_attempts = 0
        self._repair_successes = 0
        self._escalations = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        query: str,
        cached_answer: str,
        similarity_score: float,
        expected_answer: str | None = None,
    ) -> float:
        """Compute and record a quality score for a cache hit.

        :param query: the user query
        :param cached_answer: the answer returned from cache
        :param similarity_score: the embedding similarity score (0-1)
        :param expected_answer: optional, if known, the "real" answer for length comparison
        :return: quality score (0-1)
        """
        qhash = self._hash(query)

        # Component 1: Similarity (weighted 60%)
        sim_component = max(0.0, min(1.0, similarity_score))

        # Component 2: Length reasonableness (weighted 20%)
        # Answers that are very short or suspiciously long get penalised
        length_component = self._length_score(cached_answer, expected_answer)

        # Component 3: Historical feedback (weighted 20%)
        feedback_component = self._feedback_score(qhash)

        quality = 0.6 * sim_component + 0.2 * length_component + 0.2 * feedback_component
        quality = max(0.0, min(1.0, quality))

        with self._lock:
            if qhash not in self._entries:
                self._entries[qhash] = {
                    "score": quality,
                    "hits": 0,
                    "feedbacks_up": 0,
                    "feedbacks_down": 0,
                    "last_access": time.time(),
                    "query_preview": query[:80],
                }
            entry = self._entries[qhash]
            # Exponential moving average for score stability
            entry["score"] = 0.7 * entry["score"] + 0.3 * quality
            entry["hits"] += 1
            entry["last_access"] = time.time()
            self._total_scored += 1

        return quality

    def record_feedback(self, query: str, thumbs_up: bool) -> dict[str, Any]:
        """Record user feedback for a cached query.

        :param query: the query text
        :param thumbs_up: True for positive, False for negative feedback
        :return: updated entry info
        """
        qhash = self._hash(query)

        with self._lock:
            self._total_feedback += 1
            if qhash not in self._entries:
                self._entries[qhash] = {
                    "score": 0.5,
                    "hits": 0,
                    "feedbacks_up": 0,
                    "feedbacks_down": 0,
                    "last_access": time.time(),
                    "query_preview": query[:80],
                }

            entry = self._entries[qhash]
            if thumbs_up:
                entry["feedbacks_up"] += 1
                entry["score"] = min(1.0, entry["score"] + self._feedback_weight)
            else:
                entry["feedbacks_down"] += 1
                entry["score"] = max(0.0, entry["score"] - self._feedback_weight)

            entry["last_access"] = time.time()

            return {
                "query_hash": qhash,
                "new_score": round(entry["score"], 4),
                "should_evict": entry["score"] < self._auto_evict_threshold,
            }

    def get_low_quality_entries(self, threshold: float | None = None) -> list[dict[str, Any]]:
        """Return entries below the quality threshold.

        :param threshold: override the auto-evict threshold
        :return: list of low-quality entry dicts
        """
        cutoff = threshold if threshold is not None else self._auto_evict_threshold
        with self._lock:
            return [{"query_hash": k, **v} for k, v in self._entries.items() if v["score"] < cutoff]

    def should_evict(self, query: str) -> bool:
        """Check if a query's cached entry should be evicted.

        :param query: query text
        :return: True if the entry score is below auto-evict threshold
        """
        qhash = self._hash(query)
        with self._lock:
            entry = self._entries.get(qhash)
            if entry is None:
                return False
            return entry["score"] < self._auto_evict_threshold

    def stats(self) -> dict[str, Any]:
        """Return quality scoring statistics."""
        with self._lock:
            scores = [e["score"] for e in self._entries.values()]
            return {
                "total_entries_tracked": len(self._entries),
                "total_scored": self._total_scored,
                "total_feedback_received": self._total_feedback,
                "total_evicted": self._total_evicted,
                "avg_quality_score": (round(sum(scores) / len(scores), 4) if scores else 0.0),
                "low_quality_count": sum(1 for s in scores if s < self._auto_evict_threshold),
                "repair_attempts": self._repair_attempts,
                "repair_successes": self._repair_successes,
                "escalations": self._escalations,
            }

    def record_repair(self, *, applied: bool) -> None:
        with self._lock:
            self._repair_attempts += 1
            if applied:
                self._repair_successes += 1

    def record_escalation(self) -> None:
        with self._lock:
            self._escalations += 1

    def assess_request_answer(
        self,
        request_kwargs: dict[str, Any],
        answer: Any,
        *,
        context_hints: dict[str, Any] | None = None,
        config: Any | None = None,
        task_policy: dict[str, Any] | None = None,
    ) -> ResponseAssessment:
        intent = extract_request_intent(request_kwargs or {})
        request_text = _extract_request_text(request_kwargs)
        answer_text = "" if answer is None else str(answer).strip()
        if not answer_text:
            return ResponseAssessment(
                score=0.0,
                accepted=False,
                repaired_answer=None,
                reason="empty_answer",
                constraint="empty",
            )

        evidence_text = _collect_evidence_text(request_kwargs, context_hints)
        evidence_threshold = _evidence_threshold(intent.category, config, task_policy)

        trust_reference = deterministic_reference_answer(
            request_kwargs,
            context_hints=context_hints,
        )
        if trust_reference is not None:
            matched = _answer_matches_reference(trust_reference, answer_text)
            base_assessment = ResponseAssessment(
                score=round(float(trust_reference.score if matched else max(0.94, trust_reference.score - 0.02)), 4),
                accepted=True,
                repaired_answer=trust_reference.answer,
                reason=f"{trust_reference.reason}_verified"
                if matched
                else f"{trust_reference.reason}_repaired",
                constraint=trust_reference.constraint,
            )
            return _merge_evidence_assessment(
                base_assessment,
                _assess_evidence_support(
                    intent=intent,
                    request_kwargs=request_kwargs,
                    request_text=request_text,
                    answer_text=trust_reference.answer,
                    evidence_text=evidence_text,
                    constraint=trust_reference.constraint,
                    min_support=evidence_threshold,
                    config=config,
                    task_policy=task_policy,
                ),
            )

        reasoning_assessment = assess_reasoning_answer(
            request_kwargs,
            answer_text,
            config=config,
        )
        if reasoning_assessment is not None:
            base_assessment = ResponseAssessment(
                score=round(float(reasoning_assessment.get("score", 0.0) or 0.0), 4),
                accepted=bool(reasoning_assessment.get("accepted", False)),
                repaired_answer=reasoning_assessment.get("repaired_answer"),
                reason=str(
                    reasoning_assessment.get("reason", "deterministic_reasoning")
                    or "deterministic_reasoning"
                ),
                constraint=str(
                    reasoning_assessment.get("constraint", "deterministic_reasoning")
                    or "deterministic_reasoning"
                ),
            )
            return _merge_evidence_assessment(
                base_assessment,
                _assess_evidence_support(
                    intent=intent,
                    request_kwargs=request_kwargs,
                    request_text=request_text,
                    answer_text=str(reasoning_assessment.get("repaired_answer", "") or answer_text),
                    evidence_text=evidence_text,
                    constraint=base_assessment.constraint,
                    min_support=evidence_threshold,
                    config=config,
                    task_policy=task_policy,
                ),
            )

        exact_token = str(_extract_exact_token(request_text) or intent.slots.get("token") or "")
        if _is_generic_exact_token_candidate(exact_token):
            exact_token = ""
        if exact_token:
            if _contains_token(answer_text, exact_token):
                return _merge_evidence_assessment(
                    ResponseAssessment(
                        score=1.0,
                        accepted=True,
                        repaired_answer=exact_token,
                        reason="exact_token_matched",
                        constraint="exact_token",
                    ),
                    _assess_evidence_support(
                        intent=intent,
                        request_kwargs=request_kwargs,
                        request_text=request_text,
                        answer_text=exact_token,
                        evidence_text=evidence_text,
                        constraint="exact_token",
                        min_support=evidence_threshold,
                        config=config,
                        task_policy=task_policy,
                    ),
                )
            repaired_exact = _repair_exact_token_from_task_output(
                intent, request_text, answer_text, exact_token
            )
            if repaired_exact is not None:
                return _merge_evidence_assessment(
                    repaired_exact,
                    _assess_evidence_support(
                        intent=intent,
                        request_kwargs=request_kwargs,
                        request_text=request_text,
                        answer_text=exact_token,
                        evidence_text=evidence_text,
                        constraint="exact_token",
                        min_support=evidence_threshold,
                        config=config,
                        task_policy=task_policy,
                    ),
                )
            return ResponseAssessment(
                score=0.05,
                accepted=False,
                repaired_answer=None,
                reason="exact_token_missing",
                constraint="exact_token",
            )

        labels = _labels_from_slots(intent.slots) or _extract_label_candidates(request_text)
        if labels:
            matched = _match_best_label(answer_text, labels)
            semantic_label = _semantic_label_from_request(intent, request_text, labels)
            if semantic_label:
                if matched == semantic_label:
                    return _merge_evidence_assessment(
                        ResponseAssessment(
                            score=0.99,
                            accepted=True,
                            repaired_answer=semantic_label,
                            reason="label_semantics_verified",
                            constraint="label_set",
                        ),
                        _assess_evidence_support(
                            intent=intent,
                            request_kwargs=request_kwargs,
                            request_text=request_text,
                            answer_text=semantic_label,
                            evidence_text=evidence_text,
                            constraint="label_set",
                            min_support=evidence_threshold,
                            config=config,
                            task_policy=task_policy,
                        ),
                    )
                return _merge_evidence_assessment(
                    ResponseAssessment(
                        score=0.82 if matched else 0.76,
                        accepted=True,
                        repaired_answer=semantic_label,
                        reason="label_semantics_repaired"
                        if matched
                        else "label_semantics_inferred",
                        constraint="label_set",
                    ),
                    _assess_evidence_support(
                        intent=intent,
                        request_kwargs=request_kwargs,
                        request_text=request_text,
                        answer_text=semantic_label,
                        evidence_text=evidence_text,
                        constraint="label_set",
                        min_support=evidence_threshold,
                        config=config,
                        task_policy=task_policy,
                    ),
                )
            if matched:
                return _merge_evidence_assessment(
                    ResponseAssessment(
                        score=0.95,
                        accepted=True,
                        repaired_answer=matched,
                        reason="label_matched",
                        constraint="label_set",
                    ),
                    _assess_evidence_support(
                        intent=intent,
                        request_kwargs=request_kwargs,
                        request_text=request_text,
                        answer_text=matched,
                        evidence_text=evidence_text,
                        constraint="label_set",
                        min_support=evidence_threshold,
                        config=config,
                        task_policy=task_policy,
                    ),
                )
            return ResponseAssessment(
                score=0.15,
                accepted=False,
                repaired_answer=None,
                reason="label_missing",
                constraint="label_set",
            )

        structured_format = _expected_structured_format(intent, request_text)
        if structured_format:
            parsed, repaired_answer, field_coverage = _extract_structured_payload(
                answer_text,
                structured_format,
                fields=_fields_from_slots(intent.slots),
            )
            if parsed is not None:
                score = 0.9
                if field_coverage is not None:
                    score = max(0.55, min(0.95, 0.55 + 0.4 * field_coverage))
                accepted = field_coverage is None or field_coverage >= 0.999
                base_assessment = ResponseAssessment(
                    score=round(score, 4),
                    accepted=accepted,
                    repaired_answer=repaired_answer,
                    reason=f"{structured_format}_parsed"
                    if accepted
                    else f"{structured_format}_missing_fields",
                    constraint=structured_format,
                )
                return _merge_evidence_assessment(
                    base_assessment,
                    _assess_structured_evidence(
                        intent=intent,
                        request_kwargs=request_kwargs,
                        request_text=request_text,
                        parsed=parsed,
                        evidence_text=evidence_text,
                        structured_format=structured_format,
                        min_support=_structured_evidence_threshold(
                            intent.category, config, task_policy
                        ),
                        config=config,
                        task_policy=task_policy,
                    ),
                )
            return ResponseAssessment(
                score=0.15,
                accepted=False,
                repaired_answer=None,
                reason=f"{structured_format}_invalid",
                constraint=structured_format,
            )

        style_assessment = _assess_style_constraint(intent, answer_text)
        if style_assessment is not None:
            return _merge_evidence_assessment(
                style_assessment,
                _assess_evidence_support(
                    intent=intent,
                    request_kwargs=request_kwargs,
                    request_text=request_text,
                    answer_text=answer_text,
                    evidence_text=evidence_text,
                    constraint=style_assessment.constraint,
                    min_support=_summary_evidence_threshold(config, task_policy),
                    config=config,
                    task_policy=task_policy,
                ),
            )

        length_score = self._length_score(answer_text, None)
        base_assessment = ResponseAssessment(
            score=round(max(0.45, length_score), 4),
            accepted=bool(answer_text),
            repaired_answer=answer_text,
            reason="freeform_answer",
            constraint="freeform",
        )
        evidence_assessment = _assess_evidence_support(
            intent=intent,
            request_kwargs=request_kwargs,
            request_text=request_text,
            answer_text=answer_text,
            evidence_text=evidence_text,
            constraint="grounded_answer",
            min_support=evidence_threshold,
            config=config,
            task_policy=task_policy,
        )
        if evidence_assessment is None:
            return base_assessment
        return _merge_evidence_assessment(
            base_assessment, evidence_assessment, constraint="grounded_answer"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(query: str) -> str:
        """Create a short hash of the query text."""
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _length_score(cached_answer: str, expected_answer: str | None) -> float:
        """Score based on answer length reasonableness.

        If expected_answer is available, compare lengths directly.
        Otherwise use heuristic: penalise very short (<10 chars) and
        very long (>10000 chars) answers.
        """
        cached_len = len(cached_answer) if cached_answer else 0

        if expected_answer:
            expected_len = len(expected_answer)
            if expected_len == 0:
                return 0.5
            ratio = cached_len / expected_len
            # Perfect = 1.0, deviations penalised
            if 0.5 <= ratio <= 2.0:
                return 1.0
            elif 0.2 <= ratio <= 5.0:
                return 0.5
            else:
                return 0.1

        # Heuristic scoring without expected answer
        if cached_len < 5:
            return 0.2  # suspiciously short
        elif cached_len < 10:
            return 0.5
        elif cached_len > 10000:
            return 0.6  # very long but not necessarily bad
        else:
            return 0.9  # reasonable length

    def _feedback_score(self, qhash: str) -> float:
        """Score component from accumulated user feedback."""
        with self._lock:
            entry = self._entries.get(qhash)
            if not entry:
                return 0.5  # neutral default

            up = entry["feedbacks_up"]
            down = entry["feedbacks_down"]
            total = up + down
            if total == 0:
                return 0.5
            return up / total


__all__ = ["EvidenceAssessment", "QualityScorer", "ResponseAssessment"]
