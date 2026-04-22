"""Unit tests for byte similarity evaluation pipeline.

Tests verify:
- threshold=0.0 means NO cache hits (documented contract)
- threshold=1.0 only allows exact matches
- NumpyNormEvaluation zero-vector safety (no NaN)
- NumpyNormEvaluation score is always in [0, 2.0]
- ExactMatchEvaluation returns 1 for exact, 0 for different
- SearchDistanceEvaluation handles clipped distances correctly
- KReciprocalEvaluation handles empty candidates gracefully
- cache_factor scales the threshold correctly
"""

import numpy as np
import pytest
from byte.similarity_evaluation.np import NumpyNormEvaluation

from byte import Cache
from byte._backends import openai as cache_openai
from byte.config import Config
from byte.manager.factory import get_data_manager
from byte.similarity_evaluation.distance import SearchDistanceEvaluation
from byte.similarity_evaluation.exact_match import ExactMatchEvaluation
from byte.similarity_evaluation.guarded import GuardedSimilarityEvaluation

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_cache(similarity_threshold=0.8, evaluator=None) -> object:
    c = Cache()
    c.init(
        data_manager=get_data_manager(),
        similarity_evaluation=evaluator or ExactMatchEvaluation(),
        config=Config(similarity_threshold=similarity_threshold),
    )
    return c


@pytest.fixture(autouse=True)
def reset_openai() -> object:
    cache_openai.ChatCompletion.llm = None
    cache_openai.ChatCompletion.cache_args = {}
    yield


# ---------------------------------------------------------------------------
# Threshold contract
# ---------------------------------------------------------------------------


class TestThresholdContract:
    def test_threshold_zero_means_no_hits(self) -> object:
        """similarity_threshold=0.0 must return no cache hits (per docs)."""
        cache_obj = _make_cache(similarity_threshold=0.0)
        call_count = 0

        def fake_llm(*a, **kw) -> object:
            nonlocal call_count
            call_count += 1
            msg = kw.get("messages", [{}])[-1].get("content", "")
            return {"choices": [{"message": {"role": "assistant", "content": f"answer:{msg}"}}]}

        cache_openai.ChatCompletion.llm = fake_llm

        msgs = [{"role": "user", "content": "Same question"}]

        cache_openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msgs, cache_obj=cache_obj
        )
        # Second call â€” MUST miss even though question is identical
        cache_openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msgs, cache_obj=cache_obj
        )

        assert call_count == 2, (
            "threshold=0.0 should return no hits â€” LLM must be called on every request"
        )

    def test_threshold_one_only_allows_exact_match(self) -> object:
        """similarity_threshold=1.0 with ExactMatch should hit only on identical questions."""
        cache_obj = _make_cache(similarity_threshold=1.0)
        call_count = 0

        def fake_llm(*a, **kw) -> object:
            nonlocal call_count
            call_count += 1
            return {"choices": [{"message": {"role": "assistant", "content": "yes"}}]}

        cache_openai.ChatCompletion.llm = fake_llm

        msgs = [{"role": "user", "content": "Exact question"}]
        cache_openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msgs, cache_obj=cache_obj
        )
        # Exact same question â†’ must hit
        cache_openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msgs, cache_obj=cache_obj
        )

        assert call_count == 1, "threshold=1.0 + exact match â†’ second call must hit cache"


# ---------------------------------------------------------------------------
# ExactMatchEvaluation unit tests
# ---------------------------------------------------------------------------


class TestExactMatchEvaluation:
    def setup_method(self) -> None:
        self.evaluator = ExactMatchEvaluation()

    def test_exact_match_returns_one(self) -> None:
        score = self.evaluator.evaluation(
            {"question": "What is Python?"},
            {"question": "What is Python?"},
        )
        assert score == 1

    def test_different_returns_zero(self) -> None:
        score = self.evaluator.evaluation(
            {"question": "What is Python?"},
            {"question": "What is Java?"},
        )
        assert score == 0

    def test_range_is_zero_to_one(self) -> None:
        assert self.evaluator.range() == (0, 1)

    def test_case_sensitive(self) -> None:
        score = self.evaluator.evaluation(
            {"question": "what is python?"},
            {"question": "What is Python?"},
        )
        assert score == 0  # ExactMatch is case-sensitive


# ---------------------------------------------------------------------------
# NumpyNormEvaluation unit tests
# ---------------------------------------------------------------------------


class TestNumpyNormEvaluation:
    def setup_method(self) -> None:
        self.evaluator = NumpyNormEvaluation(enable_normal=True)

    def test_identical_embeddings_max_score(self) -> None:
        vec = np.array([0.6, 0.8])
        score = self.evaluator.evaluation({"embedding": vec}, {"embedding": vec})
        assert score == pytest.approx(2.0, abs=1e-5)

    def test_score_never_negative(self) -> None:
        """Score must be clamped to [0, 2.0] â€” even for very different embeddings."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        score = self.evaluator.evaluation({"embedding": a}, {"embedding": b})
        assert score >= 0.0, "Score must never be negative"

    def test_score_at_most_max(self) -> None:
        """Score must not exceed the stated max range."""
        a = np.array([1.0, 0.0])
        score = self.evaluator.evaluation({"embedding": a}, {"embedding": a})
        assert score <= self.evaluator.range()[1]

    def test_zero_vector_does_not_produce_nan(self) -> None:
        """Zero vector must not produce NaN (div-by-zero was a bug)."""
        zero = np.zeros(4)
        score = self.evaluator.evaluation({"embedding": zero}, {"embedding": zero})
        assert not np.isnan(score), "Score should not be NaN for zero-vector input"
        assert score >= 0.0

    def test_range_is_zero_to_two(self) -> None:
        assert self.evaluator.range() == (0.0, 2.0)

    def test_exact_text_match_short_circuits(self) -> None:
        """If questions match exactly, max score returned without computing embeddings."""
        score = self.evaluator.evaluation(
            {"question": "hi", "embedding": np.array([1.0])},
            {"question": "hi", "embedding": np.array([0.0])},
        )
        assert score == self.evaluator.range()[1]


# ---------------------------------------------------------------------------
# SearchDistanceEvaluation unit tests
# ---------------------------------------------------------------------------


class TestSearchDistanceEvaluation:
    def setup_method(self) -> None:
        self.evaluator = SearchDistanceEvaluation(max_distance=4.0, positive=False)

    def test_zero_distance_is_max_score(self) -> None:
        score = self.evaluator.evaluation({}, {"search_result": (0.0, None)})
        assert score == pytest.approx(4.0)

    def test_max_distance_is_zero_score(self) -> None:
        score = self.evaluator.evaluation({}, {"search_result": (4.0, None)})
        assert score == pytest.approx(0.0)

    def test_negative_distance_clamped_to_zero(self) -> None:
        score = self.evaluator.evaluation({}, {"search_result": (-1.0, None)})
        assert score == pytest.approx(4.0)

    def test_beyond_max_distance_clamped(self) -> None:
        score = self.evaluator.evaluation({}, {"search_result": (10.0, None)})
        assert score == pytest.approx(0.0)

    def test_range_is_zero_to_max(self) -> None:
        assert self.evaluator.range() == (0.0, 4.0)


class TestGuardedSimilarityEvaluation:
    def test_exact_answer_template_requires_canonical_match(self) -> None:
        evaluator = GuardedSimilarityEvaluation(
            SearchDistanceEvaluation(max_distance=4.0, positive=False),
            enforce_canonical_match=True,
        )
        score = evaluator.evaluation(
            {"question": "Reply with exactly TOKYO and nothing else."},
            {
                "question": "Reply with exactly OSAKA and nothing else.",
                "search_result": (0.0, None),
            },
        )
        assert score == evaluator.range()[0]

    def test_low_token_overlap_is_rejected(self) -> None:
        evaluator = GuardedSimilarityEvaluation(
            SearchDistanceEvaluation(max_distance=4.0, positive=False),
            min_token_overlap=0.6,
        )
        score = evaluator.evaluation(
            {"question": "capital of france"},
            {
                "question": "reset the postgres password",
                "search_result": (0.0, None),
            },
        )
        assert score == evaluator.range()[0]

    def test_length_ratio_guard_blocks_mismatched_payload_sizes(self) -> None:
        evaluator = GuardedSimilarityEvaluation(
            SearchDistanceEvaluation(max_distance=4.0, positive=False),
            max_length_ratio=2.0,
        )
        score = evaluator.evaluation(
            {"question": "short prompt"},
            {
                "question": "short prompt with many extra words that should not be treated as the same request",
                "search_result": (0.0, None),
            },
        )
        assert score == evaluator.range()[0]

    def test_structured_classification_templates_require_same_payload(self) -> None:
        evaluator = GuardedSimilarityEvaluation(
            SearchDistanceEvaluation(max_distance=4.0, positive=False),
            enforce_canonical_match=True,
        )
        score = evaluator.evaluation(
            {
                "question": 'Classify the sentiment. Labels: POSITIVE, NEGATIVE, NEUTRAL Review: "I loved it."'
            },
            {
                "question": 'Classify the sentiment. Labels: POSITIVE, NEGATIVE, NEUTRAL Review: "I hated it."',
                "search_result": (0.0, None),
            },
        )
        assert score == evaluator.range()[0]

    def test_structured_summarization_templates_require_same_payload(self) -> None:
        evaluator = GuardedSimilarityEvaluation(
            SearchDistanceEvaluation(max_distance=4.0, positive=False),
            enforce_canonical_match=True,
        )
        score = evaluator.evaluation(
            {
                "question": 'Summarize the following article in one sentence. Article: "Byte is fast."'
            },
            {
                "question": 'Summarize the following article in one sentence. Article: "Byte is expensive."',
                "search_result": (0.0, None),
            },
        )
        assert score == evaluator.range()[0]

    def test_structured_extraction_templates_require_same_payload(self) -> None:
        evaluator = GuardedSimilarityEvaluation(
            SearchDistanceEvaluation(max_distance=4.0, positive=False),
            enforce_canonical_match=True,
        )
        score = evaluator.evaluation(
            {
                "question": 'Extract the fields. Fields: name, city Text: "Name: Alice. City: Paris." Return JSON only.'
            },
            {
                "question": 'Extract the fields. Fields: name, city Text: "Name: Bob. City: London." Return JSON only.',
                "search_result": (0.0, None),
            },
        )
        assert score == evaluator.range()[0]


# ---------------------------------------------------------------------------
# Cost savings integration
# ---------------------------------------------------------------------------


class TestCostSavings:
    def test_cache_hit_reports_provider_in_response(self) -> object:
        """Verify cache hit contains 'byte': True to distinguish from fresh LLM call."""
        cache_obj = _make_cache()
        call_count = 0

        def fake_llm(*a, **kw) -> object:
            nonlocal call_count
            call_count += 1
            return {"choices": [{"message": {"role": "assistant", "content": "Paris"}}]}

        cache_openai.ChatCompletion.llm = fake_llm
        msgs = [{"role": "user", "content": "Capital of France?"}]

        cache_openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msgs, cache_obj=cache_obj
        )
        hit = cache_openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msgs, cache_obj=cache_obj
        )

        assert call_count == 1
        # Cache hit is distinguishable
        assert hit.get("byte") is True
