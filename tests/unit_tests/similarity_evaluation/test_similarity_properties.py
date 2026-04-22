import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from byte.similarity_evaluation.distance import SearchDistanceEvaluation
from byte.similarity_evaluation.numpy_similarity import NumpyNormEvaluation


@st.composite
def same_length_vectors(draw) -> object:
    size = draw(st.integers(min_value=1, max_value=16))
    elements = st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    left = np.asarray(draw(st.lists(elements, min_size=size, max_size=size)), dtype=np.float32)
    right = np.asarray(draw(st.lists(elements, min_size=size, max_size=size)), dtype=np.float32)
    return left, right


@pytest.mark.property
@settings(deadline=None, max_examples=100)
@given(pair=same_length_vectors())
def test_numpy_norm_similarity_is_symmetric_and_bounded(pair) -> None:
    left, right = pair
    evaluator = NumpyNormEvaluation(enable_normal=True)

    score_lr = evaluator.evaluation({"embedding": left}, {"embedding": right})
    score_rl = evaluator.evaluation({"embedding": right}, {"embedding": left})

    assert np.isfinite(score_lr)
    assert np.isfinite(score_rl)
    assert 0.0 <= score_lr <= evaluator.range()[1]
    assert 0.0 <= score_rl <= evaluator.range()[1]
    assert score_lr == pytest.approx(score_rl, abs=1e-5)


@pytest.mark.property
@settings(deadline=None, max_examples=50)
@given(size=st.integers(min_value=1, max_value=32))
def test_numpy_norm_similarity_handles_zero_vectors_without_nan(size) -> None:
    evaluator = NumpyNormEvaluation(enable_normal=True)
    zero = np.zeros(size, dtype=np.float32)

    score = evaluator.evaluation({"embedding": zero}, {"embedding": zero})

    assert np.isfinite(score)
    assert 0.0 <= score <= evaluator.range()[1]


@pytest.mark.property
@settings(deadline=None, max_examples=100)
@given(
    distance=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
)
def test_search_distance_similarity_clamps_scores_into_range(distance) -> None:
    evaluator = SearchDistanceEvaluation(max_distance=4.0, positive=False)

    score = evaluator.evaluation({}, {"search_result": (distance, "cache-id")})

    assert np.isfinite(score)
    assert 0.0 <= score <= evaluator.range()[1]


@pytest.mark.property
@settings(deadline=None, max_examples=100)
@given(
    distance=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
)
def test_search_distance_positive_mode_matches_clamped_distance(distance) -> None:
    evaluator = SearchDistanceEvaluation(max_distance=4.0, positive=True)

    score = evaluator.evaluation({}, {"search_result": (distance, "cache-id")})

    assert np.isfinite(score)
    assert 0.0 <= score <= evaluator.range()[1]
    if distance <= 0.0:
        assert score == pytest.approx(0.0)
    elif distance >= evaluator.max_distance:
        assert score == pytest.approx(evaluator.max_distance)
