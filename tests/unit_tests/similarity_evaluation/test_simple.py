import math

from byte.adapter.api import _get_eval
from byte.similarity_evaluation import SearchDistanceEvaluation


def _test_evaluation_default(evaluation) -> None:
    range_min, range_max = evaluation.range()
    assert math.isclose(range_min, 0.0)
    assert math.isclose(range_max, 4.0)

    score = evaluation.evaluation({}, {"search_result": (1, None)})
    assert math.isclose(score, 3.0)

    score = evaluation.evaluation({}, {"search_result": (-1, None)})
    assert math.isclose(score, 4.0)


def _test_evaluation_config(evaluation) -> None:
    range_min, range_max = evaluation.range()
    assert math.isclose(range_min, 0.0)
    assert math.isclose(range_max, 10.0)

    score = evaluation.evaluation({}, {"search_result": (5, None)})
    assert math.isclose(score, 5.0)
    score = evaluation.evaluation({}, {"search_result": (20, None)})
    assert math.isclose(score, 10.0)


def test_search_distance_evaluation() -> None:
    evaluation = SearchDistanceEvaluation()
    _test_evaluation_default(evaluation)

    evaluation = SearchDistanceEvaluation(max_distance=10, positive=True)
    _test_evaluation_config(evaluation)


def test_get_eval() -> None:
    evaluation = _get_eval("distance")
    _test_evaluation_default(evaluation)

    evaluation = _get_eval(strategy="distance", kws={"max_distance": 10, "positive": True})
    _test_evaluation_config(evaluation)
