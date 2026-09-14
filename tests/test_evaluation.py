import pytest

from caypollard.evaluation import (
    average_precision,
    dcg_at_k,
    mean_average_precision,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank():
    assert reciprocal_rank([False, True, True]) == 0.5
    assert reciprocal_rank([False, False]) == 0.0


def test_recall_at_k():
    assert recall_at_k([True, False, True], total_relevant=4, k=2) == 0.25


def test_average_precision():
    assert average_precision([True, False, True]) == pytest.approx((1.0 + 2 / 3) / 2)
    assert average_precision([True, False, True], total_relevant=4) == pytest.approx(
        (1.0 + 2 / 3) / 4
    )
    assert average_precision([False, False]) == 0.0


def test_mean_average_precision():
    value = mean_average_precision([[True, False], [False, True]])
    assert value == pytest.approx((1.0 + 0.5) / 2)
    assert mean_average_precision([]) == 0.0
    with pytest.raises(ValueError):
        mean_average_precision([[True]], total_relevant=[1, 2])


def test_dcg_and_ndcg():
    assert dcg_at_k([1.0, 0.5, 0.0], 3) > 1.0
    assert ndcg_at_k([1.0, 0.5, 0.0], 3) == pytest.approx(1.0)
    assert 0.0 < ndcg_at_k([0.5, 1.0, 0.0], 3) < 1.0
    assert ndcg_at_k([0.5], 1, ideal_relevance=[1.0, 0.5]) == pytest.approx(0.5)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        recall_at_k([True], total_relevant=0, k=1)
    with pytest.raises(ValueError):
        recall_at_k([True], total_relevant=1, k=0)
    with pytest.raises(ValueError):
        average_precision([True], total_relevant=-1)
    with pytest.raises(ValueError):
        dcg_at_k([1.0], 0)
    with pytest.raises(ValueError):
        ndcg_at_k([1.0], 0)
