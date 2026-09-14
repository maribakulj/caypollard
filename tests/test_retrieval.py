import numpy as np
import pytest

from caypollard.retrieval import cosine_scores, top_k_cosine


def test_cosine_scores_and_ranking():
    candidates = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    scores = cosine_scores(np.array([1.0, 0.0], dtype=np.float32), candidates)
    assert scores[0] == pytest.approx(1.0)
    results = top_k_cosine(candidates[0], candidates, k=2, exclude_index=0)
    assert [result.index for result in results] == [1, 2]


def test_invalid_search_inputs():
    candidates = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError):
        top_k_cosine(candidates[0], candidates, k=0)
    with pytest.raises(IndexError):
        top_k_cosine(candidates[0], candidates, k=1, exclude_index=4)
    with pytest.raises(ValueError):
        cosine_scores(np.array([0.0, 0.0], dtype=np.float32), candidates)


def test_excluded_query_is_never_returned_even_when_k_is_large():
    candidates = np.eye(3, dtype=np.float32)
    results = top_k_cosine(candidates[0], candidates, k=10, exclude_index=0)
    assert len(results) == 2
    assert 0 not in [result.index for result in results]
