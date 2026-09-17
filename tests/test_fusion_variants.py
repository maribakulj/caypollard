"""Behavioural tests for the alternative fusion rules."""

from __future__ import annotations

import numpy as np
import pytest

from caypollard.fusion import late_fusion
from caypollard.fusion_variants import RULES, combine, descending_ranks


def test_descending_ranks_breaks_ties_by_candidate_order() -> None:
    ranks = descending_ranks(np.array([[0.5, 0.9, 0.5, 0.1]]))
    # The two 0.5 entries must keep candidate order, as the evaluator's stable
    # sort does, so rank-based rules and the evaluator agree on ties.
    assert ranks.tolist() == [[2.0, 1.0, 3.0, 4.0]]


def test_linear_rule_matches_the_frozen_baseline() -> None:
    visual = np.array([[0.2, 0.8, 0.5]])
    graph = np.array([[0.9, 0.1, 0.4]])
    fused = combine(visual, graph, rule="linear", alpha=0.25)
    np.testing.assert_allclose(fused, late_fusion(visual, graph, alpha=0.25))


def test_rank_linear_ignores_a_monotone_rescaling_of_one_modality() -> None:
    visual = np.array([[0.2, 0.8, 0.5, 0.1]])
    graph = np.array([[0.9, 0.1, 0.4, 0.3]])
    plain = combine(visual, graph, rule="rank_linear", alpha=0.5)
    rescaled = combine(visual, graph * 7.0 + 3.0, rule="rank_linear", alpha=0.5)
    np.testing.assert_allclose(plain, rescaled)


def test_rrf_damps_a_saturated_plateau_that_linear_fusion_lets_dominate() -> None:
    # The flooding case, at realistic scale: ninety candidates share one high
    # graph score -- a query's own volume -- and one distant candidate is the
    # only good visual match. Linear fusion buries it beneath the whole plateau;
    # reciprocal rank fusion cannot, because a plateau member ranked ninetieth
    # by the graph contributes almost nothing.
    visual = np.full((1, 100), 0.05)
    visual[0, 99] = 0.95
    graph = np.full((1, 100), 0.10)
    graph[0, :90] = 0.90

    def rank_of(scores: np.ndarray, index: int) -> int:
        return int(descending_ranks(scores)[0, index])

    linear_rank = rank_of(combine(visual, graph, rule="linear", alpha=0.25), 99)
    rrf_rank = rank_of(combine(visual, graph, rule="rrf", alpha=0.25), 99)
    assert linear_rank > 90
    assert rrf_rank < linear_rank


def test_max_lets_either_modality_promote_alone() -> None:
    visual = np.array([[0.0, 0.9]])
    graph = np.array([[0.9, 0.0]])
    fused = combine(visual, graph, rule="max", alpha=0.5)
    np.testing.assert_allclose(fused, [[0.45, 0.45]])


def test_product_requires_both_modalities_to_agree() -> None:
    # Candidate 2 is liked by both, candidate 0 only by the graph, candidate 1
    # by neither strongly. Agreement has to win.
    visual = np.array([[0.1, 0.5, 0.9, 0.0]])
    graph = np.array([[0.9, 0.5, 0.9, 0.0]])
    fused = combine(visual, graph, rule="product", alpha=0.5)
    assert fused[0, 2] > fused[0, 1] > fused[0, 0]


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_preserves_shape_and_finiteness(rule: str) -> None:
    generator = np.random.default_rng(0)
    visual = generator.normal(size=(4, 9))
    graph = generator.normal(size=(4, 9))
    fused = combine(visual, graph, rule=rule, alpha=0.4)
    assert fused.shape == (4, 9)
    assert np.isfinite(fused).all()


@pytest.mark.parametrize("rule", RULES)
def test_alpha_outside_the_unit_interval_is_rejected(rule: str) -> None:
    visual = np.zeros((2, 3))
    with pytest.raises(ValueError, match="alpha"):
        combine(visual, visual, rule=rule, alpha=1.5)


def test_unknown_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fusion rule"):
        combine(np.zeros((1, 2)), np.zeros((1, 2)), rule="mystery", alpha=0.5)


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        combine(np.zeros((1, 2)), np.zeros((1, 3)), rule="linear", alpha=0.5)
