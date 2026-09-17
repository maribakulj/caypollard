"""Combination rules beyond the weighted sum, and why a project needs them.

Weighted late fusion adds two calibrated scores. That rule has one property the
hard-pair result made visible: a modality whose score saturates dominates the
sum. The context graph gives near-identical high similarity to every plate of a
query's own volume -- often ninety of them -- so those plates take the whole top
of the fused list and evict cross-volume iconographic matches, which are exactly
the hard positives H2 is about.

Each rule below changes how much one modality is allowed to dominate:

``linear``
    The frozen baseline, ``alpha * visual + (1 - alpha) * graph`` on
    validation-calibrated scores. Retained here so every variant is measured
    against it through one code path.
``rank_linear``
    The same weighted sum over within-query rank positions rather than scores. A
    flat plateau of ninety near-equal graph scores becomes ninety distinct
    ranks, so the plateau stops acting as one enormous score.
``rrf``
    Reciprocal rank fusion, ``w / (k + rank)`` summed over modalities. Weight
    decays with rank, so a modality contributes strongly only near its own top
    and cannot flood a list from its middle.
``max``
    The elementwise maximum of the two weighted scores. Either modality alone can
    promote an item, which is the behaviour a hard positive needs: the graph
    should be able to rescue an item the visual encoder placed nowhere.
``product``
    The weighted geometric mean, which promotes only what both modalities like.
    It is the opposite of ``max`` and is included as a control: if hard positives
    improve under ``product`` too, the effect is not about rescue.

Rules that consume ranks ignore score calibration by construction, which is a
second reason to test them: the published gain depends on a min-max calibration
fitted on validation pairs.
"""

from __future__ import annotations

import numpy as np

RULES = ("linear", "rank_linear", "rrf", "max", "product")

DEFAULT_RRF_K = 60.0


def _check(visual: np.ndarray, graph: np.ndarray, alpha: float) -> None:
    if visual.shape != graph.shape:
        raise ValueError("visual and graph score matrices must have identical shapes")
    if visual.ndim != 2:
        raise ValueError("score matrices must be two-dimensional (queries by candidates)")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")


def descending_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank each row from 1 (best) to n, breaking ties by candidate order.

    Ties are broken by position rather than averaged: the evaluator ranks with a
    stable sort on candidate order, and a rank-based rule has to agree with it or
    the two paths disagree on which of two equal-scoring candidates ranks first.
    """
    if scores.ndim != 2:
        raise ValueError("scores must be two-dimensional")
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    positions = np.arange(1, scores.shape[1] + 1)
    np.put_along_axis(ranks, order, np.broadcast_to(positions, order.shape), axis=1)
    return ranks.astype(float)


def _normalised_rank_score(scores: np.ndarray) -> np.ndarray:
    """Map ranks onto [0, 1], best candidate at 1."""
    n = scores.shape[1]
    if n < 2:
        return np.ones_like(scores, dtype=float)
    return 1.0 - (descending_ranks(scores) - 1.0) / (n - 1)


def _unit_interval(scores: np.ndarray) -> np.ndarray:
    """Row-wise min-max onto [0, 1] for the geometric mean, which needs it."""
    low = scores.min(axis=1, keepdims=True)
    high = scores.max(axis=1, keepdims=True)
    width = np.where(high > low, high - low, 1.0)
    return np.clip((scores - low) / width, 0.0, 1.0)


def combine(
    visual: np.ndarray,
    graph: np.ndarray,
    *,
    rule: str,
    alpha: float,
    rrf_k: float = DEFAULT_RRF_K,
) -> np.ndarray:
    """Fuse one batch of query rows under the named rule.

    ``visual`` and ``graph`` are expected already calibrated for the score-based
    rules; the rank-based rules are invariant to any monotone calibration.
    """
    _check(visual, graph, alpha)
    if rule == "linear":
        return alpha * visual + (1.0 - alpha) * graph
    if rule == "rank_linear":
        return alpha * _normalised_rank_score(visual) + (1.0 - alpha) * _normalised_rank_score(
            graph
        )
    if rule == "rrf":
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        return alpha / (rrf_k + descending_ranks(visual)) + (1.0 - alpha) / (
            rrf_k + descending_ranks(graph)
        )
    if rule == "max":
        return np.maximum(alpha * visual, (1.0 - alpha) * graph)
    if rule == "product":
        visual_unit = np.clip(_unit_interval(visual), 1e-9, None)
        graph_unit = np.clip(_unit_interval(graph), 1e-9, None)
        return np.exp(alpha * np.log(visual_unit) + (1.0 - alpha) * np.log(graph_unit))
    raise ValueError(f"unknown fusion rule {rule!r}; expected one of {RULES}")
