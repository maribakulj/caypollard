"""Small evaluation utilities used by tests and early notebooks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import mean


def reciprocal_rank(relevant: Sequence[bool]) -> float:
    """Return reciprocal rank for one ranked result list."""
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevant: Sequence[bool], total_relevant: int, k: int) -> float:
    """Return recall@k for a single query."""
    if total_relevant <= 0:
        raise ValueError("total_relevant must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(bool(x) for x in relevant[:k]) / total_relevant


def average_precision(relevant: Sequence[bool], *, total_relevant: int | None = None) -> float:
    """Return average precision for one ranked list.

    ``total_relevant`` should describe the number of relevant candidates in the
    complete evaluation pool. When omitted, it defaults to the number of
    relevant items present in ``relevant``. Supplying it is preferable whenever
    a truncated ranking is evaluated.
    """
    observed_relevant = sum(bool(item) for item in relevant)
    denominator = observed_relevant if total_relevant is None else total_relevant
    if denominator < 0:
        raise ValueError("total_relevant cannot be negative")
    if denominator == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / denominator


def mean_average_precision(
    rankings: Iterable[Sequence[bool]],
    *,
    total_relevant: Iterable[int] | None = None,
) -> float:
    """Return mean average precision across query rankings."""
    rows = list(rankings)
    if not rows:
        return 0.0
    if total_relevant is None:
        return mean(average_precision(row) for row in rows)

    totals = list(total_relevant)
    if len(totals) != len(rows):
        raise ValueError("total_relevant must contain one value per ranking")
    return mean(
        average_precision(row, total_relevant=total)
        for row, total in zip(rows, totals, strict=True)
    )


def dcg_at_k(relevance: Sequence[float], k: int) -> float:
    """Discounted cumulative gain using log2(rank + 1) discounting."""
    import math

    if k <= 0:
        raise ValueError("k must be positive")
    return sum(float(score) / math.log2(rank + 1) for rank, score in enumerate(relevance[:k], 1))


def ndcg_at_k(
    relevance: Sequence[float],
    k: int,
    *,
    ideal_relevance: Sequence[float] | None = None,
) -> float:
    """Normalized discounted cumulative gain for graded relevance.

    When the caller evaluates a truncated ranking, ``ideal_relevance`` can be
    supplied from the complete candidate pool. If omitted, the ideal ordering
    is computed from the provided relevance sequence itself.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    actual = dcg_at_k(relevance, k)
    ideal_source = relevance if ideal_relevance is None else ideal_relevance
    ideal = dcg_at_k(sorted((float(x) for x in ideal_source), reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0
