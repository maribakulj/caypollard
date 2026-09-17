"""Agreement and human-versus-metric comparison for the expert evaluation.

Phase 10 asks whether a metric improvement corresponds to useful scholarly
retrieval, and insists the answer be read on separate dimensions rather than
collapsed into one "relevance" score. That makes three measurements necessary,
and each has a wrong default that this module avoids.

Agreement between raters on an *ordinal* scale is not chance-corrected by
percentage agreement, and is badly served by unweighted kappa, which counts a
1-versus-5 disagreement the same as 1-versus-2. Krippendorff's alpha with an
ordinal difference function handles both, and handles missing judgements, which
will happen: an expert may decline to rate a dimension they cannot judge.

Agreement between a human ranking and an automatic one is not a correlation of
scores -- the two live on different scales -- but of *orders*, so Kendall's
tau-b, which handles the ties that ordinal ratings produce in quantity.

And a disagreement is not noise to be averaged away. The roadmap asks for them
to be analysed, so they are returned as items, not as a variance.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _ordinal_delta(values: Sequence[float], counts: Mapping[float, float]) -> np.ndarray:
    """Ordinal difference matrix in Krippendorff's formulation.

    The distance between two ordinal categories is not their numeric gap but the
    mass of the categories lying between them, which is what makes the measure
    insensitive to an arbitrary relabelling of a scale's points.
    """
    ordered = list(values)
    size = len(ordered)
    delta = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            low, high = (i, j) if i <= j else (j, i)
            between = sum(counts[ordered[g]] for g in range(low, high + 1))
            correction = (counts[ordered[low]] + counts[ordered[high]]) / 2.0
            delta[i, j] = (between - correction) ** 2
    return delta


def krippendorff_alpha(judgements: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Chance-corrected ordinal agreement across raters, tolerant of gaps.

    ``judgements`` maps a unit identifier to a mapping of rater to rating. Units
    rated by fewer than two raters carry no information about agreement and are
    reported rather than silently dropped.
    """
    usable = {unit: dict(ratings) for unit, ratings in judgements.items() if len(ratings) >= 2}
    skipped = len(judgements) - len(usable)
    if not usable:
        return {
            "alpha": None,
            "n_units": 0,
            "n_units_skipped_single_rater": skipped,
            "reason": "no unit carries two or more ratings",
        }

    categories = sorted({float(value) for ratings in usable.values() for value in ratings.values()})
    if len(categories) == 1:
        return {
            "alpha": None,
            "n_units": len(usable),
            "n_units_skipped_single_rater": skipped,
            "reason": "every rating is identical; agreement is undefined without variation",
        }
    index = {value: position for position, value in enumerate(categories)}

    coincidence = np.zeros((len(categories), len(categories)), dtype=float)
    for ratings in usable.values():
        values = [float(value) for value in ratings.values()]
        weight = len(values) - 1
        for left, right in itertools.permutations(values, 2):
            coincidence[index[left], index[right]] += 1.0 / weight

    marginals = coincidence.sum(axis=1)
    total = marginals.sum()
    counts = {value: marginals[index[value]] for value in categories}
    delta = _ordinal_delta(categories, counts)

    observed = float((coincidence * delta).sum() / total)
    # Expected disagreement draws pairs without replacement from the pooled
    # ratings, so the diagonal counts n_c(n_c - 1), not n_c squared.
    expected_matrix = np.outer(marginals, marginals)
    np.fill_diagonal(expected_matrix, marginals * (marginals - 1))
    expected = float((expected_matrix * delta).sum() / (total * (total - 1)))

    alpha = None if expected == 0 else 1.0 - observed / expected
    return {
        "alpha": None if alpha is None else round(alpha, 4),
        "observed_disagreement": round(observed, 6),
        "expected_disagreement": round(expected, 6),
        "n_units": len(usable),
        "n_units_skipped_single_rater": skipped,
        "n_categories": len(categories),
    }


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Rank correlation that accounts for ties, which ordinal ratings produce."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("both sequences must have the same length")
    if a.size < 2:
        return None
    concordant = discordant = tied_left = tied_right = 0
    for i, j in itertools.combinations(range(a.size), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        if da == 0 and db == 0:
            tied_left += 1
            tied_right += 1
            continue
        if da == 0:
            tied_left += 1
            continue
        if db == 0:
            tied_right += 1
            continue
        if da * db > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_left) * (concordant + discordant + tied_right)
    )
    if denominator == 0:
        return None
    return float((concordant - discordant) / denominator)


def compare_with_metric(
    human: Mapping[str, float],
    metric: Mapping[str, float],
) -> dict[str, Any]:
    """Correlate a human judgement with an automatic score over shared items."""
    shared = sorted(set(human) & set(metric))
    if len(shared) < 2:
        return {"tau_b": None, "n": len(shared), "reason": "fewer than two shared items"}
    return {
        "tau_b": kendall_tau_b([human[k] for k in shared], [metric[k] for k in shared]),
        "n": len(shared),
        "mean_human": round(float(np.mean([human[k] for k in shared])), 4),
        "mean_metric": round(float(np.mean([metric[k] for k in shared])), 4),
    }


def disagreements(
    judgements: Mapping[str, Mapping[str, float]],
    *,
    threshold: float = 2.0,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Return the items raters disagreed on most, as items rather than a variance.

    A mean hides the case the roadmap actually wants looked at: one expert calling
    a retrieval essential and another calling it irrelevant. Those are returned
    whole, worst first.
    """
    rows = []
    for unit, ratings in judgements.items():
        values = [float(value) for value in ratings.values()]
        if len(values) < 2:
            continue
        spread = max(values) - min(values)
        if spread >= threshold:
            rows.append(
                {
                    "unit": unit,
                    "spread": spread,
                    "ratings": dict(ratings),
                    "mean": round(float(np.mean(values)), 3),
                }
            )
    rows.sort(key=lambda row: (-row["spread"], row["unit"]))
    return rows[:limit]


def summarise_dimensions(
    judgements: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    """Agreement per dimension, kept separate exactly as the protocol requires.

    ``judgements`` maps dimension to unit to rater to rating. Collapsing the
    dimensions into one score is the thing phase 10 forbids, so this returns one
    block per dimension and no overall figure.
    """
    per_dimension = {}
    for dimension, units in judgements.items():
        agreement = krippendorff_alpha(units)
        flat: dict[str, list[float]] = defaultdict(list)
        for unit, ratings in units.items():
            flat[unit] = [float(value) for value in ratings.values()]
        per_dimension[dimension] = {
            "agreement": agreement,
            "mean_rating": round(
                float(np.mean([value for values in flat.values() for value in values])), 4
            )
            if flat
            else None,
            "n_units": len(units),
            "n_disagreements": len(disagreements(units, limit=10_000)),
        }
    return per_dimension
