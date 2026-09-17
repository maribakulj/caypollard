"""Controls matched on visual similarity, so that only iconography can separate.

A hard positive is a pair that looks unalike and means something alike. Asking
whether a retrieval score recognises one is only meaningful against a comparison
that looks equally unalike and means nothing alike -- otherwise the score is
rewarded for measuring visual distance, which is how the first hard-pair
analysis in this project came to be confounded by digitisation format.

For each hard positive ``(query, positive)`` this module draws a control
``(query, control)`` from the query's own stratum whose visual similarity to the
query is within a tolerance of the positive's, and which shares no label with
the query. Visual similarity then carries no information about which of the two
is the positive, its AUC sits at chance, and any separation above chance comes
from the other modality.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MatchedTriple:
    """One hard positive and the control drawn to be visually indistinguishable."""

    query_id: str
    positive_id: str
    control_id: str
    positive_similarity: float
    control_similarity: float


def paired_auc(positive: Sequence[float], control: Sequence[float]) -> float:
    """How often a hard positive outscores its own control, ties counted half."""
    a = np.asarray(positive, dtype=float)
    b = np.asarray(control, dtype=float)
    if a.shape != b.shape:
        raise ValueError("positive and control score arrays must have identical shapes")
    if a.size == 0:
        raise ValueError("at least one matched pair is required")
    return float((a > b).mean() + 0.5 * (a == b).mean())


def bootstrap_paired_auc(
    positive: Sequence[float],
    control: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap over matched pairs, resampling triples not items."""
    a = np.asarray(positive, dtype=float)
    b = np.asarray(control, dtype=float)
    if a.shape != b.shape:
        raise ValueError("positive and control score arrays must have identical shapes")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, a.size, size=(n_resamples, a.size))
    values = np.asarray([paired_auc(a[index], b[index]) for index in draws])
    tail = (1.0 - confidence) / 2.0 * 100.0
    return {
        "auc": paired_auc(a, b),
        "ci_low": float(np.percentile(values, tail)),
        "ci_high": float(np.percentile(values, 100.0 - tail)),
        "n_pairs": int(a.size),
        "n_resamples": n_resamples,
    }


def match_controls(
    pairs: Sequence[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    similarity: Callable[[str, Sequence[str]], np.ndarray],
    *,
    pool: Sequence[str],
    stratum_key: str = "collection",
    label_key: str = "iconclass",
    tolerance: float = 0.02,
    seed: int = 42,
) -> tuple[list[MatchedTriple], dict[str, int]]:
    """Draw one visually matched, semantically unrelated control per hard positive.

    ``similarity`` returns the visual similarity of a query to a list of
    candidates. ``pool`` is the candidate universe, normally the test split, and
    a control is drawn only from the query's own stratum inside it.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    by_stratum: dict[str, list[str]] = {}
    for item in pool:
        by_stratum.setdefault(str(records[item].get(stratum_key)), []).append(item)

    def labels(item: str) -> set[str]:
        return {str(label) for label in records[item].get(label_key, [])}

    generator = np.random.default_rng(seed)
    matched: list[MatchedTriple] = []
    skipped = {"unknown_item": 0, "empty_stratum": 0, "no_eligible_control": 0}
    for pair in pairs:
        query = str(pair["query_id"])
        positive = str(pair["candidate_id"])
        if query not in records or positive not in records or positive not in set(pool):
            skipped["unknown_item"] += 1
            continue
        candidates = [
            item
            for item in by_stratum.get(str(records[query].get(stratum_key)), [])
            if item not in (query, positive)
        ]
        if not candidates:
            skipped["empty_stratum"] += 1
            continue
        target = float(similarity(query, [positive])[0])
        scores = similarity(query, candidates)
        query_labels = labels(query)
        eligible = [
            (item, float(score))
            for item, score in zip(candidates, scores, strict=True)
            if abs(float(score) - target) <= tolerance and not (labels(item) & query_labels)
        ]
        if not eligible:
            skipped["no_eligible_control"] += 1
            continue
        control, control_score = eligible[int(generator.integers(0, len(eligible)))]
        matched.append(
            MatchedTriple(
                query_id=query,
                positive_id=positive,
                control_id=control,
                positive_similarity=target,
                control_similarity=control_score,
            )
        )
    return matched, skipped
