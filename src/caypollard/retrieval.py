"""Transparent exact retrieval baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vision.store import l2_normalize


@dataclass(frozen=True)
class SearchResult:
    index: int
    score: float


def cosine_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Return exact cosine similarities for one query against candidate rows."""
    matrix = np.asarray(candidates, dtype=np.float32)
    vector = np.asarray(query, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("candidates must be a two-dimensional matrix")
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[1]:
        raise ValueError("query must be one vector with the candidate dimension")
    query_norm = float(np.linalg.norm(vector))
    if query_norm == 0:
        raise ValueError("query cannot be a zero vector")
    normalized_candidates = l2_normalize(matrix)
    normalized_query = vector / query_norm
    return normalized_candidates @ normalized_query


def top_k_cosine(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    k: int,
    exclude_index: int | None = None,
) -> list[SearchResult]:
    """Rank candidates by exact cosine similarity with deterministic tie handling."""
    if k <= 0:
        raise ValueError("k must be positive")
    scores = cosine_scores(query, candidates)
    if exclude_index is not None:
        if exclude_index < 0 or exclude_index >= len(scores):
            raise IndexError("exclude_index is outside the candidate matrix")
        scores = scores.copy()
        scores[exclude_index] = -np.inf
    order = np.argsort(-scores, kind="stable")
    if exclude_index is not None:
        order = order[order != exclude_index]
    order = order[: min(k, len(order))]
    return [SearchResult(index=int(index), score=float(scores[index])) for index in order]
