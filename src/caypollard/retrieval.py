"""Transparent exact retrieval baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings.store import l2_normalize


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


def neighbor_overlap_at_k(
    left: "EmbeddingTable",
    right: "EmbeddingTable",
    *,
    k: int = 10,
) -> tuple[float, dict[str, float]]:
    """Compare local neighbourhoods from two aligned embedding representations.

    The score for each query is the fraction of its top-k neighbours shared by
    both representations. Only identifiers present in both tables are used.
    """
    from statistics import mean

    from .embeddings.store import EmbeddingTable

    if not isinstance(left, EmbeddingTable) or not isinstance(right, EmbeddingTable):
        raise TypeError("left and right must be EmbeddingTable instances")
    if k <= 0:
        raise ValueError("k must be positive")

    common = sorted(set(left.ids).intersection(right.ids))
    if len(common) < 2:
        raise ValueError("at least two shared ids are required")
    effective_k = min(k, len(common) - 1)
    left_row = {item_id: index for index, item_id in enumerate(left.ids)}
    right_row = {item_id: index for index, item_id in enumerate(right.ids)}
    left_matrix = np.stack([left.vectors[left_row[item_id]] for item_id in common])
    right_matrix = np.stack([right.vectors[right_row[item_id]] for item_id in common])

    per_query: dict[str, float] = {}
    for index, item_id in enumerate(common):
        left_neighbors = {
            common[result.index]
            for result in top_k_cosine(
                left_matrix[index], left_matrix, k=effective_k, exclude_index=index
            )
        }
        right_neighbors = {
            common[result.index]
            for result in top_k_cosine(
                right_matrix[index], right_matrix, k=effective_k, exclude_index=index
            )
        }
        per_query[item_id] = len(left_neighbors.intersection(right_neighbors)) / effective_k
    return mean(per_query.values()), per_query
