"""Diagnostics for graph-embedding hubness and degree effects."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr

from caypollard.embeddings.store import EmbeddingTable
from caypollard.graphs.triples import normalize_triples
from caypollard.retrieval import top_k_cosine


def entity_degrees(triples: Iterable[Sequence[str]]) -> dict[str, int]:
    """Return undirected incident-edge counts for graph entities."""
    rows = normalize_triples(triples)
    counts: Counter[str] = Counter()
    for head, _, tail in rows:
        counts[head] += 1
        counts[tail] += 1
    return dict(counts)


def neighbor_occurrence_counts(table: EmbeddingTable, *, k: int = 10) -> dict[str, int]:
    """Count how often each item occurs in another item's exact top-k list."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(table.ids) < 2:
        raise ValueError("at least two embeddings are required")
    effective_k = min(k, len(table.ids) - 1)
    counts: Counter[str] = Counter({item_id: 0 for item_id in table.ids})
    for index, vector in enumerate(table.vectors):
        for result in top_k_cosine(
            vector,
            table.vectors,
            k=effective_k,
            exclude_index=index,
        ):
            counts[table.ids[result.index]] += 1
    return dict(counts)


def degree_hubness_correlation(
    triples: Iterable[Sequence[str]],
    table: EmbeddingTable,
    *,
    k: int = 10,
) -> dict[str, float | int | None]:
    """Measure whether high-degree graph nodes dominate embedding neighbourhoods."""
    degrees = entity_degrees(triples)
    occurrences = neighbor_occurrence_counts(table, k=k)
    shared = sorted(set(degrees).intersection(occurrences))
    if len(shared) < 3:
        return {"n_entities": len(shared), "spearman_rho": None, "p_value": None}
    degree_values = np.asarray([degrees[item_id] for item_id in shared], dtype=float)
    occurrence_values = np.asarray([occurrences[item_id] for item_id in shared], dtype=float)
    if np.all(degree_values == degree_values[0]) or np.all(
        occurrence_values == occurrence_values[0]
    ):
        return {"n_entities": len(shared), "spearman_rho": None, "p_value": None}
    result = spearmanr(degree_values, occurrence_values)
    return {
        "n_entities": len(shared),
        "spearman_rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }
