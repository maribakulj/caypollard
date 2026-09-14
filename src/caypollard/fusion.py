"""Transparent multimodal fusion baselines with validation-only calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from caypollard.embeddings.store import EmbeddingTable, l2_normalize


@dataclass(frozen=True)
class SimilarityBounds:
    """Validation-derived similarity bounds for one representation family."""

    low: float
    high: float
    method: str = "validation_random_pair_minmax"

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("high must be greater than low")

    @property
    def width(self) -> float:
        return self.high - self.low


def late_fusion(
    visual_scores: np.ndarray,
    graph_scores: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Combine aligned visual and graph similarity scores."""
    if visual_scores.shape != graph_scores.shape:
        raise ValueError("visual_scores and graph_scores must have identical shapes")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return alpha * visual_scores + (1.0 - alpha) * graph_scores


def minmax_scale(scores: np.ndarray, *, low: float, high: float) -> np.ndarray:
    """Scale scores using externally fitted bounds.

    Bounds should normally be estimated on validation data, not test data.
    Values are intentionally not clipped: clipping would change rank geometry
    and would prevent the exact weighted-concatenation equivalence used below.
    """
    if high <= low:
        raise ValueError("high must be greater than low")
    return (scores - low) / (high - low)


def fit_similarity_bounds(
    table: EmbeddingTable,
    validation_ids: Sequence[str],
    *,
    sample_pairs: int = 20_000,
    seed: int = 42,
) -> tuple[SimilarityBounds, dict[str, float | int | str]]:
    """Fit min/max similarity bounds from a fixed validation-pair sample only."""
    if sample_pairs <= 0:
        raise ValueError("sample_pairs must be positive")
    row = {item_id: index for index, item_id in enumerate(table.ids)}
    ids = sorted({str(item_id) for item_id in validation_ids})
    missing = [item_id for item_id in ids if item_id not in row]
    if missing:
        raise ValueError(f"validation ids missing from embeddings: {', '.join(missing[:5])}")
    if len(ids) < 3:
        raise ValueError("at least three validation ids are required")
    vectors = l2_normalize(table.vectors)
    all_pair_count = len(ids) * (len(ids) - 1) // 2
    rng = np.random.default_rng(seed)
    if all_pair_count <= sample_pairs:
        pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    else:
        selected: set[tuple[str, str]] = set()
        while len(selected) < sample_pairs:
            i, j = rng.choice(len(ids), size=2, replace=False)
            selected.add(tuple(sorted((ids[int(i)], ids[int(j)]))))
        pairs = sorted(selected)
    similarities = np.asarray(
        [float(vectors[row[a]] @ vectors[row[b]]) for a, b in pairs], dtype=float
    )
    low = float(np.min(similarities))
    high = float(np.max(similarities))
    if high <= low:
        raise ValueError("validation similarities are constant; cannot fit min-max bounds")
    bounds = SimilarityBounds(low=low, high=high)
    return bounds, {
        "method": bounds.method,
        "seed": seed,
        "sample_pairs_requested": sample_pairs,
        "sample_pairs_used": len(pairs),
        "validation_id_count": len(ids),
        "low": low,
        "high": high,
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
    }


def _aligned_unit_vectors(
    left: EmbeddingTable,
    right: EmbeddingTable,
    ids: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    left_row = {item_id: index for index, item_id in enumerate(left.ids)}
    right_row = {item_id: index for index, item_id in enumerate(right.ids)}
    common = set(left_row).intersection(right_row)
    selected = tuple(sorted(common if ids is None else common.intersection(str(x) for x in ids)))
    if len(selected) < 2:
        raise ValueError("at least two aligned embedding ids are required")
    left_matrix = l2_normalize(np.stack([left.vectors[left_row[item_id]] for item_id in selected]))
    right_matrix = l2_normalize(np.stack([right.vectors[right_row[item_id]] for item_id in selected]))
    return selected, left_matrix, right_matrix


def late_fusion_embedding_table(
    visual: EmbeddingTable,
    graph: EmbeddingTable,
    *,
    alpha: float,
    visual_bounds: SimilarityBounds,
    graph_bounds: SimilarityBounds,
    ids: Iterable[str] | None = None,
) -> EmbeddingTable:
    """Represent min-max late-fusion rankings as one concatenated cosine space.

    For fixed external min-max bounds, additive offsets do not affect ranking.
    The remaining score is a positive weighted sum of two cosine similarities,
    exactly representable by concatenating unit vectors scaled by the square root
    of each effective weight. Row normalization preserves the same ranking.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    aligned_ids, visual_vectors, graph_vectors = _aligned_unit_vectors(visual, graph, ids)
    visual_weight = alpha / visual_bounds.width
    graph_weight = (1.0 - alpha) / graph_bounds.width
    if visual_weight == 0 and graph_weight == 0:
        raise ValueError("at least one modality must have non-zero weight")
    parts = []
    if visual_weight > 0:
        parts.append(np.sqrt(visual_weight) * visual_vectors)
    if graph_weight > 0:
        parts.append(np.sqrt(graph_weight) * graph_vectors)
    matrix = l2_normalize(np.concatenate(parts, axis=1).astype(np.float32, copy=False))
    return EmbeddingTable(
        ids=aligned_ids,
        vectors=matrix,
        metadata={
            "family": "multimodal",
            "method": "validation-minmax-late-fusion",
            "alpha": alpha,
            "visual_bounds": asdict(visual_bounds),
            "graph_bounds": asdict(graph_bounds),
            "visual_source": visual.metadata.get("method") or visual.metadata.get("model_id"),
            "graph_source": graph.metadata.get("method"),
            "ranking_equivalence": "weighted concatenated cosine",
        },
    )


def rerank_visual_candidates(
    visual: EmbeddingTable,
    graph: EmbeddingTable,
    *,
    query_id: str,
    candidate_ids: Sequence[str],
    candidate_k: int,
    alpha: float,
    visual_bounds: SimilarityBounds,
    graph_bounds: SimilarityBounds,
) -> list[tuple[str, float]]:
    """Retrieve visually, then rerank only that candidate set with graph signal."""
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    visual_row = {item_id: index for index, item_id in enumerate(visual.ids)}
    graph_row = {item_id: index for index, item_id in enumerate(graph.ids)}
    shared_candidates = sorted(
        set(str(item_id) for item_id in candidate_ids).intersection(visual_row, graph_row)
    )
    if query_id not in visual_row or query_id not in graph_row:
        raise ValueError("query id must exist in both representations")
    if not shared_candidates:
        raise ValueError("no candidate id exists in both representations")
    v = l2_normalize(visual.vectors)
    g = l2_normalize(graph.vectors)
    visual_scores = np.asarray(
        [float(v[visual_row[query_id]] @ v[visual_row[item_id]]) for item_id in shared_candidates]
    )
    initial_order = np.argsort(-visual_scores, kind="stable")
    initial_ids = [
        shared_candidates[int(index)]
        for index in initial_order
        if shared_candidates[int(index)] != query_id
    ][:candidate_k]
    if not initial_ids:
        return []
    v_scores = np.asarray(
        [float(v[visual_row[query_id]] @ v[visual_row[item_id]]) for item_id in initial_ids]
    )
    g_scores = np.asarray(
        [float(g[graph_row[query_id]] @ g[graph_row[item_id]]) for item_id in initial_ids]
    )
    fused = late_fusion(
        minmax_scale(v_scores, low=visual_bounds.low, high=visual_bounds.high),
        minmax_scale(g_scores, low=graph_bounds.low, high=graph_bounds.high),
        alpha=alpha,
    )
    order = sorted(range(len(initial_ids)), key=lambda i: (-float(fused[i]), initial_ids[i]))
    return [(initial_ids[i], float(fused[i])) for i in order]
