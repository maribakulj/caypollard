"""Pre-fusion mining of visual/iconographic disagreement pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from caypollard.benchmarks.iconclass_retrieval import IconclassRelevanceIndex
from caypollard.embeddings.store import EmbeddingTable, l2_normalize
from caypollard.provenance import write_jsonl


PAIR_CLASSES = (
    "easy_positive",
    "hard_negative",
    "hard_positive",
    "easy_negative",
)


@dataclass(frozen=True)
class HardPairThresholds:
    """Frozen thresholds used to map two continuous similarities to pair classes."""

    visual_close_min: float
    visual_distant_max: float
    semantic_close_min: float = 0.5
    semantic_distant_max: float = 0.2

    def __post_init__(self) -> None:
        if self.visual_close_min <= self.visual_distant_max:
            raise ValueError("visual_close_min must exceed visual_distant_max")
        if self.semantic_close_min <= self.semantic_distant_max:
            raise ValueError("semantic_close_min must exceed semantic_distant_max")
        for value in (self.semantic_close_min, self.semantic_distant_max):
            if not 0.0 <= value <= 1.0:
                raise ValueError("semantic thresholds must lie in [0, 1]")


@dataclass(frozen=True)
class HardPair:
    query_id: str
    candidate_id: str
    pair_class: str
    visual_similarity: float
    semantic_similarity: float
    query_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_visual_thresholds(
    table: EmbeddingTable,
    validation_ids: Sequence[str],
    *,
    sample_pairs: int = 20_000,
    close_quantile: float = 0.95,
    distant_quantile: float = 0.50,
    seed: int = 42,
    semantic_close_min: float = 0.5,
    semantic_distant_max: float = 0.2,
) -> tuple[HardPairThresholds, dict[str, Any]]:
    """Calibrate visual cutoffs from validation embeddings only.

    The predeclared default defines visually close pairs as the top 5% of the
    validation random-pair similarity distribution and visually distant pairs as
    those at or below its median. Test embeddings must not be used here.
    """
    if sample_pairs <= 0:
        raise ValueError("sample_pairs must be positive")
    if not 0.0 < distant_quantile < close_quantile < 1.0:
        raise ValueError("require 0 < distant_quantile < close_quantile < 1")
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
            a, b = sorted((ids[int(i)], ids[int(j)]))
            selected.add((a, b))
        pairs = sorted(selected)

    similarities = np.asarray(
        [float(vectors[row[a]] @ vectors[row[b]]) for a, b in pairs], dtype=float
    )
    close_min = float(np.quantile(similarities, close_quantile))
    distant_max = float(np.quantile(similarities, distant_quantile))
    if close_min <= distant_max:
        raise ValueError("validation distribution cannot separate close and distant thresholds")
    thresholds = HardPairThresholds(
        visual_close_min=close_min,
        visual_distant_max=distant_max,
        semantic_close_min=semantic_close_min,
        semantic_distant_max=semantic_distant_max,
    )
    metadata = {
        "method": "validation-random-pair-quantiles",
        "seed": seed,
        "sample_pairs_requested": sample_pairs,
        "sample_pairs_used": len(pairs),
        "close_quantile": close_quantile,
        "distant_quantile": distant_quantile,
        "validation_id_count": len(ids),
        "visual_similarity_mean": float(np.mean(similarities)),
        "visual_similarity_std": float(np.std(similarities)),
        "thresholds": asdict(thresholds),
    }
    return thresholds, metadata


def classify_pair(
    visual_similarity: float,
    semantic_similarity: float,
    thresholds: HardPairThresholds,
) -> str | None:
    """Classify a pair into one of the four preregistered quadrants."""
    visual_close = visual_similarity >= thresholds.visual_close_min
    visual_distant = visual_similarity <= thresholds.visual_distant_max
    semantic_close = semantic_similarity >= thresholds.semantic_close_min
    semantic_distant = semantic_similarity <= thresholds.semantic_distant_max
    if visual_close and semantic_close:
        return "easy_positive"
    if visual_close and semantic_distant:
        return "hard_negative"
    if visual_distant and semantic_close:
        return "hard_positive"
    if visual_distant and semantic_distant:
        return "easy_negative"
    return None


def _top_visual_candidates(
    query_vector: np.ndarray,
    candidate_matrix: np.ndarray,
    candidate_ids: Sequence[str],
    *,
    query_id: str,
    k: int,
    faiss_index: Any | None,
) -> list[tuple[str, float]]:
    if faiss_index is None:
        scores = candidate_matrix @ query_vector
        order = np.argsort(-scores, kind="stable")
        output: list[tuple[str, float]] = []
        for position in order:
            candidate_id = candidate_ids[int(position)]
            if candidate_id == query_id:
                continue
            output.append((candidate_id, float(scores[int(position)])))
            if len(output) == k:
                break
        return output
    scores, indices = faiss_index.search(
        np.ascontiguousarray(query_vector[None, :], dtype=np.float32), min(k + 1, len(candidate_ids))
    )
    output = []
    for score, position in zip(scores[0], indices[0], strict=True):
        candidate_id = candidate_ids[int(position)]
        if candidate_id == query_id:
            continue
        output.append((candidate_id, float(score)))
        if len(output) == k:
            break
    return output


def mine_hard_pairs(
    table: EmbeddingTable,
    records: Sequence[dict[str, Any]],
    parents: dict[str, set[str]],
    thresholds: HardPairThresholds,
    *,
    query_ids: Iterable[str] | None = None,
    candidate_ids: Iterable[str] | None = None,
    visual_top_k: int = 50,
    random_distant_per_query: int = 50,
    max_per_class: int = 250,
    seed: int = 42,
    backend: str = "numpy",
) -> tuple[list[HardPair], dict[str, Any]]:
    """Mine balanced visual/semantic agreement and disagreement pairs.

    Close visual candidates come from exact top-k retrieval. Hard-positive
    candidates come from a local semantic BFS and are retained only when their
    visual score falls below the frozen distant threshold. Random candidates
    supply the easy-negative pool without an O(N²) scan.
    """
    if visual_top_k <= 0 or random_distant_per_query <= 0 or max_per_class <= 0:
        raise ValueError("candidate counts must be positive")
    if backend not in {"numpy", "faiss"}:
        raise ValueError("backend must be 'numpy' or 'faiss'")
    by_id = {str(record["id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("record ids must be unique")
    row = {item_id: index for index, item_id in enumerate(table.ids)}
    available = set(row).intersection(by_id)
    queries = sorted(set(query_ids) if query_ids is not None else available)
    candidates = sorted(set(candidate_ids) if candidate_ids is not None else available)
    if not set(queries).issubset(available) or not set(candidates).issubset(available):
        raise ValueError("query/candidate ids must exist in both embeddings and records")
    if len(candidates) < 2:
        raise ValueError("at least two candidates are required")

    vectors = l2_normalize(table.vectors)
    candidate_matrix = np.stack([vectors[row[item_id]] for item_id in candidates])
    faiss_index = None
    if backend == "faiss":
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS backend requested but retrieval extra is not installed") from exc
        faiss_index = faiss.IndexFlatIP(candidate_matrix.shape[1])
        faiss_index.add(np.ascontiguousarray(candidate_matrix, dtype=np.float32))

    relevance = IconclassRelevanceIndex((by_id[item_id] for item_id in candidates), parents)
    if any(item_id not in relevance.labels_by_id for item_id in queries):
        union = sorted(set(candidates).union(queries))
        relevance = IconclassRelevanceIndex((by_id[item_id] for item_id in union), parents)
        relevance.candidate_ids = frozenset(candidates)
        # Candidate-only inverted indices prevent query-only rows from leaking into BFS results.
        candidate_index = IconclassRelevanceIndex((by_id[item_id] for item_id in candidates), parents)
        relevance.resolved_to_ids = candidate_index.resolved_to_ids
        relevance.exact_to_ids = candidate_index.exact_to_ids

    semantic_max_distance = int(np.floor(1.0 / thresholds.semantic_close_min - 1.0))
    rng = np.random.default_rng(seed)
    pools: dict[str, dict[tuple[str, str], HardPair]] = {name: {} for name in PAIR_CLASSES}

    def consider(query_id: str, candidate_id: str, score: float) -> None:
        if query_id == candidate_id:
            return
        semantic = relevance.hierarchical_relevance(query_id, candidate_id)
        pair_class = classify_pair(score, semantic, thresholds)
        if pair_class is None:
            return
        key = tuple(sorted((query_id, candidate_id)))
        pair = HardPair(
            query_id=query_id,
            candidate_id=candidate_id,
            pair_class=pair_class,
            visual_similarity=float(score),
            semantic_similarity=float(semantic),
            query_labels=tuple(str(x) for x in by_id[query_id].get("iconclass", [])),
            candidate_labels=tuple(str(x) for x in by_id[candidate_id].get("iconclass", [])),
        )
        previous = pools[pair_class].get(key)
        if previous is None:
            pools[pair_class][key] = pair

    candidate_array = np.asarray(candidates, dtype=object)
    for query_id in queries:
        query_vector = vectors[row[query_id]]
        for candidate_id, score in _top_visual_candidates(
            query_vector,
            candidate_matrix,
            candidates,
            query_id=query_id,
            k=min(visual_top_k, len(candidates) - 1),
            faiss_index=faiss_index,
        ):
            consider(query_id, candidate_id, score)

        for candidate_id in sorted(
            relevance.candidate_ids_within_distance(query_id, semantic_max_distance)
        ):
            score = float(query_vector @ vectors[row[candidate_id]])
            if score <= thresholds.visual_distant_max:
                consider(query_id, candidate_id, score)

        sample_size = min(random_distant_per_query, len(candidates) - (query_id in candidates))
        if sample_size > 0:
            eligible = candidate_array[candidate_array != query_id]
            sampled = rng.choice(eligible, size=sample_size, replace=False)
            for raw_candidate in sampled.tolist():
                candidate_id = str(raw_candidate)
                score = float(query_vector @ vectors[row[candidate_id]])
                if score <= thresholds.visual_distant_max:
                    consider(query_id, candidate_id, score)

    sort_key = {
        "easy_positive": lambda pair: (-pair.visual_similarity, -pair.semantic_similarity),
        "hard_negative": lambda pair: (-pair.visual_similarity, pair.semantic_similarity),
        "hard_positive": lambda pair: (-pair.semantic_similarity, pair.visual_similarity),
        "easy_negative": lambda pair: (pair.visual_similarity, pair.semantic_similarity),
    }
    selected: list[HardPair] = []
    available_counts: dict[str, int] = {}
    for pair_class in PAIR_CLASSES:
        values = sorted(pools[pair_class].values(), key=sort_key[pair_class])
        available_counts[pair_class] = len(values)
        selected.extend(values[:max_per_class])
    selected.sort(key=lambda pair: (PAIR_CLASSES.index(pair.pair_class), pair.query_id, pair.candidate_id))
    metadata = {
        "method": "pre-fusion-hard-pair-mining",
        "seed": seed,
        "backend": backend,
        "thresholds": asdict(thresholds),
        "visual_top_k": visual_top_k,
        "random_distant_per_query": random_distant_per_query,
        "semantic_max_distance": semantic_max_distance,
        "max_per_class": max_per_class,
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "available_counts": available_counts,
        "selected_counts": {
            pair_class: sum(pair.pair_class == pair_class for pair in selected)
            for pair_class in PAIR_CLASSES
        },
    }
    return selected, metadata


def write_hard_pairs(pairs: Iterable[HardPair], path: str | Path) -> str:
    """Persist a frozen hard-pair artifact as canonical JSONL."""
    return write_jsonl((pair.to_dict() for pair in pairs), path)
