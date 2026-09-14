"""Iconclass-aware evaluation for image-embedding retrieval baselines.

The module keeps ranking and relevance separate: image embeddings determine the
ranking, while held-out Iconclass structure supplies independent graded and
exact-label relevance judgements.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any

import numpy as np

from caypollard.embeddings.store import EmbeddingTable, l2_normalize
from caypollard.evaluation import average_precision, ndcg_at_k, reciprocal_rank
from caypollard.graphs.iconclass import (
    hierarchy_depth,
    image_hierarchical_similarity,
    resolve_notation,
)


@dataclass(frozen=True)
class RetrievedItem:
    """One persisted top-ranked result for qualitative inspection."""

    item_id: str
    score: float
    hierarchical_relevance: float
    exact_label_overlap: bool


@dataclass(frozen=True)
class QueryEvaluation:
    """Metrics and diagnostics for one retrieval query."""

    query_id: str
    n_candidates: int
    n_exact_relevant: int
    ndcg_at_10: float | None
    reciprocal_rank: float | None
    average_precision: float | None
    recall_at_1: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    median_query_label_frequency: float
    median_query_hierarchy_depth: float | None
    top_results: tuple[RetrievedItem, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["top_results"] = [asdict(item) for item in self.top_results]
        return payload


class IconclassRelevanceIndex:
    """Indexes candidate annotations for exact and hierarchy-aware relevance."""

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        parents: dict[str, set[str]],
    ) -> None:
        rows = list(records)
        self.labels_by_id: dict[str, tuple[str, ...]] = {
            str(row["id"]): tuple(str(label) for label in row.get("iconclass", [])) for row in rows
        }
        if len(self.labels_by_id) != len(rows):
            raise ValueError("record ids must be unique")
        self.parents = parents
        self.candidate_ids = frozenset(self.labels_by_id)

        exact: dict[str, set[str]] = defaultdict(set)
        resolved: dict[str, set[str]] = defaultdict(set)
        frequencies: Counter[str] = Counter()
        for item_id, labels in self.labels_by_id.items():
            for label in labels:
                exact[label].add(item_id)
                frequencies[label] += 1
                node = resolve_notation(label, parents)
                if node is not None:
                    resolved[node].add(item_id)
        self.exact_to_ids = dict(exact)
        self.resolved_to_ids = dict(resolved)
        self.label_frequencies = frequencies

        adjacency: dict[str, set[str]] = defaultdict(set)
        for child, node_parents in parents.items():
            adjacency.setdefault(child, set())
            for parent in node_parents:
                adjacency[child].add(parent)
                adjacency[parent].add(child)
        self.adjacency = dict(adjacency)

    def exact_relevant_ids(self, query_id: str) -> set[str]:
        """Candidates sharing at least one exact raw Iconclass notation."""
        labels = self.labels_by_id[query_id]
        output: set[str] = set()
        for label in labels:
            output.update(self.exact_to_ids.get(label, set()))
        output.discard(query_id)
        return output

    def hierarchical_relevance(self, query_id: str, candidate_id: str) -> float:
        return image_hierarchical_similarity(
            self.labels_by_id[query_id],
            self.labels_by_id[candidate_id],
            self.parents,
        )

    def candidate_ids_within_distance(self, query_id: str, max_distance: int) -> set[str]:
        """Return candidate IDs reachable within a hierarchy distance cutoff.

        The search is a multi-source BFS from all resolved query labels and uses
        the candidate-only inverted index, so it scales with the local hierarchy
        neighbourhood rather than with every possible image pair.
        """
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative")
        query_nodes = {
            node
            for label in self.labels_by_id[query_id]
            if (node := resolve_notation(label, self.parents)) is not None
        }
        if not query_nodes:
            return set()
        distances = dict.fromkeys(query_nodes, 0)
        queue: deque[str] = deque(sorted(query_nodes))
        output: set[str] = set()
        while queue:
            node = queue.popleft()
            distance = distances[node]
            if distance > max_distance:
                continue
            output.update(self.resolved_to_ids.get(node, set()))
            if distance == max_distance:
                continue
            for neighbour in sorted(self.adjacency.get(node, set())):
                if neighbour not in distances:
                    distances[neighbour] = distance + 1
                    queue.append(neighbour)
        output.discard(query_id)
        return output.intersection(self.candidate_ids)

    def ideal_relevance(self, query_id: str, k: int) -> list[float]:
        """Return the exact ideal top-k graded-relevance values efficiently.

        Relevance is ``1 / (1 + d)`` where ``d`` is the minimum hierarchy path
        distance between any resolved query/candidate labels. A multi-source BFS
        over the hierarchy therefore discovers candidates in non-increasing
        relevance order without evaluating every image pair.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        query_nodes = {
            node
            for label in self.labels_by_id[query_id]
            if (node := resolve_notation(label, self.parents)) is not None
        }
        if not query_nodes:
            return [0.0] * k

        distances: dict[str, int] = dict.fromkeys(query_nodes, 0)
        queue: deque[str] = deque(sorted(query_nodes))
        candidate_distance: dict[str, int] = {}
        completed_distance = -1

        while queue:
            node = queue.popleft()
            distance = distances[node]

            # If the previous distance layer already supplied k candidates, no
            # farther node can improve the ideal top-k values.
            if (
                completed_distance >= 0
                and distance > completed_distance
                and len(candidate_distance) >= k
            ):
                break
            completed_distance = distance

            for candidate_id in self.resolved_to_ids.get(node, set()):
                if candidate_id == query_id:
                    continue
                previous = candidate_distance.get(candidate_id)
                if previous is None or distance < previous:
                    candidate_distance[candidate_id] = distance

            for neighbour in sorted(self.adjacency.get(node, set())):
                if neighbour not in distances:
                    distances[neighbour] = distance + 1
                    queue.append(neighbour)

        values = sorted(
            (1.0 / (1.0 + distance) for distance in candidate_distance.values()),
            reverse=True,
        )[:k]
        if len(values) < k:
            values.extend([0.0] * (k - len(values)))
        return values

    def query_diagnostics(self, query_id: str) -> tuple[float, float | None]:
        labels = self.labels_by_id[query_id]
        frequencies = [self.label_frequencies[label] for label in labels]
        depths = [
            depth
            for label in labels
            if (depth := hierarchy_depth(label, self.parents)) is not None
        ]
        return (
            float(median(frequencies)) if frequencies else 0.0,
            float(median(depths)) if depths else None,
        )


def _validate_id_alignment(
    table: EmbeddingTable,
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id = {str(row["id"]): row for row in records}
    if len(by_id) != len(records):
        raise ValueError("manifest ids must be unique")
    missing = sorted(set(table.ids) - set(by_id))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"embedding ids missing from manifest: {preview}")
    return by_id


def _select_ids(
    table: EmbeddingTable,
    by_id: dict[str, dict[str, Any]],
    *,
    split: str | None,
) -> list[str]:
    ids = [
        item_id
        for item_id in table.ids
        if split is None or str(by_id[item_id].get("split")) == split
    ]
    return sorted(ids)


def evaluate_iconclass_retrieval(
    table: EmbeddingTable,
    records: Sequence[dict[str, Any]],
    parents: dict[str, set[str]],
    *,
    query_split: str | None = "test",
    candidate_split: str | None = "test",
    query_limit: int | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
    ndcg_k: int = 10,
    persist_top_k: int = 10,
    batch_size: int = 128,
    backend: str = "numpy",
) -> tuple[dict[str, Any], list[QueryEvaluation]]:
    """Evaluate an embedding table against independent Iconclass relevance.

    Rankings are exact cosine rankings. Queries and candidates are ordered by
    identifier before matrix operations so ties remain deterministic across
    embedding-file row orders.
    """
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    if ndcg_k != 10:
        raise ValueError("protocol-v0.2 keeps the primary endpoint fixed at nDCG@10")
    if persist_top_k <= 0 or batch_size <= 0:
        raise ValueError("persist_top_k and batch_size must be positive")
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be positive when supplied")
    if backend not in {"numpy", "faiss"}:
        raise ValueError("backend must be either 'numpy' or 'faiss'")

    by_id = _validate_id_alignment(table, records)
    query_ids = _select_ids(table, by_id, split=query_split)
    candidate_ids = _select_ids(table, by_id, split=candidate_split)
    if query_limit is not None:
        query_ids = query_ids[:query_limit]
    if not query_ids:
        raise ValueError("no query embeddings match the requested split")
    if not candidate_ids:
        raise ValueError("no candidate embeddings match the requested split")

    row_for_id = {item_id: index for index, item_id in enumerate(table.ids)}
    normalized = l2_normalize(table.vectors)
    candidate_rows = np.asarray([row_for_id[item_id] for item_id in candidate_ids], dtype=int)
    candidate_matrix = normalized[candidate_rows]
    candidate_position = {item_id: index for index, item_id in enumerate(candidate_ids)}

    faiss_index = None
    if backend == "faiss":
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS backend requested but faiss is not installed; "
                "install the retrieval extra"
            ) from exc
        faiss_index = faiss.IndexFlatIP(candidate_matrix.shape[1])
        faiss_index.add(np.ascontiguousarray(candidate_matrix, dtype=np.float32))

    relevance_index = IconclassRelevanceIndex(
        (by_id[item_id] for item_id in candidate_ids), parents
    )
    # Queries may be outside the candidate split. Their labels still need to be
    # available to relevance calculations, so use a second index over the union.
    if any(item_id not in relevance_index.labels_by_id for item_id in query_ids):
        union_ids = sorted(set(candidate_ids).union(query_ids))
        relevance_index = IconclassRelevanceIndex(
            (by_id[item_id] for item_id in union_ids), parents
        )
        relevance_index.candidate_ids = frozenset(candidate_ids)
        # Rebuild candidate-only inverted indices so ideals never use query-only rows.
        candidate_only = IconclassRelevanceIndex(
            (by_id[item_id] for item_id in candidate_ids), parents
        )
        relevance_index.exact_to_ids = candidate_only.exact_to_ids
        relevance_index.resolved_to_ids = candidate_only.resolved_to_ids
        relevance_index.label_frequencies = candidate_only.label_frequencies

    results: list[QueryEvaluation] = []
    max_persist = min(persist_top_k, len(candidate_ids))

    for start in range(0, len(query_ids), batch_size):
        batch_ids = query_ids[start : start + batch_size]
        query_rows = np.asarray([row_for_id[item_id] for item_id in batch_ids], dtype=int)
        query_matrix = normalized[query_rows]
        if backend == "numpy":
            score_matrix = query_matrix @ candidate_matrix.T
            position_matrix = None
        else:
            assert faiss_index is not None
            score_matrix, position_matrix = faiss_index.search(
                np.ascontiguousarray(query_matrix, dtype=np.float32), len(candidate_ids)
            )

        for local_index, query_id in enumerate(batch_ids):
            own_position = candidate_position.get(query_id)
            if backend == "numpy":
                dense_scores = score_matrix[local_index]
                if own_position is not None:
                    dense_scores = dense_scores.copy()
                    dense_scores[own_position] = -np.inf
                order = np.argsort(-dense_scores, kind="stable")
                if own_position is not None:
                    order = order[order != own_position]
                ranked_pairs = [
                    (int(position), float(dense_scores[int(position)])) for position in order
                ]
            else:
                assert position_matrix is not None
                ranked_pairs = [
                    (int(position), float(score))
                    for position, score in zip(
                        position_matrix[local_index], score_matrix[local_index], strict=True
                    )
                    if int(position) >= 0 and int(position) != own_position
                ]
                ranked_pairs.sort(
                    key=lambda pair: (-pair[1], candidate_ids[pair[0]])
                )

            ranked_ids = [candidate_ids[position] for position, _score in ranked_pairs]
            score_by_position = dict(ranked_pairs)

            exact_relevant = relevance_index.exact_relevant_ids(query_id).intersection(
                candidate_ids
            )
            binary = [candidate_id in exact_relevant for candidate_id in ranked_ids]
            n_exact = len(exact_relevant)

            top_for_ndcg = ranked_ids[:ndcg_k]
            graded = [
                relevance_index.hierarchical_relevance(query_id, candidate_id)
                for candidate_id in top_for_ndcg
            ]
            ideal = relevance_index.ideal_relevance(query_id, ndcg_k)
            ndcg_value = ndcg_at_k(graded, ndcg_k, ideal_relevance=ideal) if any(ideal) else None

            if n_exact:
                rr_value = reciprocal_rank(binary)
                ap_value = average_precision(binary, total_relevant=n_exact)
                recalls = {
                    k: sum(binary[:k]) / n_exact
                    for k in ks
                }
            else:
                rr_value = None
                ap_value = None
                recalls = dict.fromkeys(ks)

            frequency, depth = relevance_index.query_diagnostics(query_id)
            persisted = tuple(
                RetrievedItem(
                    item_id=candidate_id,
                    score=score_by_position[candidate_position[candidate_id]],
                    hierarchical_relevance=relevance_index.hierarchical_relevance(
                        query_id, candidate_id
                    ),
                    exact_label_overlap=candidate_id in exact_relevant,
                )
                for candidate_id in ranked_ids[:max_persist]
            )
            results.append(
                QueryEvaluation(
                    query_id=query_id,
                    n_candidates=len(ranked_ids),
                    n_exact_relevant=n_exact,
                    ndcg_at_10=ndcg_value,
                    reciprocal_rank=rr_value,
                    average_precision=ap_value,
                    recall_at_1=recalls.get(1),
                    recall_at_5=recalls.get(5),
                    recall_at_10=recalls.get(10),
                    median_query_label_frequency=frequency,
                    median_query_hierarchy_depth=depth,
                    top_results=persisted,
                )
            )

    ndcg_values = [row.ndcg_at_10 for row in results if row.ndcg_at_10 is not None]
    exact_rows = [row for row in results if row.n_exact_relevant > 0]
    summary: dict[str, Any] = {
        "n_queries": len(results),
        "n_candidates": len(candidate_ids),
        "query_split": query_split,
        "candidate_split": candidate_split,
        "query_limit": query_limit,
        "ranking_backend": backend,
        "ndcg_k": ndcg_k,
        "hierarchical_evaluable_queries": len(ndcg_values),
        "hierarchical_coverage": len(ndcg_values) / len(results),
        "mean_ndcg_at_10": mean(ndcg_values) if ndcg_values else None,
        "exact_evaluable_queries": len(exact_rows),
        "exact_coverage": len(exact_rows) / len(results),
        "mrr": mean(row.reciprocal_rank for row in exact_rows if row.reciprocal_rank is not None)
        if exact_rows
        else None,
        "map": mean(
            row.average_precision for row in exact_rows if row.average_precision is not None
        )
        if exact_rows
        else None,
    }
    for k in ks:
        values = [getattr(row, f"recall_at_{k}", None) for row in exact_rows]
        values = [value for value in values if value is not None]
        summary[f"mean_recall_at_{k}"] = mean(values) if values else None
    summary["strata"] = stratified_metric_summary(results)
    return summary, results


def stratified_metric_summary(rows: Sequence[QueryEvaluation]) -> dict[str, dict[str, Any]]:
    """Summarize primary retrieval quality by transparent frequency/depth bins.

    Frequency bins are based on the median frequency of a query's labels in the
    candidate pool: singleton (<=1), rare (2-10), medium (11-100), frequent
    (>100). Depth bins use median resolved hierarchy depth: shallow (<=2),
    middle (3-5), deep (>=6), or unresolved. These bins are descriptive and are
    not tuned per model.
    """
    frequency_groups: dict[str, list[float]] = defaultdict(list)
    depth_groups: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        if row.ndcg_at_10 is None:
            continue
        frequency = row.median_query_label_frequency
        if frequency <= 1:
            frequency_bin = "singleton"
        elif frequency <= 10:
            frequency_bin = "rare_2_10"
        elif frequency <= 100:
            frequency_bin = "medium_11_100"
        else:
            frequency_bin = "frequent_gt_100"
        frequency_groups[frequency_bin].append(row.ndcg_at_10)

        depth = row.median_query_hierarchy_depth
        if depth is None:
            depth_bin = "unresolved"
        elif depth <= 2:
            depth_bin = "shallow_0_2"
        elif depth <= 5:
            depth_bin = "middle_3_5"
        else:
            depth_bin = "deep_6_plus"
        depth_groups[depth_bin].append(row.ndcg_at_10)

    def summarize(groups: dict[str, list[float]]) -> dict[str, Any]:
        return {
            key: {"n_queries": len(values), "mean_ndcg_at_10": mean(values)}
            for key, values in sorted(groups.items())
        }

    return {
        "label_frequency": summarize(frequency_groups),
        "hierarchy_depth": summarize(depth_groups),
    }
