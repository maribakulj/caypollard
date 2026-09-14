"""Transparent graph-embedding controls for taxonomy experiments.

These baselines are intentionally simple. They establish whether graph topology
is flowing through the pipeline before relation-aware KGE models such as
ComplEx or RotatE are introduced.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, eye
from sklearn.decomposition import TruncatedSVD

from caypollard.embeddings.store import EmbeddingTable, l2_normalize
from caypollard.graphs.iconclass import resolve_notation


def adjacency_svd_embeddings(
    edges: Iterable[tuple[str, str]],
    *,
    dimension: int = 64,
    seed: int = 42,
    self_loops: bool = True,
) -> EmbeddingTable:
    """Embed an undirected graph using truncated SVD of its adjacency matrix.

    Direction is deliberately discarded in this sanity baseline. Relation-aware
    models are a separate experiment and must outperform this control to justify
    their additional complexity.
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    edge_list = sorted({(str(left), str(right)) for left, right in edges})
    nodes = sorted({node for edge in edge_list for node in edge})
    if len(nodes) < 2:
        raise ValueError("at least two graph nodes are required")

    position = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for left, right in edge_list:
        i = position[left]
        j = position[right]
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((1.0, 1.0))

    matrix = coo_matrix(
        (values, (rows, cols)), shape=(len(nodes), len(nodes)), dtype=np.float32
    ).tocsr()
    if self_loops:
        matrix = matrix + eye(len(nodes), dtype=np.float32, format="csr")

    actual_dimension = min(dimension, len(nodes) - 1)
    model = TruncatedSVD(n_components=actual_dimension, random_state=seed)
    vectors = model.fit_transform(matrix).astype(np.float32, copy=False)
    vectors = l2_normalize(vectors)
    metadata: dict[str, Any] = {
        "family": "graph",
        "method": "adjacency-svd",
        "directed": False,
        "self_loops": self_loops,
        "requested_dimension": dimension,
        "dimension": actual_dimension,
        "seed": seed,
        "n_nodes": len(nodes),
        "n_edges": len(edge_list),
        "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
        "warning": (
            "taxonomy-only sanity baseline; evaluating it against the same taxonomy "
            "is not independent evidence of semantic discovery"
        ),
    }
    return EmbeddingTable(ids=tuple(nodes), vectors=vectors, metadata=metadata)


def aggregate_concept_embeddings_to_images(
    records: Sequence[dict[str, Any]],
    concept_embeddings: EmbeddingTable,
    parents: dict[str, set[str]],
) -> EmbeddingTable:
    """Mean-pool resolved concept embeddings into one graph vector per image.

    Images with no resolvable concept represented in ``concept_embeddings`` are
    omitted. Coverage is recorded explicitly in the returned metadata.
    """
    concept_row = {item_id: index for index, item_id in enumerate(concept_embeddings.ids)}
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    unresolved_images: list[str] = []
    resolved_label_count = 0
    total_label_count = 0

    for record in records:
        item_id = str(record["id"])
        rows: list[np.ndarray] = []
        for label in record.get("iconclass", []):
            total_label_count += 1
            node = resolve_notation(str(label), parents)
            if node is not None and node in concept_row:
                resolved_label_count += 1
                rows.append(concept_embeddings.vectors[concept_row[node]])
        if not rows:
            unresolved_images.append(item_id)
            continue
        ids.append(item_id)
        vectors.append(np.mean(np.stack(rows), axis=0))

    if not vectors:
        raise ValueError("no image has a resolvable concept embedding")
    matrix = l2_normalize(np.stack(vectors).astype(np.float32, copy=False))
    metadata = {
        "family": "graph",
        "method": "mean-pooled-concept-embeddings",
        "source_concept_method": concept_embeddings.metadata.get("method"),
        "n_input_images": len(records),
        "n_embedded_images": len(ids),
        "image_coverage": len(ids) / len(records) if records else 0.0,
        "resolved_label_fraction": (
            resolved_label_count / total_label_count if total_label_count else 0.0
        ),
        "unresolved_image_ids": unresolved_images,
        "warning": (
            "taxonomy-derived control representation; not an independent semantic target"
        ),
    }
    return EmbeddingTable(ids=tuple(ids), vectors=matrix, metadata=metadata)
