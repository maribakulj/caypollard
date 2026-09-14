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


def _ppmi_svd_from_sequences(
    sequences: Sequence[Sequence[str]],
    *,
    output_ids: Sequence[str],
    dimension: int,
    window: int,
    seed: int,
    metadata: dict[str, Any],
) -> EmbeddingTable:
    """Factorise positive PMI from deterministic walk sequences.

    This is a dependency-light matrix-factorisation analogue of the skip-gram
    objective used by DeepWalk/RDF2Vec. It is intentionally named as a PPMI-SVD
    control rather than passed off as a canonical Word2Vec implementation.
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    tokens = sorted({token for sequence in sequences for token in sequence})
    if len(tokens) < 2:
        raise ValueError("walk corpus must contain at least two unique tokens")
    position = {token: index for index, token in enumerate(tokens)}

    counts: dict[tuple[int, int], float] = {}
    for sequence in sequences:
        for center_index, center in enumerate(sequence):
            left = max(0, center_index - window)
            right = min(len(sequence), center_index + window + 1)
            center_row = position[center]
            for context_index in range(left, right):
                if context_index == center_index:
                    continue
                context_col = position[sequence[context_index]]
                key = (center_row, context_col)
                counts[key] = counts.get(key, 0.0) + 1.0

    if not counts:
        raise ValueError("walk corpus produced no co-occurrence pairs")
    rows = np.fromiter((key[0] for key in counts), dtype=np.int64)
    cols = np.fromiter((key[1] for key in counts), dtype=np.int64)
    values = np.fromiter(counts.values(), dtype=np.float64)
    count_matrix = coo_matrix(
        (values, (rows, cols)), shape=(len(tokens), len(tokens)), dtype=np.float64
    ).tocsr()

    row_sums = np.asarray(count_matrix.sum(axis=1)).ravel()
    col_sums = np.asarray(count_matrix.sum(axis=0)).ravel()
    total = float(values.sum())
    coo = count_matrix.tocoo()
    denominator = row_sums[coo.row] * col_sums[coo.col]
    pmi = np.log((coo.data * total) / denominator)
    positive = pmi > 0
    if not np.any(positive):
        raise ValueError("co-occurrence corpus produced no positive PMI values")
    ppmi = coo_matrix(
        (pmi[positive], (coo.row[positive], coo.col[positive])),
        shape=count_matrix.shape,
        dtype=np.float32,
    ).tocsr()

    actual_dimension = min(dimension, len(tokens) - 1)
    model = TruncatedSVD(n_components=actual_dimension, random_state=seed)
    token_vectors = model.fit_transform(ppmi).astype(np.float32, copy=False)
    token_vectors = l2_normalize(token_vectors)
    requested = tuple(str(item_id) for item_id in output_ids)
    missing = [item_id for item_id in requested if item_id not in position]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"requested output ids are absent from walk vocabulary: {preview}")
    vectors = np.stack([token_vectors[position[item_id]] for item_id in requested])
    payload = dict(metadata)
    payload.update(
        {
            "requested_dimension": dimension,
            "dimension": actual_dimension,
            "window": window,
            "seed": seed,
            "n_walks": len(sequences),
            "vocabulary_size": len(tokens),
            "ppmi_nonzero": int(ppmi.nnz),
            "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
        }
    )
    return EmbeddingTable(ids=requested, vectors=l2_normalize(vectors), metadata=payload)


def node2vec_ppmi_embeddings(
    triples: Iterable[tuple[str, str, str]],
    *,
    dimension: int = 128,
    walks_per_node: int = 10,
    walk_length: int = 20,
    window: int = 5,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
) -> EmbeddingTable:
    """Create a deterministic Node2Vec-style PPMI-SVD topology baseline.

    Relation labels are deliberately ignored. With ``p=q=1`` this reduces to a
    DeepWalk-style unbiased random-walk control. The skip-gram objective is
    approximated by explicit PPMI factorisation for deterministic lightweight
    testing and reproducibility.
    """
    from caypollard.graphs.triples import entity_ids, normalize_triples, triples_digest

    if walks_per_node <= 0 or walk_length < 2:
        raise ValueError("walks_per_node must be positive and walk_length >= 2")
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive")
    rows = normalize_triples(triples)
    nodes = entity_ids(rows)
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for head, _, tail in rows:
        adjacency[head].add(tail)
        adjacency[tail].add(head)

    rng = np.random.default_rng(seed)
    walks: list[list[str]] = []
    for _ in range(walks_per_node):
        for start in nodes:
            walk = [start]
            previous: str | None = None
            current = start
            while len(walk) < walk_length:
                neighbours = sorted(adjacency[current])
                if not neighbours:
                    break
                if previous is None:
                    next_node = neighbours[int(rng.integers(len(neighbours)))]
                else:
                    weights = []
                    previous_neighbours = adjacency[previous]
                    for candidate in neighbours:
                        if candidate == previous:
                            weight = 1.0 / p
                        elif candidate in previous_neighbours:
                            weight = 1.0
                        else:
                            weight = 1.0 / q
                        weights.append(weight)
                    probabilities = np.asarray(weights, dtype=np.float64)
                    probabilities /= probabilities.sum()
                    next_node = neighbours[int(rng.choice(len(neighbours), p=probabilities))]
                walk.append(next_node)
                previous, current = current, next_node
            walks.append(walk)

    return _ppmi_svd_from_sequences(
        walks,
        output_ids=nodes,
        dimension=dimension,
        window=window,
        seed=seed,
        metadata={
            "family": "graph",
            "method": "node2vec-style-ppmi-svd",
            "relation_aware": False,
            "walks_per_node": walks_per_node,
            "walk_length": walk_length,
            "p": p,
            "q": q,
            "n_triples": len(rows),
            "triples_sha256": triples_digest(rows),
            "warning": (
                "dependency-light Node2Vec/DeepWalk-style control using PPMI-SVD, "
                "not the canonical Word2Vec implementation"
            ),
        },
    )


def rdf2vec_ppmi_embeddings(
    triples: Iterable[tuple[str, str, str]],
    *,
    dimension: int = 128,
    walks_per_entity: int = 10,
    walk_length: int = 8,
    window: int = 5,
    seed: int = 42,
    include_inverse: bool = True,
) -> EmbeddingTable:
    """Create a deterministic predicate-aware RDF2Vec-style PPMI embedding.

    Walk sequences alternate entity and predicate tokens, preserving relation
    identity. Inverse predicate tokens are optionally added so entities that only
    occur as tails remain traversable. This is a transparent RDF2Vec-style
    control, not a claim to reproduce a particular Word2Vec implementation.
    """
    from caypollard.graphs.triples import entity_ids, normalize_triples, triples_digest

    if walks_per_entity <= 0 or walk_length <= 0:
        raise ValueError("walks_per_entity and walk_length must be positive")
    rows = normalize_triples(triples)
    entities = entity_ids(rows)
    transitions: dict[str, list[tuple[str, str]]] = {entity: [] for entity in entities}
    for head, relation, tail in rows:
        transitions[head].append((f"rel:{relation}", tail))
        if include_inverse:
            transitions[tail].append((f"rel:{relation}^-1", head))
    for entity in transitions:
        transitions[entity].sort()

    rng = np.random.default_rng(seed)
    walks: list[list[str]] = []
    for _ in range(walks_per_entity):
        for start in entities:
            sequence = [start]
            current = start
            for _step in range(walk_length):
                options = transitions[current]
                if not options:
                    break
                relation_token, next_entity = options[int(rng.integers(len(options)))]
                sequence.extend((relation_token, next_entity))
                current = next_entity
            walks.append(sequence)

    return _ppmi_svd_from_sequences(
        walks,
        output_ids=entities,
        dimension=dimension,
        window=window,
        seed=seed,
        metadata={
            "family": "graph",
            "method": "rdf2vec-style-ppmi-svd",
            "relation_aware": True,
            "walks_per_entity": walks_per_entity,
            "walk_length": walk_length,
            "include_inverse": include_inverse,
            "n_triples": len(rows),
            "n_relations": len({relation for _, relation, _ in rows}),
            "triples_sha256": triples_digest(rows),
            "warning": (
                "predicate-aware RDF2Vec-style control using PPMI-SVD, not the "
                "canonical RDF2Vec Word2Vec implementation"
            ),
        },
    )


def pykeen_kge_embeddings(
    triples: Iterable[tuple[str, str, str]],
    *,
    model: str,
    dimension: int = 256,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 256,
    device: str | None = None,
) -> EmbeddingTable:
    """Train ComplEx or RotatE with PyKEEN and return real-valued entity vectors.

    Complex-valued entity representations are converted to real vectors by
    concatenating real and imaginary components. PyKEEN is imported lazily so
    the core research package remains usable without the optional ``graph``
    dependency group.
    """
    from importlib.metadata import version

    from caypollard.graphs.triples import normalize_triples, triples_digest

    model_aliases = {"complex": "ComplEx", "rotate": "RotatE"}
    key = model.strip().lower()
    if key not in model_aliases:
        raise ValueError("model must be either 'complex' or 'rotate'")
    if dimension <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("dimension, epochs, and batch_size must be positive")
    try:
        import torch
        from pykeen.models import ComplEx, RotatE
        from pykeen.training import SLCWATrainingLoop
        from pykeen.triples import TriplesFactory
    except ImportError as exc:
        raise RuntimeError(
            "PyKEEN KGE requested but the graph extra is not installed; "
            "install with `pip install -e '.[graph]'` or `uv sync --extra graph`"
        ) from exc

    rows = normalize_triples(triples)
    labeled = np.asarray(rows, dtype=str)
    factory = TriplesFactory.from_labeled_triples(labeled, create_inverse_triples=False)
    model_class = {"complex": ComplEx, "rotate": RotatE}[key]
    model_instance = model_class(
        triples_factory=factory,
        embedding_dim=dimension,
        random_seed=seed,
    )
    if device is not None:
        model_instance = model_instance.to(torch.device(device))
    training_loop = SLCWATrainingLoop(
        model=model_instance,
        triples_factory=factory,
        optimizer="Adam",
    )
    losses = training_loop.train(
        triples_factory=factory,
        num_epochs=epochs,
        batch_size=batch_size,
        use_tqdm=False,
        use_tqdm_batch=False,
    )
    if losses is None:
        raise RuntimeError("PyKEEN training returned no loss history")
    representation_modules = model_instance.entity_representations
    if len(representation_modules) != 1:
        raise RuntimeError("expected exactly one entity representation from PyKEEN model")
    with torch.no_grad():
        tensor = representation_modules[0](indices=None).detach().cpu()
    raw = tensor.numpy()
    if np.iscomplexobj(raw):
        raw = np.concatenate((raw.real, raw.imag), axis=-1)
    matrix = np.asarray(raw, dtype=np.float32).reshape(factory.num_entities, -1)
    ids = tuple(
        label for label, _ in sorted(factory.entity_to_id.items(), key=lambda item: item[1])
    )
    return EmbeddingTable(
        ids=ids,
        vectors=l2_normalize(matrix),
        metadata={
            "family": "graph",
            "method": f"pykeen-{model_aliases[key].lower()}",
            "model": model_aliases[key],
            "relation_aware": True,
            "requested_dimension": dimension,
            "dimension": int(matrix.shape[1]),
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": str(model_instance.device),
            "final_training_loss": float(losses[-1]),
            "n_entities": factory.num_entities,
            "n_relations": factory.num_relations,
            "n_triples": len(rows),
            "triples_sha256": triples_digest(rows),
            "pykeen_version": version("pykeen"),
            "torch_version": torch.__version__,
        },
    )
