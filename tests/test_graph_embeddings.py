from pathlib import Path

import numpy as np

from caypollard.graphs.embeddings import (
    adjacency_svd_embeddings,
    aggregate_concept_embeddings_to_images,
)
from caypollard.graphs.iconclass import build_parent_index, child_edges, parse_notations

FIXTURE = Path("data/samples/iconclass_notations_fixture.txt")


def _graph():
    records = parse_notations(FIXTURE)
    edges = child_edges(records)
    return edges, build_parent_index(edges)


def test_adjacency_svd_graph_embedding_is_deterministic_and_normalized():
    edges, _ = _graph()
    first = adjacency_svd_embeddings(edges, dimension=4, seed=7)
    second = adjacency_svd_embeddings(edges, dimension=4, seed=7)
    assert first.ids == second.ids
    assert np.allclose(first.vectors @ first.vectors.T, second.vectors @ second.vectors.T)
    assert np.allclose(np.linalg.norm(first.vectors, axis=1), 1.0)
    assert first.metadata["method"] == "adjacency-svd"


def test_concept_embeddings_can_be_pooled_to_image_graph_vectors():
    edges, parents = _graph()
    concepts = adjacency_svd_embeddings(edges, dimension=4, seed=7)
    records = [
        {"id": "one", "iconclass": ["25G411"]},
        {"id": "two", "iconclass": ["25G412", "25G41"]},
        {"id": "missing", "iconclass": ["unresolved"]},
    ]
    images = aggregate_concept_embeddings_to_images(records, concepts, parents)
    assert images.ids == ("one", "two")
    assert images.metadata["n_embedded_images"] == 2
    assert images.metadata["image_coverage"] == 2 / 3
    assert np.allclose(np.linalg.norm(images.vectors, axis=1), 1.0)
