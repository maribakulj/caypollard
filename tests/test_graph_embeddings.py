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


def test_node2vec_ppmi_is_deterministic_relation_agnostic_and_normalized():
    from caypollard.graphs.embeddings import node2vec_ppmi_embeddings

    triples = [
        ("a", "r1", "b"),
        ("b", "r2", "c"),
        ("c", "r1", "d"),
        ("d", "r2", "a"),
        ("a", "r3", "c"),
    ]
    relabeled = [(h, f"changed-{r}", t) for h, r, t in triples]
    first = node2vec_ppmi_embeddings(
        triples, dimension=3, walks_per_node=8, walk_length=8, window=2, seed=7
    )
    second = node2vec_ppmi_embeddings(
        triples, dimension=3, walks_per_node=8, walk_length=8, window=2, seed=7
    )
    changed = node2vec_ppmi_embeddings(
        relabeled, dimension=3, walks_per_node=8, walk_length=8, window=2, seed=7
    )
    assert first.ids == second.ids == changed.ids
    assert np.allclose(first.vectors @ first.vectors.T, second.vectors @ second.vectors.T)
    assert np.allclose(first.vectors @ first.vectors.T, changed.vectors @ changed.vectors.T)
    assert np.allclose(np.linalg.norm(first.vectors, axis=1), 1.0)
    assert first.metadata["relation_aware"] is False


def test_rdf2vec_ppmi_is_deterministic_predicate_aware_and_normalized():
    from caypollard.graphs.embeddings import rdf2vec_ppmi_embeddings

    triples = [
        ("a", "made_by", "x"),
        ("b", "made_by", "x"),
        ("c", "held_by", "x"),
        ("a", "part_of", "y"),
        ("b", "part_of", "z"),
        ("c", "part_of", "z"),
    ]
    first = rdf2vec_ppmi_embeddings(
        triples, dimension=4, walks_per_entity=10, walk_length=6, window=2, seed=11
    )
    second = rdf2vec_ppmi_embeddings(
        triples, dimension=4, walks_per_entity=10, walk_length=6, window=2, seed=11
    )
    assert first.ids == second.ids
    assert np.allclose(first.vectors @ first.vectors.T, second.vectors @ second.vectors.T)
    assert np.allclose(np.linalg.norm(first.vectors, axis=1), 1.0)
    assert first.metadata["relation_aware"] is True
    assert first.metadata["n_relations"] == 3


def test_pykeen_wrapper_fails_helpfully_without_optional_dependency():
    import importlib.util

    if importlib.util.find_spec("pykeen") is not None:
        return
    from caypollard.graphs.embeddings import pykeen_kge_embeddings

    try:
        pykeen_kge_embeddings([("a", "r", "b")], model="complex", epochs=1)
    except RuntimeError as exc:
        assert "graph extra" in str(exc)
    else:
        raise AssertionError("expected missing PyKEEN dependency error")
