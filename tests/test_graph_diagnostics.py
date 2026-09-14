import numpy as np

from caypollard.embeddings.store import EmbeddingTable
from caypollard.graphs.diagnostics import (
    degree_hubness_correlation,
    entity_degrees,
    neighbor_occurrence_counts,
)

TRIPLES = [
    ("a", "r", "b"),
    ("a", "r", "c"),
    ("a", "r", "d"),
    ("d", "r", "e"),
]


def _table():
    return EmbeddingTable(
        ids=("a", "b", "c", "d", "e"),
        vectors=np.asarray(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        metadata={"fixture": True},
    )


def test_degree_and_hubness_diagnostics_have_all_entities():
    degrees = entity_degrees(TRIPLES)
    assert degrees["a"] == 3
    assert degrees["e"] == 1
    counts = neighbor_occurrence_counts(_table(), k=2)
    assert set(counts) == {"a", "b", "c", "d", "e"}
    assert sum(counts.values()) == 10


def test_degree_hubness_correlation_is_machine_readable():
    result = degree_hubness_correlation(TRIPLES, _table(), k=2)
    assert result["n_entities"] == 5
    assert set(result) == {"n_entities", "spearman_rho", "p_value"}
