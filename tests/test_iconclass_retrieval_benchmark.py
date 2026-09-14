from pathlib import Path

import numpy as np

from caypollard.benchmarks.iconclass_retrieval import (
    IconclassRelevanceIndex,
    evaluate_iconclass_retrieval,
)
from caypollard.embeddings.store import EmbeddingTable
from caypollard.graphs.iconclass import build_parent_index, child_edges, parse_notations

FIXTURE = Path("data/samples/iconclass_notations_fixture.txt")


def _parents():
    return build_parent_index(child_edges(parse_notations(FIXTURE)))


def _records():
    return [
        {"id": "a", "iconclass": ["25G41"], "split": "test"},
        {"id": "b", "iconclass": ["25G411"], "split": "test"},
        {"id": "c", "iconclass": ["25G411"], "split": "test"},
        {"id": "d", "iconclass": ["25G412"], "split": "test"},
        {"id": "e", "iconclass": ["25G41(+1)"], "split": "test"},
        {"id": "f", "iconclass": ["unresolved"], "split": "test"},
    ]


def test_relevance_index_exact_and_ideal_values():
    index = IconclassRelevanceIndex(_records(), _parents())
    assert index.exact_relevant_ids("b") == {"c"}
    ideal = index.ideal_relevance("b", 4)
    assert ideal[0] == 1.0  # c has the same exact concept
    assert ideal[1] == 0.5  # parent 25G41
    assert ideal[2] == 1 / 3  # sibling 25G412 or sibling key node
    assert ideal == sorted(ideal, reverse=True)


def test_visual_evaluation_reports_independent_iconclass_metrics():
    # a is visually closest to unresolved f, deliberately making the visual
    # ranking imperfect with respect to the hierarchy.
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.98],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.99, 0.01],
        ],
        dtype=np.float32,
    )
    table = EmbeddingTable(
        ids=("a", "b", "c", "d", "e", "f"),
        vectors=vectors,
        metadata={"fixture": True},
    )
    summary, rows = evaluate_iconclass_retrieval(
        table,
        _records(),
        _parents(),
        query_split="test",
        candidate_split="test",
        ndcg_k=10,
        persist_top_k=3,
        batch_size=2,
    )

    assert summary["n_queries"] == 6
    assert summary["n_candidates"] == 6
    assert summary["hierarchical_evaluable_queries"] == 5
    assert summary["exact_evaluable_queries"] == 2
    assert 0.0 <= summary["mean_ndcg_at_10"] <= 1.0
    assert 0.0 <= summary["mrr"] <= 1.0
    assert 0.0 <= summary["map"] <= 1.0
    assert len(rows) == 6

    query_a = next(row for row in rows if row.query_id == "a")
    assert query_a.top_results[0].item_id == "f"
    assert query_a.top_results[0].hierarchical_relevance == 0.0


def test_evaluation_can_query_outside_candidate_split():
    records = _records()
    records[0]["split"] = "validation"
    vectors = np.eye(6, dtype=np.float32)
    table = EmbeddingTable(
        ids=tuple(row["id"] for row in records),
        vectors=vectors,
        metadata={},
    )
    summary, rows = evaluate_iconclass_retrieval(
        table,
        records,
        _parents(),
        query_split="validation",
        candidate_split="test",
        persist_top_k=2,
    )
    assert summary["n_queries"] == 1
    assert rows[0].query_id == "a"


def test_stratified_summary_is_exposed_in_run_summary():
    vectors = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]],
        dtype=np.float32,
    )
    table = EmbeddingTable(
        ids=tuple(row["id"] for row in _records()), vectors=vectors, metadata={}
    )
    summary, _ = evaluate_iconclass_retrieval(table, _records(), _parents())
    assert "strata" in summary
    assert "label_frequency" in summary["strata"]
    assert "hierarchy_depth" in summary["strata"]


def test_invalid_ranking_backend_is_rejected():
    table = EmbeddingTable(
        ids=tuple(row["id"] for row in _records()),
        vectors=np.eye(6, dtype=np.float32),
        metadata={},
    )
    import pytest

    with pytest.raises(ValueError, match="backend"):
        evaluate_iconclass_retrieval(table, _records(), _parents(), backend="approximate-magic")
