import numpy as np

from caypollard.benchmarks.hard_pairs import (
    HardPairThresholds,
    calibrate_visual_thresholds,
    classify_pair,
    mine_hard_pairs,
)
from caypollard.embeddings.store import EmbeddingTable

PARENTS = {
    "A": set(),
    "A1": {"A"},
    "A2": {"A"},
    "B": set(),
    "B1": {"B"},
    "B2": {"B"},
}
RECORDS = [
    {"id": "a1", "iconclass": ["A1"]},
    {"id": "a2", "iconclass": ["A1"]},
    {"id": "a3", "iconclass": ["A2"]},
    {"id": "b1", "iconclass": ["B1"]},
    {"id": "b2", "iconclass": ["B1"]},
    {"id": "b3", "iconclass": ["B2"]},
]


def _table():
    # a1 is visually close to b1 despite semantic disconnection (hard negative),
    # while a2 is semantically identical to a1 but visually opposite (hard positive).
    return EmbeddingTable(
        ids=("a1", "a2", "a3", "b1", "b2", "b3"),
        vectors=np.asarray(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [-0.9, 0.1],
                [0.99, 0.01],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        metadata={"fixture": True},
    )


def test_pair_quadrant_classification():
    thresholds = HardPairThresholds(visual_close_min=0.8, visual_distant_max=0.0)
    assert classify_pair(0.9, 1.0, thresholds) == "easy_positive"
    assert classify_pair(0.9, 0.0, thresholds) == "hard_negative"
    assert classify_pair(-0.5, 1.0, thresholds) == "hard_positive"
    assert classify_pair(-0.5, 0.0, thresholds) == "easy_negative"
    assert classify_pair(0.4, 0.4, thresholds) is None


def test_visual_threshold_calibration_uses_only_requested_ids():
    thresholds, metadata = calibrate_visual_thresholds(
        _table(), ["a1", "a2", "a3", "b1", "b2"], sample_pairs=100, seed=5
    )
    assert thresholds.visual_close_min > thresholds.visual_distant_max
    assert metadata["validation_id_count"] == 5
    assert metadata["sample_pairs_used"] == 10


def test_hard_pair_mining_finds_disagreement_cases():
    thresholds = HardPairThresholds(
        visual_close_min=0.8,
        visual_distant_max=0.0,
        semantic_close_min=0.5,
        semantic_distant_max=0.0,
    )
    pairs, metadata = mine_hard_pairs(
        _table(),
        RECORDS,
        PARENTS,
        thresholds,
        query_ids=["a1", "a2", "b1"],
        visual_top_k=3,
        random_distant_per_query=4,
        max_per_class=10,
        seed=3,
    )
    classes = {pair.pair_class for pair in pairs}
    assert "hard_negative" in classes
    assert "hard_positive" in classes
    assert metadata["selected_counts"]["hard_negative"] >= 1
    assert metadata["selected_counts"]["hard_positive"] >= 1


def _stratified_records():
    return [
        {"id": "a", "iconclass": ["25F23(BEAR)"], "collection": "X"},
        {"id": "b", "iconclass": ["25F23(BEAR)"], "collection": "X"},
        {"id": "c", "iconclass": ["25F23(BEAR)"], "collection": "Y"},
        {"id": "d", "iconclass": ["99Z"], "collection": "Y"},
    ]


def test_stratum_key_rejects_cross_stratum_pairs():
    import numpy as np

    from caypollard.benchmarks.hard_pairs import HardPairThresholds, mine_hard_pairs
    from caypollard.embeddings.store import EmbeddingTable

    records = _stratified_records()
    ids = [r["id"] for r in records]
    # a/b similar; c/d far from everything, so every surviving pair is decided by
    # the stratum rather than by the vectors.
    vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    table = EmbeddingTable(ids=ids, vectors=vectors, metadata={})
    parents = {"25F23": set(), "99Z": set(), "25F23(BEAR)": {"25F23"}}
    thresholds = HardPairThresholds(
        visual_close_min=0.9, visual_distant_max=0.5,
        semantic_close_min=0.5, semantic_distant_max=0.2,
    )

    pairs, mining = mine_hard_pairs(
        table, records, parents, thresholds, stratum_key="collection", max_per_class=50
    )
    assert mining["stratum_key"] == "collection"
    for pair in pairs:
        left = next(r for r in records if r["id"] == pair.query_id)
        right = next(r for r in records if r["id"] == pair.candidate_id)
        assert left["collection"] == right["collection"]


def test_without_a_stratum_key_cross_stratum_pairs_survive():
    import numpy as np

    from caypollard.benchmarks.hard_pairs import HardPairThresholds, mine_hard_pairs
    from caypollard.embeddings.store import EmbeddingTable

    records = _stratified_records()
    table = EmbeddingTable(
        ids=[r["id"] for r in records],
        vectors=np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
        metadata={},
    )
    parents = {"25F23": set(), "99Z": set(), "25F23(BEAR)": {"25F23"}}
    thresholds = HardPairThresholds(
        visual_close_min=0.9, visual_distant_max=0.5,
        semantic_close_min=0.5, semantic_distant_max=0.2,
    )
    pairs, _ = mine_hard_pairs(table, records, parents, thresholds, max_per_class=50)
    assert any(
        next(r for r in records if r["id"] == p.query_id)["collection"]
        != next(r for r in records if r["id"] == p.candidate_id)["collection"]
        for p in pairs
    )


def test_records_missing_the_stratum_field_are_excluded():
    import numpy as np

    from caypollard.benchmarks.hard_pairs import HardPairThresholds, mine_hard_pairs
    from caypollard.embeddings.store import EmbeddingTable

    records = [
        {"id": "a", "iconclass": ["25F23(BEAR)"], "collection": None},
        {"id": "b", "iconclass": ["25F23(BEAR)"], "collection": None},
    ]
    table = EmbeddingTable(
        ids=["a", "b"], vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), metadata={}
    )
    pairs, _ = mine_hard_pairs(
        table, records, {"25F23": set(), "25F23(BEAR)": {"25F23"}},
        HardPairThresholds(0.9, 0.5, 0.5, 0.2), stratum_key="collection",
    )
    # Two nulls are not a shared stratum; provenance must be positive, not absent.
    assert pairs == []
