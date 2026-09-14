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
