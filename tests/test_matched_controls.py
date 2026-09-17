"""Tests for visually matched control selection and paired AUC."""

from __future__ import annotations

import numpy as np
import pytest

from caypollard.benchmarks.matched_controls import (
    bootstrap_paired_auc,
    match_controls,
    paired_auc,
)

RECORDS = {
    "q": {"id": "q", "collection": "A", "iconclass": ["11H"]},
    "pos": {"id": "pos", "collection": "A", "iconclass": ["11H"]},
    "good_control": {"id": "good_control", "collection": "A", "iconclass": ["25F"]},
    "too_similar": {"id": "too_similar", "collection": "A", "iconclass": ["31A"]},
    "shares_label": {"id": "shares_label", "collection": "A", "iconclass": ["11H"]},
    "other_stratum": {"id": "other_stratum", "collection": "B", "iconclass": ["44G"]},
}
SIMILARITIES = {
    "pos": 0.40,
    "good_control": 0.41,
    "too_similar": 0.90,
    "shares_label": 0.40,
    "other_stratum": 0.40,
}
PAIRS = [{"query_id": "q", "candidate_id": "pos", "pair_class": "hard_positive"}]


def similarity(query: str, candidates):
    assert query == "q"
    return np.asarray([SIMILARITIES[item] for item in candidates], dtype=float)


def test_control_matches_visual_similarity_and_shares_no_label() -> None:
    matched, skipped = match_controls(
        PAIRS, RECORDS, similarity, pool=list(SIMILARITIES), tolerance=0.02
    )
    assert len(matched) == 1
    assert skipped == {"unknown_item": 0, "empty_stratum": 0, "no_eligible_control": 0}
    triple = matched[0]
    # Only good_control is in the stratum, close enough, and semantically unrelated:
    # too_similar is 0.5 away, shares_label shares 11H, other_stratum is collection B.
    assert triple.control_id == "good_control"
    assert abs(triple.positive_similarity - triple.control_similarity) <= 0.02


def test_a_positive_with_no_eligible_control_is_reported_not_dropped_silently() -> None:
    pool = ["pos", "too_similar", "shares_label"]
    matched, skipped = match_controls(PAIRS, RECORDS, similarity, pool=pool, tolerance=0.02)
    assert matched == []
    assert skipped["no_eligible_control"] == 1


def test_a_positive_outside_the_pool_is_skipped() -> None:
    matched, skipped = match_controls(
        PAIRS, RECORDS, similarity, pool=["good_control"], tolerance=0.02
    )
    assert matched == []
    assert skipped["unknown_item"] == 1


def test_matching_is_deterministic_for_a_fixed_seed() -> None:
    kwargs = {"pool": list(SIMILARITIES), "tolerance": 0.5}
    first, _ = match_controls(PAIRS, RECORDS, similarity, seed=7, **kwargs)
    second, _ = match_controls(PAIRS, RECORDS, similarity, seed=7, **kwargs)
    assert [t.control_id for t in first] == [t.control_id for t in second]


def test_paired_auc_counts_ties_as_half() -> None:
    assert paired_auc([1.0, 0.0, 0.5], [0.0, 1.0, 0.5]) == pytest.approx(0.5)
    assert paired_auc([1.0, 1.0], [0.0, 0.0]) == 1.0
    assert paired_auc([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_paired_auc_rejects_mismatched_or_empty_input() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        paired_auc([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least one"):
        paired_auc([], [])


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    generator = np.random.default_rng(0)
    positive = generator.normal(0.6, 0.2, size=200)
    control = generator.normal(0.4, 0.2, size=200)
    result = bootstrap_paired_auc(positive, control, n_resamples=300, seed=1)
    assert result["ci_low"] <= result["auc"] <= result["ci_high"]
    assert result["n_pairs"] == 200
    assert result["auc"] > 0.5


def test_tolerance_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        match_controls(PAIRS, RECORDS, similarity, pool=list(SIMILARITIES), tolerance=0.0)


def test_controls_are_balanced_around_the_target_similarity() -> None:
    # Hard positives sit in the low tail, so a symmetric caliper offers many more
    # candidates above the target than below it. Drawing uniformly would make the
    # control reliably more similar to the query than the positive is, and the
    # visual AUC would land near 0.40 where the design requires 0.50.
    records = {"q": {"id": "q", "collection": "A", "iconclass": ["11H"]}}
    similarities = {"pos": 0.50}
    for index in range(3):
        records[f"low{index}"] = {"id": f"low{index}", "collection": "A", "iconclass": ["99Z"]}
        similarities[f"low{index}"] = 0.49
    for index in range(30):
        records[f"high{index}"] = {"id": f"high{index}", "collection": "A", "iconclass": ["99Z"]}
        similarities[f"high{index}"] = 0.51
    records["pos"] = {"id": "pos", "collection": "A", "iconclass": ["11H"]}

    def sim(query: str, candidates):
        return np.asarray([similarities[item] for item in candidates], dtype=float)

    pairs = [
        {"query_id": "q", "candidate_id": "pos", "pair_class": "hard_positive"} for _ in range(20)
    ]
    matched, _ = match_controls(pairs, records, sim, pool=list(similarities), tolerance=0.02)
    differences = [t.control_similarity - t.positive_similarity for t in matched]
    below = sum(1 for d in differences if d <= 0)
    assert len(matched) == 20
    assert abs(below - len(matched) / 2) <= 1
