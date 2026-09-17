from __future__ import annotations

import pytest

from caypollard.datasets.near_duplicates import (
    HASH_BITS,
    assign_near_duplicate_groups,
    candidate_pairs,
    difference_hash,
    group_near_duplicates,
    hamming_distance,
    near_duplicate_report,
)

Image = pytest.importorskip("PIL.Image")


def _gradient(width: int = 64, height: int = 64, *, offset: int = 0):
    image = Image.new("L", (width, height))
    image.putdata([(x * 3 + y + offset) % 256 for y in range(height) for x in range(width)])
    return image


def test_difference_hash_is_deterministic_and_sized():
    image = _gradient()
    assert difference_hash(image) == difference_hash(image)
    assert 0 <= difference_hash(image) < 2**HASH_BITS
    assert 0 <= difference_hash(image, side=4) < 2**16


def test_difference_hash_survives_uniform_brightness_shift():
    image = _gradient()
    brighter = image.point(lambda value: min(255, value + 20))
    assert hamming_distance(difference_hash(image), difference_hash(brighter)) <= 3


def test_difference_hash_separates_unrelated_images():
    stripes = Image.new("L", (64, 64))
    stripes.putdata([255 if (x // 4) % 2 else 0 for _ in range(64) for x in range(64)])
    assert hamming_distance(difference_hash(_gradient()), difference_hash(stripes)) > 3


def test_difference_hash_rejects_degenerate_side():
    with pytest.raises(ValueError):
        difference_hash(_gradient(), side=0)


def test_candidate_pairs_never_miss_a_true_near_duplicate():
    # Pigeonhole guarantee: banding must be exact, not approximate.
    base = 0xA5A5_1234_DEAD_BEEF
    hashes = [base, base ^ 0b1011, base ^ (0b11 << 40), 0x0000_0000_0000_0000]
    threshold = 3
    pairs = candidate_pairs(hashes, threshold=threshold)
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if hamming_distance(hashes[left], hashes[right]) <= threshold:
                assert (left, right) in pairs


def test_candidate_pairs_validates_threshold():
    with pytest.raises(ValueError):
        candidate_pairs([1, 2], threshold=-1)
    with pytest.raises(ValueError):
        candidate_pairs([1, 2], threshold=HASH_BITS)


def test_group_near_duplicates_links_transitively():
    base = 0x0F0F_0F0F_0F0F_0F0F
    hashes = [base, base ^ 0b11, base ^ 0b1111, ~base & (2**HASH_BITS - 1)]
    groups = group_near_duplicates(hashes, threshold=2)
    assert [0, 1, 2] in groups  # chained through the middle element
    assert [3] in groups


def test_group_near_duplicates_is_order_independent():
    hashes = [0x00FF, 0x00FE, 0xFF00, 0xFF01]
    forward = group_near_duplicates(hashes, threshold=1)
    reversed_groups = group_near_duplicates(list(reversed(hashes)), threshold=1)
    sizes = sorted(len(group) for group in forward)
    assert sizes == sorted(len(group) for group in reversed_groups)


def test_assign_groups_uses_stable_identifier_and_skips_missing_hashes():
    records = [
        {"id": "b", "phash": 0x0F0F},
        {"id": "a", "phash": 0x0F0E},
        {"id": "c", "phash": 0xF0F0},
        {"id": "d", "phash": None},
    ]
    assigned = {row["id"]: row["near_duplicate_group"] for row in
                assign_near_duplicate_groups(records, threshold=1)}
    assert assigned["a"] == assigned["b"] == "a"
    assert assigned["c"] == "c"
    assert assigned["d"] is None


def test_assign_groups_does_not_mutate_input():
    records = [{"id": "a", "phash": 1}]
    assign_near_duplicate_groups(records)
    assert "near_duplicate_group" not in records[0]


def test_report_counts_multi_image_groups():
    records = [
        {"id": "a", "near_duplicate_group": "a"},
        {"id": "b", "near_duplicate_group": "a"},
        {"id": "c", "near_duplicate_group": "c"},
        {"id": "d", "near_duplicate_group": None},
    ]
    report = near_duplicate_report(records)
    assert report["n_hashed_images"] == 3
    assert report["n_images_without_hash"] == 1
    assert report["n_groups"] == 2
    assert report["n_multi_image_groups"] == 1
    assert report["n_images_in_multi_image_groups"] == 2
    assert report["largest_group_size"] == 2
    assert report["multi_image_group_size_histogram"] == {"2": 1}


def test_fine_hash_rejects_a_coarse_layout_collision():
    # Two images can share a page layout closely enough to collide at 64 bits
    # while differing everywhere at 256 -- the failure that merged 211 unrelated
    # paintings on the real corpus.
    coarse = [0x0F0F_0F0F_0F0F_0F0F, 0x0F0F_0F0F_0F0F_0F0F]
    agreeing = [0, 0]
    disagreeing = [0, (1 << 60) - 1]

    assert group_near_duplicates(coarse, fine_hashes=agreeing) == [[0, 1]]
    assert group_near_duplicates(coarse, fine_hashes=disagreeing) == [[0], [1]]


def test_fine_hash_keeps_pairs_inside_the_calibrated_threshold():
    from caypollard.datasets.near_duplicates import DEFAULT_FINE_THRESHOLD

    coarse = [0x00FF, 0x00FF]
    just_inside = [0, (1 << DEFAULT_FINE_THRESHOLD) - 1]
    just_outside = [0, (1 << (DEFAULT_FINE_THRESHOLD + 1)) - 1]

    assert group_near_duplicates(coarse, fine_hashes=just_inside) == [[0, 1]]
    assert group_near_duplicates(coarse, fine_hashes=just_outside) == [[0], [1]]


def test_fine_hashes_must_match_the_coarse_count():
    with pytest.raises(ValueError, match="one entry per coarse hash"):
        group_near_duplicates([1, 2], fine_hashes=[1])


def test_assign_uses_fine_hashes_only_when_every_record_has_one():
    records = [
        {"id": "a", "phash": 0x0F0F, "phash_fine": 0},
        {"id": "b", "phash": 0x0F0F, "phash_fine": (1 << 200) - 1},
    ]
    split = assign_near_duplicate_groups(records)
    assert split[0]["near_duplicate_group"] != split[1]["near_duplicate_group"]

    # Missing fine hashes fall back to coarse-only linkage rather than erroring.
    partial = [{"id": "a", "phash": 0x0F0F}, {"id": "b", "phash": 0x0F0F}]
    merged = assign_near_duplicate_groups(partial)
    assert merged[0]["near_duplicate_group"] == merged[1]["near_duplicate_group"]
