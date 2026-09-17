from caypollard.splitting import (
    assign_split,
    find_checksum_leakage,
    find_group_leakage,
    split_records,
)


def test_stable_assignment():
    assert assign_split("x", seed="a") == assign_split("x", seed="a")


def test_group_split_keeps_groups_together():
    records = [
        {"id": "a", "group_id": "book-1"},
        {"id": "b", "group_id": "book-1"},
        {"id": "c", "group_id": "book-2"},
    ]
    split = split_records(records, group_key="group_id", seed="test")
    assert split[0]["split"] == split[1]["split"]
    assert find_group_leakage(split) == {}


def test_checksum_leakage_is_detected():
    records = [
        {"id": "a", "sha256": "same", "split": "train"},
        {"id": "b", "sha256": "same", "split": "test"},
    ]
    assert find_checksum_leakage(records) == {"same": {"train", "test"}}


def test_combine_group_keys_merges_signals_transitively():
    from caypollard.splitting import combine_group_keys

    records = [
        {"id": "a", "book_id": "book-1", "near_duplicate_group": None},
        {"id": "b", "book_id": "book-1", "near_duplicate_group": "dup-x"},
        # Linked to the book only through b's duplicate group.
        {"id": "c", "book_id": None, "near_duplicate_group": "dup-x"},
        {"id": "d", "book_id": None, "near_duplicate_group": None},
    ]
    grouped = {row["id"]: row["group_id"] for row in
               combine_group_keys(records, keys=["book_id", "near_duplicate_group"])}

    assert grouped["a"] == grouped["b"] == grouped["c"] == "a"
    assert grouped["d"] == "d"  # absent provenance isolates rather than pools


def test_combine_group_keys_is_stable_and_non_mutating():
    from caypollard.splitting import combine_group_keys

    records = [
        {"id": "z", "book_id": "book-1"},
        {"id": "y", "book_id": "book-1"},
        {"id": "x", "book_id": "book-2"},
    ]
    forward = combine_group_keys(records, keys=["book_id"])
    backward = combine_group_keys(list(reversed(records)), keys=["book_id"])

    assert {row["id"]: row["group_id"] for row in forward} == \
           {row["id"]: row["group_id"] for row in backward}
    assert forward[0]["group_id"] == "y"  # lexicographically smallest member
    assert "group_id" not in records[0]


def test_combine_group_keys_requires_the_item_key():
    import pytest

    from caypollard.splitting import combine_group_keys

    with pytest.raises(KeyError):
        combine_group_keys([{"book_id": "b"}], keys=["book_id"])
