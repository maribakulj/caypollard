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
