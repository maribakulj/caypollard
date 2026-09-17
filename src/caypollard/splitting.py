"""Deterministic data split helpers with leakage checks."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


def _bucket(value: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{value}".encode()).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / 2**64


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios cannot be negative")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1")


def assign_split(
    value: str,
    *,
    seed: str = "iconclass-v0.1",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> str:
    """Assign a stable train/validation/test split using a content hash."""
    _validate_ratios(ratios)
    train, validation, _test = ratios
    position = _bucket(value, seed)
    if position < train:
        return "train"
    if position < train + validation:
        return "validation"
    return "test"


def split_records(
    records: Iterable[dict[str, Any]],
    *,
    item_key: str = "id",
    group_key: str | None = None,
    seed: str = "iconclass-v0.1",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> list[dict[str, Any]]:
    """Return copied records with deterministic split assignments.

    When ``group_key`` is supplied, every non-null group is assigned as a unit.
    Null groups deliberately fall back to the item identifier so missing source
    provenance does not collapse unrelated objects into one giant group.
    """
    output: list[dict[str, Any]] = []
    for record in records:
        if item_key not in record:
            raise KeyError(f"Record is missing item key {item_key!r}")
        grouping_value = record.get(group_key) if group_key else None
        split_value = str(grouping_value) if grouping_value is not None else str(record[item_key])
        copy = dict(record)
        copy["split"] = assign_split(split_value, seed=seed, ratios=ratios)
        output.append(copy)
    return output


def combine_group_keys(
    records: Iterable[dict[str, Any]],
    *,
    keys: Sequence[str],
    item_key: str = "id",
    target_key: str = "group_id",
) -> list[dict[str, Any]]:
    """Merge several grouping signals into one transitively closed partition.

    Leakage can enter through more than one door: the same plate rescanned under
    a different filename, and two plates cut from the same book. Splitting on
    either signal alone still lets the other leak, so records sharing a non-null
    value under *any* key are merged into a single group, and membership is
    transitive — a near-duplicate of a plate joins that plate's whole book.

    Records with no non-null value under any key keep their own item identifier
    as group, so absent provenance isolates an item rather than pooling it with
    every other item whose provenance is also missing. Group identifiers are the
    lexicographically smallest member id, making them stable under reordering.
    """
    rows = [dict(record) for record in records]
    for row in rows:
        if item_key not in row:
            raise KeyError(f"Record is missing item key {item_key!r}")

    parent: dict[str, str] = {str(row[item_key]): str(row[item_key]) for row in rows}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            parent[high] = low

    for key in keys:
        seen: dict[str, str] = {}
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            marker = f"{key}\0{value}"
            if marker in seen:
                union(seen[marker], str(row[item_key]))
            else:
                seen[marker] = str(row[item_key])

    for row in rows:
        row[target_key] = find(str(row[item_key]))
    return rows


def find_group_leakage(
    records: Iterable[dict[str, Any]], *, group_key: str = "group_id"
) -> dict[str, set[str]]:
    """Return groups occurring in more than one split."""
    memberships: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group = record.get(group_key)
        split = record.get("split")
        if group is not None and split is not None:
            memberships[str(group)].add(str(split))
    return {group: splits for group, splits in memberships.items() if len(splits) > 1}


def find_checksum_leakage(records: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    """Return exact file checksums that occur across multiple splits."""
    memberships: dict[str, set[str]] = defaultdict(set)
    for record in records:
        checksum = record.get("sha256")
        split = record.get("split")
        if checksum and split:
            memberships[str(checksum)].add(str(split))
    return {checksum: splits for checksum, splits in memberships.items() if len(splits) > 1}
