"""Deterministic data split helpers with leakage checks."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable


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
