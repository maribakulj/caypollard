"""Model-independent near-duplicate image detection.

Exact duplicates are already caught by SHA-256 grouping. Near duplicates are
not: reprints, rescans, crops, and JPEG requantisations of the same plate
produce different bytes while remaining the same object for retrieval purposes.
Leaving them split across partitions lets a model score by recognising a copy
of the query rather than by generalising.

Detection deliberately uses a perceptual hash rather than the visual encoder
under evaluation. Removing duplicates with DINOv2 embeddings and then reporting
DINOv2 retrieval on the cleaned benchmark would entangle the cleaning step with
the model being tested. A difference hash depends only on the image.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

HASH_SIDE = 8
"""Rows in the downscaled grid; a side of ``n`` yields an ``n * n`` bit hash."""

HASH_BITS = HASH_SIDE * HASH_SIDE

FINE_SIDE = 16
"""Grid side of the confirmation hash."""

FINE_BITS = FINE_SIDE * FINE_SIDE

DEFAULT_FINE_THRESHOLD = 28
"""Maximum differing bits of the 256-bit confirmation hash.

Calibrated by inspection against the full Iconclass AI Test Set rather than
chosen for roundness. Byte-identical images score 0. Manual review of sampled
pairs found genuine near duplicates -- the same woodcut in two states, an ink
drawing and its red-chalk version, one plate rescanned at a different tone --
still confirming at distances up to 25, while the first false merges appear from
35: two medal portraits of different emperors sharing a page layout, and two
unrelated paintings photographed on the same mount. 28 sits inside that gap.
"""


def difference_hash(image: Any, *, side: int = HASH_SIDE) -> int:
    """Return a ``side * side``-bit difference hash of a PIL image.

    The image is converted to greyscale and resized to ``(side + 1, side)``,
    then each bit records whether a pixel is brighter than its right neighbour.
    Comparing neighbours rather than absolute levels makes the hash invariant to
    uniform brightness and contrast shifts, which is what separates a rescan of
    a plate from a genuinely different plate.
    """
    if side < 1:
        raise ValueError("side must be positive")

    from PIL import Image

    resized = image.convert("L").resize((side + 1, side), Image.Resampling.LANCZOS)
    # `getdata` is deprecated from Pillow 14 but `get_flattened_data` only exists
    # on recent releases, and the declared floor is Pillow 10.4.
    reader = getattr(resized, "get_flattened_data", None) or resized.getdata
    pixels = list(reader())

    value = 0
    for row in range(side):
        offset = row * (side + 1)
        for column in range(side):
            value <<= 1
            if pixels[offset + column] > pixels[offset + column + 1]:
                value |= 1
    return value


def hash_image_file(path: str | Path, *, side: int = HASH_SIDE) -> int:
    """Open one image file and return its difference hash."""
    from PIL import Image

    with Image.open(path) as image:
        return difference_hash(image, side=side)


def hamming_distance(left: int, right: int) -> int:
    """Return the number of differing bits between two hashes."""
    return (left ^ right).bit_count()


def _band_keys(value: int, *, bands: int, bits: int) -> Iterator[tuple[int, int]]:
    """Split a hash into ``bands`` contiguous chunks, yielding ``(index, chunk)``."""
    width = bits // bands
    remainder = bits % bands
    offset = 0
    for index in range(bands):
        size = width + (1 if index < remainder else 0)
        offset += size
        yield index, (value >> (bits - offset)) & ((1 << size) - 1)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, node: int) -> int:
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Attach to the smaller root so group representatives stay stable
            # under reordering of the input.
            low, high = sorted((left_root, right_root))
            self._parent[high] = low


def candidate_pairs(
    hashes: Sequence[int], *, threshold: int, bits: int = HASH_BITS
) -> set[tuple[int, int]]:
    """Return index pairs that may lie within ``threshold`` bits of each other.

    Comparing every pair is quadratic and infeasible at corpus scale. Splitting
    each hash into ``threshold + 1`` bands makes the shortcut exact rather than
    approximate: by the pigeonhole principle, two hashes differing in at most
    ``threshold`` bits must agree exactly on at least one band, so no true near
    duplicate can be missed by only comparing within-band collisions.
    """
    if threshold < 0:
        raise ValueError("threshold cannot be negative")
    bands = threshold + 1
    if bands > bits:
        raise ValueError(f"threshold {threshold} is too large for a {bits}-bit hash")

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, value in enumerate(hashes):
        for key in _band_keys(value, bands=bands, bits=bits):
            buckets[key].append(index)

    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                pairs.add((left, right))
    return pairs


def group_near_duplicates(
    hashes: Sequence[int],
    *,
    threshold: int = 3,
    bits: int = HASH_BITS,
    fine_hashes: Sequence[int] | None = None,
    fine_threshold: int = DEFAULT_FINE_THRESHOLD,
) -> list[list[int]]:
    """Cluster hash indices into transitively linked near-duplicate groups.

    Returns one sorted list of indices per group, groups ordered by their first
    member, singletons included. Linkage is transitive: an image within
    ``threshold`` of two others joins both into a single group even when those
    two exceed the threshold between themselves. That is deliberate — a chain of
    successive rescans is one object, and splitting the chain across partitions
    is exactly the leakage this guards against.

    A 64-bit hash alone is too coarse on this corpus: it reads *layout* as much
    as content, so it merges pages of printed text sharing a two-column frame,
    and photographic reproductions of different paintings shot on one mount.
    Passing ``fine_hashes`` adds a confirmation stage — the coarse hash proposes
    candidates, and a pair is linked only when the finer hash agrees. Without it
    the largest group on the full test set reached 211 unrelated images.
    """
    if fine_hashes is not None and len(fine_hashes) != len(hashes):
        raise ValueError("fine_hashes must have one entry per coarse hash")

    components = _UnionFind(len(hashes))
    for left, right in candidate_pairs(hashes, threshold=threshold, bits=bits):
        if hamming_distance(hashes[left], hashes[right]) > threshold:
            continue
        if fine_hashes is not None and (
            hamming_distance(fine_hashes[left], fine_hashes[right]) > fine_threshold
        ):
            continue
        components.union(left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(hashes)):
        grouped[components.find(index)].append(index)
    return [sorted(members) for _, members in sorted(grouped.items())]


def assign_near_duplicate_groups(
    records: Iterable[dict[str, Any]],
    *,
    hash_key: str = "phash",
    fine_hash_key: str = "phash_fine",
    group_key: str = "near_duplicate_group",
    threshold: int = 3,
    bits: int = HASH_BITS,
    fine_threshold: int = DEFAULT_FINE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return copies of ``records`` carrying a deterministic near-duplicate group.

    Records missing ``hash_key`` keep a null group rather than being pooled into
    one bucket of unknowns, mirroring how ``build_manifest`` refuses to fabricate
    absent provenance. Group identifiers derive from the lexicographically first
    record id in each group, so they survive reordering of the input.
    """
    rows = [dict(record) for record in records]
    hashable = [row for row in rows if row.get(hash_key) is not None]

    hashes = [int(row[hash_key]) for row in hashable]
    fine = (
        [int(row[fine_hash_key]) for row in hashable]
        if all(row.get(fine_hash_key) is not None for row in hashable)
        else None
    )
    for members in group_near_duplicates(
        hashes, threshold=threshold, bits=bits, fine_hashes=fine, fine_threshold=fine_threshold
    ):
        group_rows = [hashable[index] for index in members]
        identifier = min(str(row.get("id", "")) for row in group_rows)
        for row in group_rows:
            row[group_key] = identifier

    for row in rows:
        row.setdefault(group_key, None)
    return rows


def near_duplicate_report(
    records: Iterable[dict[str, Any]], *, group_key: str = "near_duplicate_group"
) -> dict[str, Any]:
    """Summarise near-duplicate structure for the frozen corpus audit."""
    sizes: dict[str, int] = defaultdict(int)
    unhashed = 0
    for record in records:
        group = record.get(group_key)
        if group is None:
            unhashed += 1
            continue
        sizes[str(group)] += 1

    multi = sorted((size for size in sizes.values() if size > 1), reverse=True)
    return {
        "n_hashed_images": sum(sizes.values()),
        "n_images_without_hash": unhashed,
        "n_groups": len(sizes),
        "n_multi_image_groups": len(multi),
        "n_images_in_multi_image_groups": sum(multi),
        "largest_group_size": multi[0] if multi else 0,
        "multi_image_group_size_histogram": {
            str(size): multi.count(size) for size in sorted(set(multi))
        },
    }
