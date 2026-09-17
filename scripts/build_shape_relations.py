#!/usr/bin/env python3
"""Add the grammar: which sign sits where, relative to which.

The shape vocabulary alone reaches the other medium abundantly and does not know
which neighbour is right -- 28.1% of its neighbours cross the support and its
best iconographic partner still sits at rank 196. A bag of signs cannot
distinguish a lion beneath a crown from a lion wearing one, and the
cross-depiction literature reports that models encoding spatial relations between
parts survive a change of depiction where appearance-based ones over-fit to a
single one.

Relations are a small closed set, computed from region geometry the segmenter
already recorded. They are deliberately coarse: a print and a painting of the
same subject will not agree on exact positions, but they will agree that one
element sits above another, or that one contains another.

``au-dessus`` / ``en-dessous``
    Vertical order, when the vertical gap exceeds the horizontal one.
``gauche`` / ``droite``
    Horizontal order, under the same condition reversed.
``contient`` / ``dans``
    One bounding box encloses the other.
``touche``
    Boxes overlap without enclosure.

Each feature is the ordered pair of signs plus the relation, so the vocabulary of
relations is the square of the sign vocabulary and has to be pruned: only pairs
occurring often enough to be a pattern rather than an accident are kept.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from caypollard.embeddings.store import load_embedding_table, save_embedding_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_shape_vocabulary import descriptor


def relation(a: dict, b: dict) -> str | None:
    """The coarse spatial relation from region ``a`` to region ``b``."""
    ax0, ax1 = a["centroid_x"] - a["extent_x"] / 2, a["centroid_x"] + a["extent_x"] / 2
    ay0, ay1 = a["centroid_y"] - a["extent_y"] / 2, a["centroid_y"] + a["extent_y"] / 2
    bx0, bx1 = b["centroid_x"] - b["extent_x"] / 2, b["centroid_x"] + b["extent_x"] / 2
    by0, by1 = b["centroid_y"] - b["extent_y"] / 2, b["centroid_y"] + b["extent_y"] / 2

    if ax0 <= bx0 and ay0 <= by0 and ax1 >= bx1 and ay1 >= by1:
        return "contient"
    if bx0 <= ax0 and by0 <= ay0 and bx1 >= ax1 and by1 >= ay1:
        return "dans"
    overlaps = not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)
    dx = b["centroid_x"] - a["centroid_x"]
    dy = b["centroid_y"] - a["centroid_y"]
    if overlaps:
        return "touche"
    if abs(dy) >= abs(dx):
        return "au-dessus" if dy > 0 else "en-dessous"
    return "droite" if dx > 0 else "gauche"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vocabulary_dir", help="Directory holding shape-signs.npz")
    parser.add_argument(
        "--regions", action="append", required=True, metavar="CORPUS=PATH",
    )
    parser.add_argument("--max-regions", type=int, default=10,
                        help="Largest regions considered; pairs grow quadratically")
    parser.add_argument("--min-support", type=int, default=30,
                        help="A relation feature seen fewer times than this is an accident")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    table = load_embedding_table(str(Path(args.vocabulary_dir) / "shape-signs.npz"))
    n_signs = int(table.metadata["signs"])
    rows = []
    for spec in args.regions:
        _name, _, path = spec.partition("=")
        rows.extend(
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    # The vocabulary's cluster model is not persisted, so it is refitted here with
    # the same descriptors, normalisation, seed and cluster count. Refitting on the
    # same data with a fixed seed reproduces the same partition.
    pool = np.stack([descriptor(region) for row in rows for region in row["regions"]])
    centre, scale = pool.mean(axis=0), pool.std(axis=0)
    scale[scale == 0] = 1.0
    model = MiniBatchKMeans(n_clusters=n_signs, random_state=42, n_init=8, batch_size=4096).fit(
        (pool - centre) / scale
    )

    counts: collections.Counter[tuple[int, str, int]] = collections.Counter()
    per_item: dict[str, collections.Counter] = {}
    for row in rows:
        regions = row["regions"][: args.max_regions]
        if len(regions) < 2:
            per_item[str(row["id"])] = collections.Counter()
            continue
        signs = model.predict((np.stack([descriptor(r) for r in regions]) - centre) / scale)
        local: collections.Counter[tuple[int, str, int]] = collections.Counter()
        for i, a in enumerate(regions):
            for j, b in enumerate(regions):
                if i == j:
                    continue
                name = relation(a, b)
                if name is None:
                    continue
                key = (int(signs[i]), name, int(signs[j]))
                local[key] += 1
                counts[key] += 1
        per_item[str(row["id"])] = local

    kept = {key for key, count in counts.items() if count >= args.min_support}
    index = {key: position for position, key in enumerate(sorted(kept))}
    ids = sorted(per_item)
    matrix = np.zeros((len(ids), max(len(index), 1)), dtype=np.float32)
    for position, item in enumerate(ids):
        for key, count in per_item[item].items():
            if key in index:
                matrix[position, index[key]] = float(count)

    # An item with fewer than two regions, or none of whose pairs survived the
    # support threshold, has an all-zero vector and no relation to contribute. It
    # is dropped and counted rather than normalised into a fiction.
    keep_rows = [i for i in range(len(ids)) if matrix[i].any()]
    dropped = len(ids) - len(keep_rows)
    kept_ids = tuple(ids[i] for i in keep_rows)
    matrix = matrix[keep_rows]

    output = Path(args.output)
    save_embedding_table(
        output,
        ids=kept_ids,
        vectors=matrix,
        metadata={
            "family": "symbolic",
            "method": "shape signs plus coarse spatial relations",
            "signs": n_signs,
            "relations": sorted({key[1] for key in kept}),
            "features": len(index),
            "min_support": args.min_support,
            "max_regions": args.max_regions,
            "items_dropped_without_relations": dropped,
        },
        normalize=True,
    )
    print(
        json.dumps(
            {
                "items": len(kept_ids),
                "items_dropped_without_relations": dropped,
                "candidate_features": len(counts),
                "kept_features": len(index),
                "relations": sorted({key[1] for key in kept}),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
