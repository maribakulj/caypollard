#!/usr/bin/env python3
"""Group touching regions into composite signs, because an object is not a patch.

Half the regions the segmenter finds cover less than 2% of the picture. They are
fragments, and the coherence test says so from the other side: a sign's
associated notations share a sixteenth of their ancestry, which is above chance
and nowhere near a name. An anchor is not one patch. It is a shank, a stock and
two flukes in an arrangement, and the arrangement is what recurs.

So regions are grouped before they are named. Two regions join the same group
when their bounding boxes touch or one encloses the other, and the relation is
closed transitively, which gives the connected clumps of ink or paint a picture
is actually made of. Each group is then described twice over:

* by **what it contains** -- the multiset of its members' atomic signs, which is
  what makes two groups comparable across media;
* by **its own outline** -- total area, the elongation and fill of its bounding
  box, and how many parts it has, which is what distinguishes one arrangement of
  the same parts from another.

Groups are clustered into composite signs on that joint description. The question
the whole exercise asks is then testable with the measure already built: are
composite signs more iconographically coherent than atomic ones?
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from caypollard.embeddings.store import save_embedding_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_shape_vocabulary import descriptor


def box(region: dict) -> tuple[float, float, float, float]:
    return (
        region["centroid_x"] - region["extent_x"] / 2,
        region["centroid_y"] - region["extent_y"] / 2,
        region["centroid_x"] + region["extent_x"] / 2,
        region["centroid_y"] + region["extent_y"] / 2,
    )


def touching(a: dict, b: dict, *, margin: float, ratio: float) -> bool:
    """Near enough, and comparable enough in size, to be parts of one thing.

    Absolute box overlap is useless here: the dark and light regions of a picture
    interpenetrate by construction, so almost every pair of boxes overlaps and a
    transitive closure swallows the whole plate. Two tests fix it. The gap must be
    small *relative to the parts' own size*, so a speck beside a background mass
    does not join it; and the parts must be within an order of magnitude of each
    other in area, so a background region cannot act as a hub joining everything
    that happens to lie on it.
    """
    left, right = float(a["area"]), float(b["area"])
    if max(left, right) / max(min(left, right), 1e-9) > ratio:
        return False
    ax0, ay0, ax1, ay1 = box(a)
    bx0, by0, bx1, by1 = box(b)
    gap_x = max(0.0, max(bx0 - ax1, ax0 - bx1))
    gap_y = max(0.0, max(by0 - ay1, ay0 - by1))
    gap = (gap_x**2 + gap_y**2) ** 0.5
    diagonal = min(
        (a["extent_x"] ** 2 + a["extent_y"] ** 2) ** 0.5,
        (b["extent_x"] ** 2 + b["extent_y"] ** 2) ** 0.5,
    )
    return gap <= margin * max(diagonal, 1e-6)


def groups_of(regions: list[dict], *, margin: float, ratio: float) -> list[list[int]]:
    """Connected clumps of regions, by transitive closure of touching."""
    parent = list(range(len(regions)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if touching(regions[i], regions[j], margin=margin, ratio=ratio):
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    clumps: dict[int, list[int]] = collections.defaultdict(list)
    for index in range(len(regions)):
        clumps[find(index)].append(index)
    return list(clumps.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vocabulary_dir", help="Atomic vocabulary, for the member signs")
    parser.add_argument("--regions", action="append", required=True, metavar="CORPUS=PATH")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.35,
        help="Largest gap between two parts, as a fraction of the smaller part's diagonal",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=12.0,
        help="Largest area ratio between two parts of one group",
    )
    parser.add_argument("--min-parts", type=int, default=2)
    parser.add_argument("--composites", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    atoms = int(
        json.loads(
            (Path(args.vocabulary_dir) / "vocabulary-report.json").read_text(encoding="utf-8")
        )["signs"]
    )

    rows = []
    for spec in args.regions:
        _name, _, path = spec.partition("=")
        rows.extend(
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    pool = np.stack([descriptor(region) for row in rows for region in row["regions"]])
    centre, scale = pool.mean(axis=0), pool.std(axis=0)
    scale[scale == 0] = 1.0
    atom_model = MiniBatchKMeans(
        n_clusters=atoms, random_state=args.seed, n_init=8, batch_size=4096
    ).fit((pool - centre) / scale)

    described: list[np.ndarray] = []
    owners: list[str] = []
    sizes: list[int] = []
    for row in rows:
        regions = row["regions"]
        if not regions:
            continue
        signs = atom_model.predict((np.stack([descriptor(r) for r in regions]) - centre) / scale)
        for members in groups_of(regions, margin=args.margin, ratio=args.ratio):
            if len(members) < args.min_parts:
                continue
            xs0, ys0, xs1, ys1 = zip(*(box(regions[i]) for i in members), strict=True)
            width = max(xs1) - min(xs0)
            height = max(ys1) - min(ys0)
            area = float(sum(regions[i]["area"] for i in members))
            histogram = np.zeros(atoms, dtype=np.float32)
            for index in members:
                histogram[int(signs[index])] += float(regions[index]["area"])
            norm = np.linalg.norm(histogram)
            if norm > 0:
                histogram /= norm
            outline = np.asarray(
                [
                    np.log1p(area * 100.0),
                    np.log1p(max(width, height) / max(min(width, height), 1e-6)),
                    area / max(width * height, 1e-6),
                    np.log1p(len(members)),
                    float(np.median([regions[i]["centroid_y"] for i in members])),
                ],
                dtype=np.float32,
            )
            described.append(np.concatenate([histogram, outline]))
            owners.append(str(row["id"]))
            sizes.append(len(members))

    if not described:
        raise SystemExit("no group met the minimum part count")
    matrix = np.stack(described)
    # The outline block is five numbers against `atoms` histogram entries, so it
    # is scaled up to contribute rather than be averaged away.
    matrix[:, atoms:] *= np.sqrt(atoms / 5.0)
    group_model = MiniBatchKMeans(
        n_clusters=args.composites, random_state=args.seed, n_init=8, batch_size=4096
    ).fit(matrix)
    assigned = group_model.predict(matrix)

    per_item: dict[str, np.ndarray] = {}
    for composite, owner in zip(assigned, owners, strict=True):
        vector = per_item.setdefault(owner, np.zeros(args.composites, dtype=np.float32))
        vector[int(composite)] += 1.0
    ids = sorted(item for item, vector in per_item.items() if vector.any())

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_embedding_table(
        output / "composite-signs.npz",
        ids=tuple(ids),
        vectors=np.stack([per_item[item] for item in ids]),
        metadata={
            "family": "symbolic",
            "method": "clumps of touching regions, described by member signs and own outline",
            "atoms": atoms,
            "composites": args.composites,
            "margin": args.margin,
            "ratio": args.ratio,
            "min_parts": args.min_parts,
        },
        normalize=True,
    )
    report = {
        "groups": len(described),
        "items_with_a_group": len(ids),
        "items_total": len({str(row["id"]) for row in rows}),
        "median_parts_per_group": int(np.median(sizes)),
        "mean_parts_per_group": round(float(np.mean(sizes)), 2),
        "largest_group": int(max(sizes)),
        "composites": args.composites,
        "atoms": atoms,
    }
    (output / "group-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
