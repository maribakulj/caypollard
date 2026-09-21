#!/usr/bin/env python3
"""Repetition as a typed quantity, which is Neurath's own idea.

An Isotype says three by drawing the sign three times. The shape record counts
signs in a histogram, which records the same thing and types it as a magnitude
rather than as a fact: a picture with one figure and a picture with five score as
nearby points on one axis instead of as two different arrangements.

Two descriptions, kept side by side:

*the profile* -- how many of the picture's signs occur once, twice, three or
four times, five or more. This is repetition independent of which sign repeats,
and is what separates a portrait from a procession;

*the typed multiplicity* -- for each sign, whether it occurs once, a few times,
or many. Three columns, three lions and three trees are different pictures, and
this is what says so.
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

BUCKETS = ((1, 1, "une fois"), (2, 2, "deux fois"), (3, 4, "quelques"), (5, 10**6, "beaucoup"))


def bucket_of(count: int) -> int:
    for index, (low, high, _name) in enumerate(BUCKETS):
        if low <= count <= high:
            return index
    return len(BUCKETS) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vocabulary_dir")
    parser.add_argument("--regions", action="append", required=True, metavar="CORPUS=PATH")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    signs = int(
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
    model = MiniBatchKMeans(
        n_clusters=signs, random_state=args.seed, n_init=8, batch_size=4096
    ).fit((pool - centre) / scale)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        regions = row["regions"]
        if not regions:
            continue
        assigned = model.predict((np.stack([descriptor(r) for r in regions]) - centre) / scale)
        counts = collections.Counter(int(sign) for sign in assigned)

        profile = np.zeros(len(BUCKETS), dtype=np.float32)
        typed = np.zeros(signs * len(BUCKETS), dtype=np.float32)
        for sign, count in counts.items():
            index = bucket_of(count)
            profile[index] += 1.0
            typed[sign * len(BUCKETS) + index] = 1.0
        total = profile.sum()
        if total > 0:
            profile /= total
        ids.append(str(row["id"]))
        vectors.append(np.concatenate([profile, typed]))

    save_embedding_table(
        args.output,
        ids=tuple(ids),
        vectors=np.stack(vectors),
        metadata={
            "family": "repetition",
            "method": "multiplicity profile plus typed per-sign multiplicity",
            "signs": signs,
            "buckets": [name for _low, _high, name in BUCKETS],
        },
        normalize=True,
    )
    print(json.dumps({"items": len(ids), "dimensions": int(vectors[0].size)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
