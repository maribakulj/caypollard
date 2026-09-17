#!/usr/bin/env python3
"""Turn segmented regions into a closed vocabulary of named shape signs.

A coarse silhouette halves the distance to a cross-medium partner and still
cannot recognise an anchor: it has learned not to see the paper, not to see the
motif. Naming is the missing step, and a name needs a finite set to be drawn
from -- that is what makes the vocabulary closed, and a closed vocabulary is what
cannot encode a paper texture.

The signs are found rather than designed, by clustering region descriptors, but
the descriptors are chosen so that the clustering cannot rediscover the support:
area, elongation, solidity and moment invariants are all normalised for scale and
translation, and tone is reduced to darker-or-lighter than the page, which is the
only photometric fact that survives a monochrome scan. No colour, no texture, no
resolution.

Clusters are fitted on a balanced sample across corpora. Fitting on the pooled
corpus as it comes would let the larger one define the vocabulary, and the
vocabulary would then describe prints and merely approximate paintings.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from caypollard.embeddings.store import save_embedding_table

FEATURES = ("area", "elongation", "solidity")


def descriptor(region: dict) -> np.ndarray:
    """A support-invariant description of one region."""
    hu = [float(v) for v in region.get("hu", [0, 0, 0, 0])]
    # Moment invariants span many orders of magnitude; a signed log keeps their
    # sign and brings them onto a scale a Euclidean metric can use.
    hu = [float(np.sign(v) * np.log1p(abs(v) * 1e4)) for v in hu]
    return np.asarray(
        [
            np.log1p(float(region["area"]) * 100.0),
            np.log1p(float(region["elongation"])),
            float(region["solidity"]),
            1.0 if region.get("tone") == "sombre" else 0.0,
            *hu,
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions", action="append", required=True, metavar="CORPUS=PATH",
        help="Segmented regions per corpus, repeatable",
    )
    parser.add_argument("--signs", type=int, default=256)
    parser.add_argument("--sample-per-corpus", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    per_corpus: dict[str, list[dict]] = {}
    for spec in args.regions:
        name, _, path = spec.partition("=")
        per_corpus[name] = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    generator = np.random.default_rng(args.seed)
    training = []
    for rows in per_corpus.values():
        pool = [descriptor(region) for row in rows for region in row["regions"]]
        if not pool:
            continue
        stacked = np.stack(pool)
        take = min(args.sample_per_corpus, stacked.shape[0])
        index = generator.choice(stacked.shape[0], size=take, replace=False)
        training.append(stacked[index])
    matrix = np.concatenate(training, axis=0)
    centre = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    normalised = (matrix - centre) / scale

    model = MiniBatchKMeans(
        n_clusters=args.signs, random_state=args.seed, n_init=8, batch_size=4096
    ).fit(normalised)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    usage: collections.Counter[int] = collections.Counter()
    corpus_usage: dict[str, collections.Counter[int]] = {}
    for name, rows in per_corpus.items():
        corpus_usage[name] = collections.Counter()
        for row in rows:
            if not row["regions"]:
                continue
            described = (np.stack([descriptor(r) for r in row["regions"]]) - centre) / scale
            assigned = model.predict(described)
            histogram = np.zeros(args.signs, dtype=np.float32)
            for sign, region in zip(assigned, row["regions"], strict=True):
                # Weighted by area: a sign covering a quarter of the picture is
                # not the same evidence as one covering a hundredth.
                histogram[int(sign)] += float(region["area"])
                usage[int(sign)] += 1
                corpus_usage[name][int(sign)] += 1
            ids.append(str(row["id"]))
            vectors.append(histogram)

    save_embedding_table(
        output / "shape-signs.npz",
        ids=tuple(ids),
        vectors=np.stack(vectors),
        metadata={
            "family": "symbolic",
            "method": "clustered region descriptors, area-weighted histogram",
            "signs": args.signs,
            "seed": args.seed,
            "descriptor": "area, elongation, solidity, tone, 4 log moment invariants",
        },
        normalize=True,
    )

    # A sign used overwhelmingly by one corpus is a sign for a support, not a
    # motif, and its share is the diagnostic worth keeping.
    skew = []
    for sign in range(args.signs):
        counts = {name: corpus_usage[name][sign] for name in per_corpus}
        total = sum(counts.values())
        if total >= 50:
            skew.append(max(counts.values()) / total)
    report = {
        "signs": args.signs,
        "items": len(ids),
        "regions_clustered": int(matrix.shape[0]),
        "signs_used": len(usage),
        "median_corpus_skew": round(float(np.median(skew)), 4) if skew else None,
        "signs_over_90_percent_one_corpus": sum(1 for value in skew if value > 0.9),
        "per_corpus_items": {name: len(rows) for name, rows in per_corpus.items()},
    }
    (output / "vocabulary-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
