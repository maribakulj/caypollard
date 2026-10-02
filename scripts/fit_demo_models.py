#!/usr/bin/env python3
"""Refit and save the clustering models the frozen tables were built with.

The shape vocabularies and the composite signs were fitted inside the scripts
that produced their tables, and the fitted centres were never written down. To
describe a *new* picture in the same vocabulary, the centres are needed: an
atom is the nearest centre to a region's standardised descriptor.

Every fit here repeats the exact call that produced a frozen table -- same
regions, same order, same sampling, same seed -- and is then verified against
that table by recomputing rows from the regions on disk. The verification is
written next to the models, so a mismatch is a fact and not a suspicion.

Outputs, under data/derived/demo/:

  atoms-v2-256.npz   shape signs of v2-256 (formes muettes): centre, scale, centres
  atoms-pool.npz     the atoms repetition-v3 and v3-groups were built on
  atoms-v4-256.npz   the balanced vocabulary mix-tout uses for its formes channel
  composites-v3.npz  the 128 composite signs of v3-groups
  atoms-64.npz       the 64-sign vocabulary of relations-wide64
  relations-64.npz   the 9 518 (sign, relation, sign) keys that table keeps
  fit-report.json    what was fitted, from what, and how closely it reproduces
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_repetition_channel import BUCKETS, bucket_of  # noqa: E402
from build_shape_groups import box, groups_of  # noqa: E402
from build_shape_relations import relation  # noqa: E402
from build_shape_vocabulary import descriptor  # noqa: E402

from caypollard.embeddings.store import load_embedding_table  # noqa: E402

SHAPES = ROOT / "data/derived/shapes"


def rows_of(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def descriptors_of(rows: list[dict]) -> np.ndarray:
    return np.stack([descriptor(region) for row in rows for region in row["regions"]])


def fit_atoms(matrix: np.ndarray, *, signs: int, seed: int) -> dict[str, np.ndarray]:
    centre = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    model = MiniBatchKMeans(n_clusters=signs, random_state=seed, n_init=8, batch_size=4096)
    model.fit((matrix - centre) / scale)
    return {"centre": centre, "scale": scale, "centres": model.cluster_centers_.astype(np.float32)}


def assign(described: np.ndarray, atoms: dict[str, np.ndarray]) -> np.ndarray:
    """Nearest centre in standardised space: what MiniBatchKMeans.predict computes."""
    z = (described - atoms["centre"]) / atoms["scale"]
    d = ((z[:, None, :] - atoms["centres"][None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def histogram(regions: list[dict], atoms: dict[str, np.ndarray]) -> np.ndarray:
    signs = assign(np.stack([descriptor(r) for r in regions]), atoms)
    h = np.zeros(atoms["centres"].shape[0], dtype=np.float32)
    for sign, region in zip(signs, regions, strict=True):
        h[int(sign)] += float(region["area"])
    return l2(h)


def repetition(regions: list[dict], atoms: dict[str, np.ndarray]) -> np.ndarray:
    signs = assign(np.stack([descriptor(r) for r in regions]), atoms)
    counts = collections.Counter(int(s) for s in signs)
    n = atoms["centres"].shape[0]
    profile = np.zeros(len(BUCKETS), dtype=np.float32)
    typed = np.zeros(n * len(BUCKETS), dtype=np.float32)
    for sign, count in counts.items():
        b = bucket_of(count)
        profile[b] += 1.0
        typed[sign * len(BUCKETS) + b] = 1.0
    if profile.sum() > 0:
        profile /= profile.sum()
    return l2(np.concatenate([profile, typed]))


def group_descriptions(
    regions: list[dict],
    atoms: dict[str, np.ndarray],
    *,
    margin: float,
    ratio: float,
    min_parts: int,
) -> tuple[np.ndarray, list[list[int]]]:
    n = atoms["centres"].shape[0]
    signs = assign(np.stack([descriptor(r) for r in regions]), atoms)
    described, kept = [], []
    for members in groups_of(regions, margin=margin, ratio=ratio):
        if len(members) < min_parts:
            continue
        xs0, ys0, xs1, ys1 = zip(*(box(regions[i]) for i in members), strict=True)
        width, height = max(xs1) - min(xs0), max(ys1) - min(ys0)
        area = float(sum(regions[i]["area"] for i in members))
        h = np.zeros(n, dtype=np.float32)
        for i in members:
            h[int(signs[i])] += float(regions[i]["area"])
        h = l2(h)
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
        described.append(np.concatenate([h, outline]))
        kept.append(members)
    if not described:
        return np.zeros((0, n + 5), dtype=np.float32), []
    # The outline block is scaled after stacking, in place and in float32, exactly as the
    # script does it: doing the multiplication per row changes the low bits, and a k-means
    # fitted on those bits lands on different centres.
    matrix = np.stack(described)
    matrix[:, n:] *= np.sqrt(n / 5.0)
    return matrix, kept


def composite_vector(described: np.ndarray, centres: np.ndarray) -> np.ndarray:
    v = np.zeros(centres.shape[0], dtype=np.float32)
    if described.shape[0]:
        d = ((described[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        for c in d.argmin(axis=1):
            v[int(c)] += 1.0
    return l2(v)


def verify(
    name: str, table_path: Path, rows: list[dict], compute, *, sample: int, seed: int
) -> dict:
    table = load_embedding_table(table_path)
    row_of = {i: r for r in rows for i in [str(r["id"])] if r["regions"]}
    ids = [i for i in table.ids if i in row_of]
    rng = np.random.default_rng(seed)
    picked = [ids[i] for i in rng.choice(len(ids), size=min(sample, len(ids)), replace=False)]
    index = {i: n for n, i in enumerate(table.ids)}
    worst = 0.0
    for item in picked:
        got = compute(row_of[item]["regions"])
        worst = max(worst, float(np.abs(got - table.vectors[index[item]]).max()))
    report = {
        "table": str(table_path.relative_to(ROOT)),
        "rows_checked": len(picked),
        "max_abs_diff": worst,
    }
    print(
        f"  {name}: {len(picked)} rows against {table_path.name}, max |diff| = {worst:.2e}",
        file=sys.stderr,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/derived/demo")
    parser.add_argument("--sample", type=int, default=300, help="rows verified per table")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    emb_v2 = rows_of(SHAPES / "emblemes-v2.jsonl")
    mus_v2 = rows_of(SHAPES / "museums-v2.jsonl")
    mus_v3 = rows_of(SHAPES / "museums-v3.jsonl")

    # 1. v2-256: every region of each corpus, drawn by a seeded permutation, emblems first.
    t = time.time()
    rng = np.random.default_rng(args.seed)
    parts = []
    for rows in (emb_v2, mus_v2):
        m = descriptors_of(rows)
        parts.append(m[rng.choice(m.shape[0], size=min(120_000, m.shape[0]), replace=False)])
    atoms_v2 = fit_atoms(np.concatenate(parts), signs=256, seed=args.seed)
    np.savez(args.output / "atoms-v2-256.npz", **atoms_v2)
    print(f"atoms-v2-256 fitted in {time.time() - t:.0f}s", file=sys.stderr)
    report["atoms-v2-256"] = {
        "regions": ["emblemes-v2.jsonl", "museums-v2.jsonl"],
        "sampling": "all regions, seeded permutation per corpus",
        **verify(
            "formes muettes",
            SHAPES / "v2-256/shape-signs.npz",
            emb_v2 + mus_v2,
            lambda r: histogram(r, atoms_v2),
            sample=args.sample,
            seed=args.seed,
        ),
    }

    # 2. the pool atoms of repetition-v3 and v3-groups: emblems v2 then museums v3, no sampling.
    t = time.time()
    pool_rows = emb_v2 + mus_v3
    atoms_pool = fit_atoms(descriptors_of(pool_rows), signs=256, seed=args.seed)
    np.savez(args.output / "atoms-pool.npz", **atoms_pool)
    print(f"atoms-pool fitted in {time.time() - t:.0f}s", file=sys.stderr)
    report["atoms-pool"] = {
        "regions": ["emblemes-v2.jsonl", "museums-v3.jsonl"],
        "sampling": "none",
        **verify(
            "répétition",
            SHAPES / "repetition-v3.npz",
            pool_rows,
            lambda r: repetition(r, atoms_pool),
            sample=args.sample,
            seed=args.seed,
        ),
    }

    # 3. composites of v3-groups, on the pool atoms.
    t = time.time()
    described = []
    for row in pool_rows:
        if row["regions"]:
            d, _ = group_descriptions(
                row["regions"], atoms_pool, margin=0.15, ratio=3.0, min_parts=2
            )
            described.append(d)
    matrix = np.concatenate(described)
    model = MiniBatchKMeans(n_clusters=128, random_state=args.seed, n_init=8, batch_size=4096).fit(
        matrix
    )
    composites = model.cluster_centers_.astype(np.float32)
    np.savez(args.output / "composites-v3.npz", centres=composites)
    print(
        f"composites-v3 fitted on {matrix.shape[0]} groups in {time.time() - t:.0f}s",
        file=sys.stderr,
    )

    def composite_of(regions: list[dict]) -> np.ndarray:
        d, _ = group_descriptions(regions, atoms_pool, margin=0.15, ratio=3.0, min_parts=2)
        return composite_vector(d, composites)

    report["composites-v3"] = {
        "groups": int(matrix.shape[0]),
        "margin": 0.15,
        "ratio": 3.0,
        "min_parts": 2,
        **verify(
            "signes composites",
            SHAPES / "v3-groups/composite-signs.npz",
            pool_rows,
            composite_of,
            sample=args.sample,
            seed=args.seed,
        ),
    }

    # 4. v4-256: balanced, the same number of regions from each corpus, emblems first.
    t = time.time()
    rng = np.random.default_rng(args.seed)
    pools = [descriptors_of(emb_v2), descriptors_of(mus_v3)]
    take = min(120_000, min(m.shape[0] for m in pools))
    parts = [m[rng.choice(m.shape[0], size=take, replace=False)] for m in pools]
    atoms_v4 = fit_atoms(np.concatenate(parts), signs=256, seed=args.seed)
    np.savez(args.output / "atoms-v4-256.npz", **atoms_v4)
    print(f"atoms-v4-256 fitted in {time.time() - t:.0f}s", file=sys.stderr)
    report["atoms-v4-256"] = {
        "regions": ["emblemes-v2.jsonl", "museums-v3.jsonl"],
        "sampling": f"{take} per corpus, seeded",
        **verify(
            "formes v4",
            SHAPES / "v4-256/shape-signs.npz",
            pool_rows,
            lambda r: histogram(r, atoms_v4),
            sample=args.sample,
            seed=args.seed,
        ),
    }

    # 5. the 64-sign vocabulary and the relation keys of relations-wide64: pool rows, emblems
    # first, the ten largest regions of each picture, keys kept at support >= 30.
    t = time.time()
    atoms_64 = fit_atoms(descriptors_of(pool_rows), signs=64, seed=args.seed)
    np.savez(args.output / "atoms-64.npz", **atoms_64)
    support: collections.Counter = collections.Counter()
    for row in pool_rows:
        regions = row["regions"][:10]
        if len(regions) < 2:
            continue
        signs = assign(np.stack([descriptor(r) for r in regions]), atoms_64)
        for i, a in enumerate(regions):
            for j, b in enumerate(regions):
                if i != j:
                    support[(int(signs[i]), relation(a, b), int(signs[j]))] += 1
    keys = sorted(k for k, c in support.items() if c >= 30)
    np.savez(
        args.output / "relations-64.npz",
        sign_a=np.asarray([k[0] for k in keys], dtype=np.int32),
        relation=np.asarray([k[1] for k in keys]),
        sign_b=np.asarray([k[2] for k in keys], dtype=np.int32),
    )
    print(f"atoms-64 and {len(keys)} relation keys in {time.time() - t:.0f}s", file=sys.stderr)
    index = {k: n for n, k in enumerate(keys)}

    def relations_of(regions: list[dict]) -> np.ndarray:
        regions = regions[:10]
        v = np.zeros(len(keys), dtype=np.float32)
        if len(regions) < 2:
            return v
        signs = assign(np.stack([descriptor(r) for r in regions]), atoms_64)
        for i, a in enumerate(regions):
            for j, b in enumerate(regions):
                if i != j:
                    k = (int(signs[i]), relation(a, b), int(signs[j]))
                    if k in index:
                        v[index[k]] += 1.0
        return l2(v)

    report["relations-64"] = {
        "keys": len(keys),
        "min_support": 30,
        "max_regions": 10,
        **verify(
            "relations",
            SHAPES / "relations-wide64.npz",
            pool_rows,
            relations_of,
            sample=args.sample,
            seed=args.seed,
        ),
    }

    (args.output / "fit-report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    print(f"written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
