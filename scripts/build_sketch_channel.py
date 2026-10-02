#!/usr/bin/env python3
"""Strokes for a whole corpus: the contour drawing of every picture, and a vocabulary of strokes.

For each picture of the manifests: the two-colour contour drawing is written to disk (so the
same DINOv2 that embeds the pool can embed the drawings), and the forty longest strokes of
the sketch are described by `caypollard.sketch.stroke_descriptor`. The strokes are then
clustered into a closed vocabulary -- the same number drawn from each corpus, seeded --
whose centres are SAVED next to the table, so a new picture can be assigned to it. The
picture's vector is the histogram of its stroke signs weighted by stroke length, L2.

Outputs:
  <renders>/<id-without-prefix>.png        the contour drawing (two colours)
  <renders>-manifest.jsonl                 manifest of the drawings, for embed_images.py
  <output>/strokes.jsonl                   descriptors of the sketch strokes per picture
  <output>/stroke-signs.npz (+ metadata)   the table
  <output>/stroke-atoms.npz                centre, scale, centres of the vocabulary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from caypollard import sketch as sketching
from caypollard.embeddings.store import save_embedding_table

ROOT = Path(__file__).resolve().parent.parent


def one(
    job: tuple[str, str, str, dict],
) -> tuple[str, str, list[list[float]], list[float], str | None]:
    item_id, image_path, render_path, params = job
    try:
        with Image.open(image_path) as image:
            sk = sketching.sketch(image.convert("RGB"), params)
        sketching.edges_image(sk).save(render_path, "PNG")
        described = [sketching.stroke_descriptor(line, sk.size).tolist() for line in sk.sketch]
        lengths = [sketching.length(line) for line in sk.sketch]
        return item_id, str(Path(render_path).name), described, lengths, None
    except Exception as error:  # counted, not fatal
        return item_id, "", [], [], f"{type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+", metavar="CORPUS=PATH")
    parser.add_argument(
        "--renders", type=Path, required=True, help="directory for the contour drawings"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signs", type=int, default=128)
    parser.add_argument("--sample-per-corpus", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--params", default="{}", help="JSON overrides of caypollard.sketch.DEFAULTS"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    params = {**sketching.DEFAULTS, **json.loads(args.params)}
    args.renders.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    per_corpus: dict[str, list[dict]] = {}
    for spec in args.manifest:
        name, _, path = spec.partition("=")
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = [r for r in rows if r.get("image_path") and Path(r["image_path"]).exists()]
        per_corpus[name] = rows[: args.limit] if args.limit else rows

    # 1. strokes and drawings, in parallel
    t = time.time()
    strokes_path = args.output / "strokes.jsonl"
    done: dict[str, dict] = {}
    if strokes_path.exists():
        for line in strokes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    jobs = []
    for rows in per_corpus.values():
        for r in rows:
            if r["id"] in done:
                continue
            render_path = args.renders / (r["id"].split(":")[-1] + ".png")
            jobs.append((r["id"], r["image_path"], str(render_path), params))
    total = sum(len(v) for v in per_corpus.values())
    print(f"{total} pictures, {len(done)} already done, {len(jobs)} to sketch", file=sys.stderr)
    failures = 0
    with strokes_path.open("a", encoding="utf-8") as out, ProcessPoolExecutor() as pool:
        for n, (item_id, render_name, described, lengths, error) in enumerate(
            pool.map(one, jobs, chunksize=32), 1
        ):
            if error:
                failures += 1
                continue
            row = {"id": item_id, "render": render_name, "strokes": described, "lengths": lengths}
            done[item_id] = row
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)} ({time.time() - t:.0f}s)", file=sys.stderr, end="\r")
    print(f"\nstrokes done in {time.time() - t:.0f}s, {failures} failures", file=sys.stderr)

    # 2. the manifest of drawings, for embed_images.py
    with (args.output / "renders-manifest.jsonl").open("w", encoding="utf-8") as out:
        for name, rows in per_corpus.items():
            for r in rows:
                if r["id"] in done and done[r["id"]]["render"]:
                    out.write(
                        json.dumps(
                            {"id": r["id"], "filename": done[r["id"]]["render"], "corpus": name}
                        )
                        + "\n"
                    )

    # 3. the vocabulary: the same number of strokes from each corpus, seeded
    rng = np.random.default_rng(args.seed)
    pools = {}
    for name, rows in per_corpus.items():
        m = [d for r in rows if r["id"] in done for d in done[r["id"]]["strokes"]]
        if m:
            pools[name] = np.asarray(m, dtype=np.float32)
    take = min(args.sample_per_corpus, min(m.shape[0] for m in pools.values()))
    training = np.concatenate(
        [m[rng.choice(m.shape[0], size=take, replace=False)] for m in pools.values()]
    )
    centre, scale = training.mean(axis=0), training.std(axis=0)
    scale[scale == 0] = 1.0
    model = MiniBatchKMeans(
        n_clusters=args.signs, random_state=args.seed, n_init=8, batch_size=4096
    )
    model.fit((training - centre) / scale)
    centres = model.cluster_centers_.astype(np.float32)
    np.savez(args.output / "stroke-atoms.npz", centre=centre, scale=scale, centres=centres)

    # 4. the table: stroke signs weighted by stroke length
    ids, vectors = [], []
    for rows in per_corpus.values():
        for r in rows:
            row = done.get(r["id"])
            if not row or not row["strokes"]:
                continue
            z = (np.asarray(row["strokes"], dtype=np.float32) - centre) / scale
            assigned = ((z[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            h = np.zeros(args.signs, dtype=np.float32)
            for sign, weight in zip(assigned, row["lengths"], strict=True):
                h[int(sign)] += float(weight)
            if h.any():
                ids.append(r["id"])
                vectors.append(h)
    save_embedding_table(
        args.output / "stroke-signs.npz",
        ids=tuple(ids),
        vectors=np.stack(vectors),
        metadata={
            "family": "symbolic",
            "method": "stroke signs of the sketch (40 longest strokes), weighted by length",
            "signs": args.signs,
            "sampled_per_corpus": int(take),
            "params": params,
            "features": list(sketching.STROKE_FEATURES),
        },
        normalize=True,
    )
    print(f"{len(ids)} pictures in stroke-signs.npz", file=sys.stderr)


if __name__ == "__main__":
    main()
