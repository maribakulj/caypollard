#!/usr/bin/env python3
"""A global layout channel, because a scene is an arrangement and not a large object.

Broken down by Iconclass division, the shape record places the best cross-medium
partner at median rank 458 for division 2 -- animals and plants, where a notation
names a thing -- and at 6 403 for division 1, religion, where it names an
episode. The pool is 12 479, so a random ordering would give about 6 240: on
scenes the record is worth nothing, and every aggregate figure reported so far
was an average over these two behaviours.

The reason is structural rather than a defect of tuning. A cat has a distinctive
silhouette; an Annunciation and a Nativity share figures and architecture whose
local shape statistics are generic. What two pictures of one episode do share is
*where things are*: a figure at the left facing one at the right, a mass low and
an opening high, a symmetry or its deliberate breaking.

So this channel describes the arrangement and nothing else:

* a coarse grid of ink density, contrast-normalised, which is the layout at a
  resolution too low to carry hatching or brushwork;
* left-right mirror agreement, and top-bottom, which is symmetry as a number;
* the vertical and horizontal mass profiles, which place a horizon and a centre;
* the spread and principal axis of the region centroids, which say whether the
  picture is one clump, two facing groups, or a scatter.

Nothing here is local. That is the point: the local channel already exists and is
measured to fail on exactly the material this one is built for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from caypollard.embeddings.store import save_embedding_table
from caypollard.provenance import read_jsonl


def mass_map(path: Path, *, grid: int) -> np.ndarray:
    """Ink density on a coarse grid, normalised so paper tone cannot register."""
    image = Image.open(path).convert("L").resize((grid * 8, grid * 8), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    # Ink is what departs from the page, in either direction: a dark figure on
    # light paper and a light one on a dark ground are the same arrangement.
    array = np.abs(array - float(np.median(array)))
    block = array.reshape(grid, 8, grid, 8).mean(axis=(1, 3))
    total = block.sum()
    return block / total if total > 0 else block


def composition(path: Path, regions: list[dict], *, grid: int) -> np.ndarray:
    block = mass_map(path, grid=grid)
    rows = block.sum(axis=1)
    columns = block.sum(axis=0)

    mirror_lr = float(1.0 - np.abs(block - block[:, ::-1]).sum() / 2.0)
    mirror_tb = float(1.0 - np.abs(block - block[::-1, :]).sum() / 2.0)

    centre = block[grid // 4 : 3 * grid // 4, grid // 4 : 3 * grid // 4].sum()
    flat = block.ravel()
    positive = flat[flat > 0]
    entropy = float(-(positive * np.log(positive)).sum() / np.log(max(flat.size, 2)))

    if regions:
        xs = np.asarray([r["centroid_x"] for r in regions], dtype=float)
        ys = np.asarray([r["centroid_y"] for r in regions], dtype=float)
        spread = float(np.hypot(xs.std(), ys.std()))
        cloud = np.stack([xs - xs.mean(), ys - ys.mean()])
        if cloud.shape[1] >= 2:
            values = np.linalg.eigvalsh(np.cov(cloud) + 1e-9 * np.eye(2))
            elongation = float(values.max() / max(values.min(), 1e-9))
        else:
            elongation = 1.0
        count = float(np.log1p(len(regions)))
    else:
        spread, elongation, count = 0.0, 1.0, 0.0

    return np.concatenate(
        [
            flat,
            rows,
            columns,
            np.asarray(
                [
                    mirror_lr,
                    mirror_tb,
                    float(centre),
                    entropy,
                    spread,
                    np.log1p(elongation),
                    count,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--regions", help="Segmented regions, for the centroid-cloud features")
    parser.add_argument("--image-dir", default="/")
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    by_item: dict[str, list[dict]] = {}
    if args.regions:
        for line in Path(args.regions).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_item[str(row["id"])] = row["regions"]

    records = read_jsonl(args.manifest)
    if args.limit:
        records = records[: args.limit]
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    skipped = 0
    for record in records:
        recorded = record.get("image_path")
        path = Path(recorded) if recorded else Path(args.image_dir) / str(record["filename"])
        if not path.is_file():
            skipped += 1
            continue
        try:
            vectors.append(
                composition(path, by_item.get(str(record["id"]), []), grid=args.grid)
            )
        except Exception:
            skipped += 1
            continue
        ids.append(str(record["id"]))

    if not ids:
        raise SystemExit("no usable image")
    save_embedding_table(
        args.output,
        ids=tuple(ids),
        vectors=np.stack(vectors),
        metadata={
            "family": "composition",
            "method": "coarse mass grid, symmetry, mass profiles, centroid-cloud statistics",
            "grid": args.grid,
            "dimensions": int(vectors[0].size),
        },
        normalize=True,
    )
    print(json.dumps({"items": len(ids), "skipped": skipped,
                      "dimensions": int(vectors[0].size)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
