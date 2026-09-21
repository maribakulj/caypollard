#!/usr/bin/env python3
"""Cut an image into named regions, the channel a shape index needs.

A coarse silhouette halves the distance to a cross-medium partner and still does
not recognise an anchor: it has learned not to see the paper, not to see the
motif. The missing step is naming, and naming needs regions to name.

Segmentation here is deliberately classical rather than learned. A learned
segmenter is trained on photographs and inherits their statistics, which is the
confound the whole exercise is trying to escape; a watershed over a smoothed
gradient asks only "where does one region end", which is a question a print and a
painting answer the same way. Each region is then described by properties that
survive a change of support:

* **shape** -- area, elongation, solidity, hole count, scale relative to the
  picture's own parts, the first Hu moments, and a
  twenty-four bin radial signature of the contour, all scale- and
  rotation-normalised. The signature is there because the moments alone were
  measured not to carry iconography: a representation built on them ranks a
  cross-medium partner well and cannot predict a notation better than counting;
* **position** -- the centroid in a normalised frame, so relations between
  regions can be read off later;
* **tone** -- whether the region is darker or lighter than the page, which is
  the only photometric fact that survives a monochrome scan.

Colour is deliberately absent. It identifies the scanning session -- within one
collection and one format it names the volume 61% of the time against a 31%
baseline -- so it belongs in a separate channel that is consulted and never
matched on.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from caypollard.provenance import read_jsonl


def holes(mask: np.ndarray) -> int:
    """How many enclosed holes the region has.

    Topology separates shapes that every metric descriptor confuses: a ring and a
    disc have the same area, elongation and solidity, and differ only in whether
    the middle is inside or outside. A wreath, an arch, a chain link and a letter
    O are all holed, and nothing in the descriptor said so until now.
    """
    filled = ndimage.binary_fill_holes(mask)
    difference = filled & ~mask
    if not difference.any():
        return 0
    _labelled, count = ndimage.label(difference)
    return int(count)


def radial_signature(mask: np.ndarray, *, bins: int = 24) -> list[float]:
    """Distance from the centroid to the region's edge, by angle.

    Moment invariants summarise a shape in four numbers and are known to
    discriminate poorly: an anchor and a cloud can share them. A radial signature
    keeps the contour's profile -- where the shape reaches out and where it is
    hollow -- at a fixed cost, and is made rotation-comparable by starting at the
    longest radius and scale-free by dividing through by the mean.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 12:
        return [0.0] * bins
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy
    radius = np.hypot(dx, dy)
    angle = (np.arctan2(dy, dx) + 2 * math.pi) % (2 * math.pi)
    index = np.minimum((angle / (2 * math.pi) * bins).astype(int), bins - 1)
    profile = np.zeros(bins, dtype=float)
    for slot in range(bins):
        selected = radius[index == slot]
        profile[slot] = float(selected.max()) if selected.size else 0.0
    mean = profile.mean()
    if mean <= 0:
        return [0.0] * bins
    profile = profile / mean
    # Rotation is a nuisance here, not a signal: a tilted anchor is an anchor.
    # Starting the profile at its longest radius removes it.
    shift = int(np.argmax(profile))
    return [round(float(v), 4) for v in np.roll(profile, -shift)]


def hu_moments(mask: np.ndarray) -> list[float]:
    """Scale- and translation-normalised moment invariants of a binary region."""
    ys, xs = np.nonzero(mask)
    if xs.size < 8:
        return [0.0] * 4
    x, y = xs.astype(float), ys.astype(float)
    x -= x.mean()
    y -= y.mean()
    total = float(xs.size)

    def mu(p: int, q: int) -> float:
        return float((x**p * y**q).sum() / total ** (1 + (p + q) / 2))

    n20, n02, n11 = mu(2, 0), mu(0, 2), mu(1, 1)
    n30, n03, n21, n12 = mu(3, 0), mu(0, 3), mu(2, 1), mu(1, 2)
    return [
        n20 + n02,
        (n20 - n02) ** 2 + 4 * n11**2,
        (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
        (n30 + n12) ** 2 + (n21 + n03) ** 2,
    ]


def segment(path: Path, *, side: int, min_area: float, max_regions: int) -> list[dict[str, Any]]:
    image = Image.open(path).convert("L").resize((side, side), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    smoothed = ndimage.gaussian_filter(array, sigma=side / 64.0)
    low, high = np.percentile(smoothed, 2.0), np.percentile(smoothed, 98.0)
    smoothed = np.clip((smoothed - low) / (high - low), 0.0, 1.0) if high > low else smoothed

    # Ink is dark on light in a print and can be either in a painting, so both
    # polarities are segmented and the regions pooled.
    page_tone = float(np.median(smoothed))
    regions: list[dict[str, Any]] = []
    for polarity, name in ((-1.0, "sombre"), (1.0, "clair")):
        binary = (polarity * (smoothed - page_tone)) > 0.12
        binary = ndimage.binary_opening(binary, np.ones((3, 3)))
        labelled, count = ndimage.label(binary)
        if count == 0:
            continue
        for index in range(1, count + 1):
            mask = labelled == index
            area = float(mask.sum()) / mask.size
            if area < min_area:
                continue
            ys, xs = np.nonzero(mask)
            height = (ys.max() - ys.min() + 1) / side
            width = (xs.max() - xs.min() + 1) / side
            box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            filled = float(mask.sum()) / max(box, 1)
            regions.append(
                {
                    "area": round(area, 5),
                    "elongation": round(max(height, width) / max(min(height, width), 1e-6), 3),
                    "solidity": round(filled, 3),
                    "tone": name,
                    "centroid_x": round(float(xs.mean()) / side, 4),
                    "centroid_y": round(float(ys.mean()) / side, 4),
                    "extent_x": round(width, 4),
                    "extent_y": round(height, 4),
                    "hu": [round(v, 6) if math.isfinite(v) else 0.0 for v in hu_moments(mask)],
                    "radial": radial_signature(mask),
                    "holes": holes(mask),
                }
            )
    regions.sort(key=lambda region: -region["area"])
    regions = regions[:max_regions]

    # Scale is only meaningful against the picture's own other parts. A figure
    # twice the size of every other figure is the hierarchy of importance of
    # medieval art, and it reads identically whether the picture is a miniature
    # or a mural; an absolute area does not.
    if regions:
        areas = sorted(region["area"] for region in regions)
        median = areas[len(areas) // 2]
        largest = areas[-1]
        for region in regions:
            region["scale_vs_median"] = round(region["area"] / max(median, 1e-9), 3)
            region["scale_vs_largest"] = round(region["area"] / max(largest, 1e-9), 4)
    return regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("image_dir")
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument("--min-area", type=float, default=0.004)
    parser.add_argument("--max-regions", type=int, default=24)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    if args.limit:
        records = records[: args.limit]
    source = Path(args.image_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    counts: list[int] = []
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
                # As in embed_images: a merged manifest knows where its images are.
            recorded = record.get("image_path")
            path = Path(recorded) if recorded else source / str(record["filename"])
            if not path.is_file():
                skipped += 1
                continue
            try:
                regions = segment(
                    path,
                    side=args.side,
                    min_area=args.min_area,
                    max_regions=args.max_regions,
                )
            except Exception:
                skipped += 1
                continue
            counts.append(len(regions))
            handle.write(
                json.dumps(
                    {"id": str(record["id"]), "regions": regions}, ensure_ascii=False
                )
                + "\n"
            )
            written += 1
    print(
        json.dumps(
            {
                "written": written,
                "skipped": skipped,
                "median_regions": int(np.median(counts)) if counts else 0,
                "mean_regions": round(float(np.mean(counts)), 2) if counts else 0,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
