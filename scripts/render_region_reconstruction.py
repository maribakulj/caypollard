#!/usr/bin/env python3
"""Redraw a picture from nothing but its typed signs.

The region channel fails at naming on both corpora, including the one where a
plain binary silhouette of the same picture succeeds by twenty points. Two
diagnoses fit that gap and they call for opposite repairs. Either the
segmentation and its vocabulary already throw the information away, in which
case no classifier over regions can recover it; or the vocabulary keeps it and
the loss happens afterwards, when a bag of parts is summarised by a mean and a
max and the layout goes with it.

This decides between them without a new classifier. Each region is drawn back
into an image from its description alone -- its contour profile, its size, its
place, its tone -- and the reconstruction is encoded and scored on the same
ladder as the renderings it is meant to imitate. If it scores like the
silhouette, the signs carry the picture and the fault is in how they were
consumed. If it scores like the region bag, the signs are where the picture was
lost.

The reconstruction is also the only artefact in this project a historian can
argue with directly: it is the picture as the sign vocabulary sees it, and one
look says whether that is a cat or a smear. Orientation is the one part of the
vocabulary that was discarded on purpose -- the radial signature is rotated to
start at its longest radius, a tilted anchor being an anchor -- so the bin it
started from is now kept and rolled back here, with ``--no-orientation`` to
score the reconstruction both ways. A single motif is orientation-free in a way
a scene is not.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

GROUND = 128
TONES = {"clair": 235, "sombre": 20}


def polygon(region: dict, side: int, *, oriented: bool) -> list[tuple[float, float]]:
    """The region's contour, placed and scaled inside the frame."""
    profile = np.asarray(region.get("radial") or [], dtype=np.float64)
    if profile.size < 3 or not np.isfinite(profile).all() or profile.max() <= 0:
        return []
    # The stored profile begins at the region's longest radius, so drawing it
    # from angle zero puts every part at an orientation it never had. Rolling
    # it back by the bin it started from is the only way the reconstruction can
    # be about the vocabulary rather than about that normalisation.
    if oriented:
        profile = np.roll(profile, int(region.get("radial_start", 0)))
    angles = np.arange(profile.size) * (2 * math.pi / profile.size)
    xs, ys = profile * np.cos(angles), profile * np.sin(angles)
    span_x = max(xs.max() - xs.min(), 1e-6)
    span_y = max(ys.max() - ys.min(), 1e-6)
    # The profile is scale-free, so the bounding box has to come from the
    # descriptor that kept a size: the region's extent in each direction.
    half_x = max(float(region.get("extent_x", 0.1)), 0.01) * side / 2
    half_y = max(float(region.get("extent_y", 0.1)), 0.01) * side / 2
    cx = float(region.get("centroid_x", 0.5)) * side
    cy = float(region.get("centroid_y", 0.5)) * side
    return [
        (cx + x * 2 * half_x / span_x, cy + y * 2 * half_y / span_y)
        for x, y in zip(xs, ys, strict=True)
    ]


def reconstruct(regions: list[dict], side: int, *, oriented: bool = True) -> Image.Image:
    canvas = Image.new("L", (side, side), GROUND)
    draw = ImageDraw.Draw(canvas)
    # Largest first, so a part that sits inside another is not buried by it.
    for region in sorted(regions, key=lambda r: -float(r.get("area", 0.0))):
        points = polygon(region, side, oriented=oriented)
        if len(points) >= 3:
            draw.polygon(points, fill=TONES.get(str(region.get("tone")), GROUND))
    return canvas.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("regions")
    parser.add_argument("output_dir")
    parser.add_argument("--manifest", help="Restrict to the ids this manifest carries")
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Draw every contour from angle zero, as the stored profile does. "
             "The contribution of orientation is then measured rather than assumed.",
    )
    parser.add_argument("--side", type=int, default=448)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    keep: set[str] | None = None
    if args.manifest:
        keep = {
            str(json.loads(line)["id"])
            for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for line in Path(args.regions).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item = str(row["id"])
        if keep is not None and item not in keep:
            continue
        if args.limit and written >= args.limit:
            break
        name = item.split(":")[-1] + ".jpg"
        target = destination / name
        if target.is_file():
            written += 1
            continue
        if not row.get("regions"):
            skipped += 1
            continue
        reconstruct(
            row["regions"], args.side, oriented=not args.no_orientation
        ).save(target, format="JPEG", quality=92)
        written += 1
    print(json.dumps({"written": written, "skipped": skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
