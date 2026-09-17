#!/usr/bin/env python3
"""Render images into a representation from which the medium cannot be guessed.

A confound encoded redundantly in the input cannot be removed downstream:
projecting the holding-library direction out of a DINOv2 space left 91% of the
top-10 neighbourhoods unchanged, because paper texture, typography, ink density,
page layout and aspect ratio all carry the institution and all survive the
projection.

So the confound has to be attacked where it enters. This renders each image into
a form that keeps the shape of the motif and discards what identifies the
support, and it does so in stages so the contribution of each stage is
attributable rather than assumed:

``gray``
    Greyscale, squared to a fixed side. Aspect ratio is itself a strong format
    cue here -- full pages sit near 1.5, cropped picturae near 0.9 -- so
    squaring removes a cue before any edge detection is credited with it.
``edges``
    The above, then a Gaussian-smoothed Sobel magnitude, contrast-normalised.
    Tone, paper colour and ink density go; contour stays.

The referee is a linear probe on the resulting embeddings: if it can still name
the holding library, the rendering has not done its job, whatever it looks like.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from caypollard.provenance import read_jsonl


def render(path: Path, *, mode: str, side: int, sigma: float) -> Image.Image:
    image = Image.open(path).convert("L").resize((side, side), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if mode == "edges":
        smoothed = ndimage.gaussian_filter(array, sigma=sigma)
        gx = ndimage.sobel(smoothed, axis=1)
        gy = ndimage.sobel(smoothed, axis=0)
        array = np.hypot(gx, gy)
        # Percentile normalisation rather than min-max: a single specular
        # highlight or a dust speck would otherwise set the scale for the page.
        high = np.percentile(array, 99.0)
        array = np.clip(array / high, 0.0, 1.0) if high > 0 else array
    scaled = (array * 255.0).astype(np.uint8)
    return Image.fromarray(np.stack([scaled] * 3, axis=-1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("image_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--mode", choices=("gray", "edges"), required=True)
    parser.add_argument("--side", type=int, default=448)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    if args.limit:
        records = records[: args.limit]
    source = Path(args.image_dir)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    for record in records:
        name = str(record["filename"])
        origin = source / name
        target = destination / name
        if target.is_file():
            written += 1
            continue
        if not origin.is_file():
            skipped += 1
            continue
        try:
            render(origin, mode=args.mode, side=args.side, sigma=args.sigma).save(
                target, format="JPEG", quality=92
            )
            written += 1
        except Exception:
            skipped += 1
    print(json.dumps({"mode": args.mode, "written": written, "skipped": skipped,
                      "output_dir": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
