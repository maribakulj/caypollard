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
    Tone, paper colour and ink density go; contour stays -- and so, it turns out,
    does the hatching that identifies an engraving.
``shape``
    Reduced to a coarse grid before being restored to size, which averages
    hatching and brushwork out of existence while keeping gross composition.
    An engraving is already made of lines and a painting is not, so any
    rendering that preserves line structure preserves the medium; this one
    deliberately cannot.
``silhouette``
    The above, then thresholded at its own median into black and white
    regions. A cheap approximation to an abstract drawing, and one that only
    works where the ground is uniform: on a museum photograph it recovers the
    object's outline, on an engraving it thresholds a coarse average of
    hatching and returns noise.
``mass``
    Figure and ground separated *before* anything is averaged. The full-
    resolution greyscale is split into ink and paper at its Otsu threshold,
    that binary mask is downsampled so each cell holds a local ink fraction,
    and the fraction is thresholded in turn. The order is the whole point: a
    line drawing has no mass until its lines are counted, and averaging grey
    levels first destroys the lines that were going to be counted.

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


def render(path: Path, *, mode: str, side: int, sigma: float, coarse: int = 48) -> Image.Image:
    image = Image.open(path).convert("L").resize((side, side), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if mode == "mass":
        # Otsu on the full-resolution greyscale: an engraving is ink on paper,
        # and that is a genuinely bimodal quantity before any averaging.
        counts, edges = np.histogram(array, bins=256, range=(0.0, 1.0))
        weight = counts.cumsum()
        total = weight[-1]
        centres = (edges[:-1] + edges[1:]) / 2.0
        mean = (counts * centres).cumsum()
        with np.errstate(invalid="ignore", divide="ignore"):
            between = (mean[-1] * weight / total - mean) ** 2 / (
                weight * (total - weight) / total**2
            )
        threshold = centres[int(np.nanargmax(between))]
        ink = (array < threshold).astype(np.float32)
        # Local ink fraction, which is where the mass of a line drawing lives.
        # A resize averages the mask, and unlike a reshape it does not require
        # the grid to divide the side.
        fraction = np.asarray(
            Image.fromarray((ink * 255).astype(np.uint8)).resize(
                (coarse, coarse), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        array = (fraction > max(fraction.mean(), 0.15)).astype(np.float32)
        array = np.asarray(
            Image.fromarray((array * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    if mode in {"shape", "silhouette"}:
        small = Image.fromarray((array * 255).astype(np.uint8)).resize(
            (coarse, coarse), Image.LANCZOS
        )
        array = np.asarray(small, dtype=np.float32) / 255.0
        # Local contrast normalisation: a dark paper and a bright one should give
        # the same silhouette, and only the relative values carry the shape.
        low, high = np.percentile(array, 2.0), np.percentile(array, 98.0)
        array = np.clip((array - low) / (high - low), 0.0, 1.0) if high > low else array
        if mode == "silhouette":
            array = (array > np.median(array)).astype(np.float32)
        array = np.asarray(
            Image.fromarray((array * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
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
    parser.add_argument(
        "--mode", choices=("gray", "edges", "shape", "silhouette", "mass"), required=True
    )
    parser.add_argument(
        "--coarse",
        type=int,
        default=48,
        help="Grid the image is reduced to before restoration, for shape modes",
    )
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
        # A merged manifest spreads its images across one directory per
        # collection, so the record carries the path and image_dir is only the
        # fallback for a single-collection corpus.
        recorded = record.get("image_path")
        origin = Path(recorded) if recorded else source / name
        target = destination / name
        if target.is_file():
            written += 1
            continue
        if not origin.is_file():
            skipped += 1
            continue
        try:
            render(
                origin, mode=args.mode, side=args.side, sigma=args.sigma, coarse=args.coarse
            ).save(
                target, format="JPEG", quality=92
            )
            written += 1
        except Exception:
            skipped += 1
    print(json.dumps({"mode": args.mode, "written": written, "skipped": skipped,
                      "output_dir": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
