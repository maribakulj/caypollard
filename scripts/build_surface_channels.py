#!/usr/bin/env python3
"""Two channels the record deliberately refuses, built so they can be judged fairly.

Colour was measured once and set aside: within one collection, one format and
seventeen books -- so that only the scanning session and the artwork vary -- a raw
histogram names the volume 61.3% of the time against a 31.0% baseline. Reducing
to eleven basic terms brings that to 53.2% and grey-world balancing to 39.7%,
and giving the channel any weight degrades every column of the record.

That verdict was reached by judging colour on the wrong question. A channel that
identifies the scanning session is a *bad* channel for finding a subject and a
*good* one for finding the collection, and the regime benchmark can now ask each
question separately. So both surface channels are built properly rather than
excluded by reputation.

``palette``
    Eleven basic colour terms after Berlin and Kay, assigned in a perceptual
    space so that "a slightly greenish dark blue" lands on blue rather than
    between two bins, and after grey-world balancing so that a warm scanning
    lamp does not turn a whole volume yellow. Seven per cent of digitisations
    are effectively monochrome, so the channel has to degrade to its absence
    rather than break on it.

``signal``
    What the sensor and the operator left behind rather than what the object is:
    mean luminance and its spread, a coarse luminance histogram, and the share
    of energy in the high frequencies, which is grain, sharpening and
    compression. This is the confound named directly instead of being chased out
    of other channels -- and a confound that can be measured is a channel for
    the question it answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from caypollard.embeddings.store import save_embedding_table

# sRGB anchors for the eleven basic terms, converted to Lab once at import.
TERMS = {
    "noir": (0, 0, 0), "blanc": (255, 255, 255), "rouge": (196, 32, 33),
    "vert": (42, 140, 60), "jaune": (240, 200, 40), "bleu": (36, 74, 168),
    "brun": (120, 74, 38), "orange": (222, 120, 32), "rose": (232, 150, 172),
    "violet": (118, 54, 148), "gris": (128, 128, 128),
}


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in 0-255 to CIELAB, D65, vectorised over the last axis."""
    srgb = rgb.astype(np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883])
    ratio = xyz / white
    f = np.where(ratio > 0.008856, np.cbrt(ratio), 7.787 * ratio + 16 / 116)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


ANCHORS = to_lab(np.array(list(TERMS.values())))


def palette_of(image: Image.Image, side: int) -> np.ndarray:
    small = image.convert("RGB").resize((side, side), Image.BILINEAR)
    pixels = np.asarray(small, dtype=np.float64)
    # Grey-world: the average of a scene is assumed neutral, so a warm lamp is
    # divided out before any colour is named.
    means = pixels.reshape(-1, 3).mean(axis=0)
    means[means == 0] = 1.0
    pixels = np.clip(pixels * (means.mean() / means), 0, 255)
    lab = to_lab(pixels)
    distance = ((lab[:, :, None, :] - ANCHORS[None, None, :, :]) ** 2).sum(axis=-1)
    counts = np.bincount(distance.argmin(axis=-1).reshape(-1), minlength=len(TERMS))
    return counts.astype(np.float32) / max(counts.sum(), 1)


def signal_of(image: Image.Image, side: int) -> np.ndarray:
    grey = np.asarray(
        image.convert("L").resize((side, side), Image.BILINEAR), dtype=np.float32
    ) / 255.0
    histogram = np.histogram(grey, bins=12, range=(0.0, 1.0))[0].astype(np.float32)
    histogram /= max(histogram.sum(), 1)
    # A Laplacian's energy against the image's own energy: grain, sharpening and
    # compression live here, and the object mostly does not.
    laplace = (
        -4 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1]
        + grey[1:-1, :-2] + grey[1:-1, 2:]
    )
    roughness = float(np.abs(laplace).mean())
    return np.concatenate(
        [histogram, [float(grey.mean()), float(grey.std()), roughness,
                     float(np.percentile(grey, 95) - np.percentile(grey, 5))]]
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--palette-output", required=True)
    parser.add_argument("--signal-output", required=True)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r.get("image_path") and Path(r["image_path"]).is_file()]
    if args.limit:
        rows = rows[: args.limit]

    ids, palettes, signals = [], [], []
    for position, row in enumerate(rows, start=1):
        try:
            with Image.open(row["image_path"]) as handle:
                image = handle.convert("RGB")
                palettes.append(palette_of(image, args.side))
                signals.append(signal_of(image, args.side))
        except (OSError, ValueError):
            continue
        ids.append(str(row["id"]))
        if args.progress_every and position % args.progress_every == 0:
            print(f"{position}/{len(rows)}", flush=True)

    for path, vectors, note in (
        (args.palette_output, palettes, "11 basic colour terms, grey-world balanced"),
        (args.signal_output, signals, "luminance histogram, spread and high-frequency energy"),
    ):
        save_embedding_table(
            path,
            ids=tuple(ids),
            vectors=np.stack(vectors),
            metadata={
                "family": "surface",
                "method": note,
                "side": args.side,
                "warning": "identifies the scanning session; a channel for that question "
                           "and not for a subject",
            },
            normalize=True,
        )
    print(json.dumps({"items": len(ids), "palette": args.palette_output,
                      "signal": args.signal_output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
