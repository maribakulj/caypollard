#!/usr/bin/env python3
"""Transcribe a picture into named nodes from the discipline's own vocabulary.

Every naming route tried so far has produced numbers. A cluster is "sign 312",
and an art historian shown sign 312 has no way to say it is wrong. The record
only becomes arguable when its nodes carry names that exist outside this
project, which is what Iconclass supplies.

The output is **constrained by construction**: a region is scored against a
closed vocabulary and assigned the best member of it. There is no free text to
generate, so there is no channel through which an author's style or a page's
format could re-enter, and no output that falls outside the vocabulary.

Three properties keep this from being circular, and each is a decision rather
than an accident.

*The vocabulary is chosen by the hierarchy, not by the data.* Every Iconclass
notation of at most four characters that Wikidata gives a label to, and nothing
else -- 472 concepts. Choosing them by how often they occur in the annotation
being predicted is what made the ``isotype-*`` channels pure leakage, and is
exactly what is avoided here.

*The annotation is never read.* Names come from pixels and from the vocabulary.
This script does not open the manifest's iconclass field.

*Two controls ship with the result.* The names are also assigned at random from
the same vocabulary with the same per-picture counts, so a gain can be checked
against the arrangement alone; and the record is a histogram of names by
position, carrying no texture, so the medium probe can be run on it downstream.

The record is nodes by place: a coarse grid, and in each cell the names of the
parts whose centre falls there, weighted by area. That is the form the pooling
measurement selected -- *where* the parts are pays, and how they stand to each
other does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from caypollard.embeddings.store import save_embedding_table
from caypollard.vision.encoders import _projected_image_features

TEMPLATES = (
    "a depiction of {}",
    "{}",
    "an artwork showing {}",
)


def vocabulary(path: Path, *, max_depth: int) -> list[tuple[str, str]]:
    """Iconclass notations of at most ``max_depth`` characters, with a label."""
    seen: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        notation = str(row["notation"])
        matches = row.get("wikidata") or []
        if len(notation) <= max_depth and matches:
            label = str(matches[0].get("label") or "").strip()
            if label and notation not in seen:
                seen[notation] = label
    return sorted(seen.items())


def crop(image: Image.Image, region: dict, *, margin: float) -> Image.Image:
    """The region's bounding box, with a margin, in image coordinates."""
    width, height = image.size
    half_x = max(float(region.get("extent_x", 0.05)), 0.02) / 2 + margin
    half_y = max(float(region.get("extent_y", 0.05)), 0.02) / 2 + margin
    cx = float(region.get("centroid_x", 0.5))
    cy = float(region.get("centroid_y", 0.5))
    box = (
        max(int((cx - half_x) * width), 0),
        max(int((cy - half_y) * height), 0),
        min(int((cx + half_x) * width), width),
        min(int((cy + half_y) * height), height),
    )
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return image
    return image.crop(box)


def cell_of(region: dict, side: int) -> int:
    x = min(int(float(region.get("centroid_x", 0.5)) * side), side - 1)
    y = min(int(float(region.get("centroid_y", 0.5)) * side), side - 1)
    return y * side + x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("regions")
    parser.add_argument("manifest")
    parser.add_argument("--vocabulary", default="data/derived/iconclass-wikidata.jsonl")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=12,
                        help="Largest regions named per picture")
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--transcriptions", required=True,
                        help="Readable jsonl: which name each part was given")
    parser.add_argument("--output", required=True, help="The record table")
    parser.add_argument("--shuffled-output",
                        help="The same record with names drawn at random from the "
                             "vocabulary, keeping each picture's counts and places")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import CLIPModel, CLIPProcessor

    concepts = vocabulary(Path(args.vocabulary), max_depth=args.max_depth)
    if not concepts:
        raise SystemExit("the vocabulary is empty at this depth")
    notations = [notation for notation, _ in concepts]
    labels = [label for _, label in concepts]

    paths: dict[str, str] = {}
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        recorded = row.get("image_path")
        if recorded:
            paths[str(row["id"])] = str(recorded)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    with torch.inference_mode():
        columns = []
        for template in TEMPLATES:
            prompts = [template.format(label) for label in labels]
            batch = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            # Transformers 5 returns the whole output object where 4 returned
            # the tensor; the repository already has the unwrapper.
            features = _projected_image_features(model.get_text_features(**batch))
            columns.append(features / features.norm(dim=-1, keepdim=True))
        # Averaging the templates before normalising again is the standard
        # zero-shot recipe: one phrasing of a concept is a sample of it, not the
        # concept.
        text = torch.stack(columns).mean(dim=0)
        text = text / text.norm(dim=-1, keepdim=True)

    rows: list[dict[str, Any]] = []
    for line in Path(args.regions).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row["id"]) in paths and row.get("regions"):
            rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    width = args.grid * args.grid * len(concepts)
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    shuffled: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)
    transcriptions = Path(args.transcriptions)
    transcriptions.parent.mkdir(parents=True, exist_ok=True)

    with transcriptions.open("w", encoding="utf-8") as sink:
        for position, row in enumerate(rows):
            item = str(row["id"])
            regions = sorted(
                row["regions"], key=lambda r: -float(r.get("area", 0.0))
            )[: args.max_nodes]
            try:
                with Image.open(paths[item]) as handle:
                    image = handle.convert("RGB")
                    crops = [crop(image, region, margin=args.margin) for region in regions]
            except (OSError, ValueError):
                continue
            if not crops:
                continue
            with torch.inference_mode():
                assigned: list[int] = []
                for start in range(0, len(crops), args.batch_size):
                    batch = processor(
                        images=crops[start : start + args.batch_size], return_tensors="pt"
                    )
                    batch = {key: value.to(device) for key, value in batch.items()}
                    features = _projected_image_features(model.get_image_features(**batch))
                    features = features / features.norm(dim=-1, keepdim=True)
                    assigned.extend((features @ text.T).argmax(dim=-1).tolist())

            vector = np.zeros(width, dtype=np.float32)
            noise = np.zeros(width, dtype=np.float32)
            nodes = []
            for region, choice in zip(regions, assigned, strict=True):
                cell = cell_of(region, args.grid)
                weight = float(region.get("area", 0.0)) + 1e-3
                vector[cell * len(concepts) + choice] += weight
                noise[cell * len(concepts) + int(rng.integers(len(concepts)))] += weight
                nodes.append(
                    {
                        "notation": notations[choice],
                        "name": labels[choice],
                        "cell": cell,
                        "area": round(float(region.get("area", 0.0)), 5),
                    }
                )
            ids.append(item)
            vectors.append(vector)
            shuffled.append(noise)
            sink.write(
                json.dumps({"id": item, "nodes": nodes}, ensure_ascii=False) + "\n"
            )
            if position % 500 == 0 and device == "mps":
                torch.mps.empty_cache()

    save_embedding_table(
        args.output,
        ids=tuple(ids),
        vectors=np.stack(vectors),
        metadata={
            "family": "symbolic",
            "method": "Iconclass nodes assigned by constrained zero-shot scoring, by grid cell",
            "vocabulary": len(concepts),
            "max_depth": args.max_depth,
            "grid": args.grid,
            "max_nodes": args.max_nodes,
            "model": args.model,
            "templates": list(TEMPLATES),
        },
        normalize=True,
    )
    if args.shuffled_output:
        save_embedding_table(
            args.shuffled_output,
            ids=tuple(ids),
            vectors=np.stack(shuffled),
            metadata={
                "family": "control",
                "method": "the same places and counts, names drawn at random from the "
                          "same vocabulary -- what the arrangement alone is worth",
                "vocabulary": len(concepts),
                "seed": args.seed,
            },
            normalize=True,
        )
    print(
        json.dumps(
            {
                "pictures": len(ids),
                "vocabulary": len(concepts),
                "dimensions": width,
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
