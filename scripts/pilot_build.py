#!/usr/bin/env python3
"""Outline each proposed figure and object, then assemble the pilot's records.

For every picture described by ``scripts/pilot_describe.py``: SlimSAM (a 39 MB distilled
Segment Anything) cuts the figure or object inside its proposed box, and the mask's outer
contour, simplified to a few dozen vertices, becomes that item's outline -- its geometric
abstraction, and the shape the relations are measured against. Then
``caypollard.figures.build_record`` assembles the record, applying any corrections a
person saved from the viewer.

Run with ``uv run --with opencv-python-headless``. The SlimSAM weights are expected under
``data/models/slimsam-77`` (config.json, preprocessor_config.json, model.safetensors from
the Hugging Face repository Zigeng/SlimSAM-uniform-77).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from caypollard.figures import build_record

PILOTE = Path("data/derived/pilote")


def outline(mask: np.ndarray, detail: float) -> list[list[int]] | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    simple = cv2.approxPolyDP(contour, detail * cv2.arcLength(contour, True), True)
    return simple[:, 0, :].tolist() if len(simple) >= 3 else None


def segment(image: Image.Image, items: list[dict], model, processor, detail: float) -> dict:
    if not items:
        return {}
    boxes = [[list(map(float, it["box"])) for it in items]]
    inputs = processor(image, input_boxes=boxes, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0]
    scores = out.iou_scores[0]
    polygons = {}
    for i, it in enumerate(items):
        best = int(scores[i].argmax())
        mask = masks[i, best].numpy()
        # Keep the mask inside its box: SAM sometimes spills onto a neighbour.
        x0, y0, x1, y1 = (round(v) for v in it["box"])
        clipped = np.zeros_like(mask)
        clipped[max(y0, 0) : y1, max(x0, 0) : x1] = mask[max(y0, 0) : y1, max(x0, 0) : x1]
        polygons[it["id"]] = outline(clipped, detail)
    return polygons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("data/raw/emblematica/full"))
    parser.add_argument("--weights", type=Path, default=Path("data/models/slimsam-77"))
    parser.add_argument(
        "--detail", type=float, default=0.004, help="contour simplification, share of perimeter"
    )
    parser.add_argument("--resegment", action="store_true")
    args = parser.parse_args()

    model = processor = None
    (PILOTE / "masks").mkdir(parents=True, exist_ok=True)
    (PILOTE / "records").mkdir(parents=True, exist_ok=True)
    index = []
    for desc_path in sorted((PILOTE / "describe").glob("*.json")):
        name = desc_path.stem
        description = json.loads(desc_path.read_text())
        mask_path = PILOTE / "masks" / f"{name}.json"
        if args.resegment or not mask_path.exists():
            if model is None:
                model = SamModel.from_pretrained(args.weights).eval()
                processor = SamProcessor.from_pretrained(args.weights)
            image = Image.open(args.images / f"{name}.jpg").convert("RGB")
            items = description.get("figures", []) + description.get("objects", [])
            mask_path.write_text(json.dumps(segment(image, items, model, processor, args.detail)))
        polygons = json.loads(mask_path.read_text())
        corr_path = PILOTE / "corrections" / f"{name}.json"
        corrections = json.loads(corr_path.read_text()) if corr_path.exists() else None
        record = build_record(description, polygons, corrections)
        record["id"] = name
        record["image"] = f"data/raw/emblematica/full/{name}.jpg"
        (PILOTE / "records" / f"{name}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1)
        )
        index.append(
            {
                "id": name,
                "phrase": record["phrase"],
                "figures": [f["name"] for f in record["figures"]],
            }
        )
        print(f"{name}: {len(record['relations'])} relations", flush=True)
    (PILOTE / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
