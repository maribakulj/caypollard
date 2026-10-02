#!/usr/bin/env python3
"""Outline each proposed figure and object, then assemble the pilot's records.

For every picture described by ``scripts/pilot_describe.py``: SlimSAM (a 39 MB distilled
Segment Anything) cuts the figure or object inside its proposed box -- for a figure, guided
by its own body points and kept off the ground by points far from them -- and the mask's outer
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
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from caypollard.figures import build_record, torso_axis

PILOTE = Path("data/derived/pilote")
BONES = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
)


def outline(mask: np.ndarray, detail: float) -> list[list[int]] | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    simple = cv2.approxPolyDP(contour, detail * cv2.arcLength(contour, True), True)
    return simple[:, 0, :].tolist() if len(simple) >= 3 else None


def skeleton_distance(shape: tuple[int, int], k: dict) -> np.ndarray:
    """Each pixel's distance to the figure's skeleton (its bones and its points)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    best = np.full((h, w), np.inf, np.float32)
    for a, b in BONES:
        if k.get(a) and k.get(b):
            (x0, y0), (x1, y1) = k[a], k[b]
            dx, dy = x1 - x0, y1 - y0
            t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / ((dx * dx + dy * dy) or 1), 0, 1)
            best = np.minimum(best, np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy)))
    for p in k.values():
        best = np.minimum(best, np.hypot(xx - p[0], yy - p[1]))
    return best


def torso_length(k: dict) -> float:
    axis = torso_axis(k)
    return math.dist(*axis) if axis else 100.0


def in_box(mask: np.ndarray, box: list[float]) -> np.ndarray:
    """The mask cut to its box: SAM sometimes spills onto a neighbour."""
    x0, y0, x1, y1 = (round(v) for v in box)
    clipped = np.zeros_like(mask)
    clipped[max(y0, 0) : y1, max(x0, 0) : x1] = mask[max(y0, 0) : y1, max(x0, 0) : x1]
    return clipped


def masks_for(model, processor, image, embeddings, **prompt) -> np.ndarray:
    inputs = processor(image, return_tensors="pt", **prompt)
    inputs.pop("pixel_values")
    with torch.no_grad():
        out = model(image_embeddings=embeddings, multimask_output=True, **inputs)
    return processor.image_processor.post_process_masks(
        out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0][0].numpy()


def figure_mask(model, processor, image, embeddings, figure: dict, others: list[dict]):
    """Cut one figure out of a busy ground.

    A box alone is enough on blank paper; against a hatched landscape SAM takes the whole
    box. So the prompt also carries the figure's own body points (inside), the other
    figures' points that fall in its box (outside), and a grid of points in the box far from
    its skeleton (outside: likely ground). Of SAM's three proposals the one kept covers the
    most of the body points while spending the least area far from the skeleton.
    """
    k = {n: p for n, p in (figure.get("keypoints") or {}).items() if p}
    h, w = image.size[1], image.size[0]
    distance = skeleton_distance((h, w), k)
    far = 0.7 * torso_length(k)
    points = [list(map(float, p)) for p in k.values()]
    labels = [1] * len(points)
    x0, y0, x1, y1 = figure["box"]
    for other in others:
        for p in (other.get("keypoints") or {}).values():
            if p and x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                points.append(list(map(float, p)))
                labels.append(0)
    for gx in np.linspace(x0, x1, 7)[1:-1]:
        for gy in np.linspace(y0, y1, 7)[1:-1]:
            if k and distance[min(int(gy), h - 1), min(int(gx), w - 1)] > far:
                points.append([float(gx), float(gy)])
                labels.append(0)
    prompt = {"input_boxes": [[list(map(float, figure["box"]))]]}
    if points:
        prompt |= {"input_points": [[points]], "input_labels": [[labels]]}
    best, best_score = None, -np.inf
    for mask in masks_for(model, processor, image, embeddings, **prompt):
        mask = in_box(mask, figure["box"])
        if not mask.any():
            continue
        covered = (
            np.mean([mask[min(int(p[1]), h - 1), min(int(p[0]), w - 1)] for p in k.values()])
            if k
            else 0.0
        )
        stray = (mask & (distance > 0.6 * torso_length(k))).sum() / mask.sum()
        if covered - stray > best_score:
            best, best_score = mask, covered - stray
    return best


def segment(image: Image.Image, description: dict, model, processor, detail: float) -> dict:
    with torch.no_grad():
        embeddings = model.get_image_embeddings(
            processor(image, return_tensors="pt")["pixel_values"]
        )
    figures = description.get("figures", [])
    polygons = {}
    for f in figures:
        mask = figure_mask(
            model, processor, image, embeddings, f, [o for o in figures if o is not f]
        )
        polygons[f["id"]] = outline(mask, detail) if mask is not None else None
    for o in description.get("objects", []):
        proposals = masks_for(
            model, processor, image, embeddings, input_boxes=[[list(map(float, o["box"]))]]
        )
        # An object is compact: keep the proposal that fills its box best without spilling.
        fills = [
            in_box(m, o["box"]).sum() / max(m.sum(), 1) * in_box(m, o["box"]).sum()
            for m in proposals
        ]
        polygons[o["id"]] = outline(in_box(proposals[int(np.argmax(fills))], o["box"]), detail)
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
            polygons = segment(image, description, model, processor, args.detail)
            mask_path.write_text(json.dumps(polygons))
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
