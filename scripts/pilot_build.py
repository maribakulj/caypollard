#!/usr/bin/env python3
"""Outline each proposed figure and object, then assemble the pilot's records.

For every picture described by ``scripts/pilot_describe.py``: SlimSAM (a 39 MB distilled
Segment Anything) cuts the figure or object inside its proposed box -- for a figure, guided
by its own body points and kept off the ground by points far from them -- and the mask's outer
contour, simplified to a few dozen vertices, becomes that item's outline -- its geometric
abstraction, and the shape the relations are measured against. Then
``caypollard.figures.build_record`` assembles the record, applying any corrections a
person saved from the viewer.

Run with ``uv run --with opencv-python-headless``. Two sets of weights are expected, each as
config.json, preprocessor_config.json and model.safetensors from Hugging Face:
``data/models/slimsam-77`` (Zigeng/SlimSAM-uniform-77, 39 MB) and
``data/models/sam-vit-base`` (facebook/sam-vit-base, 375 MB); each figure keeps the better cut.
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

from caypollard.figures import BONES, apply_corrections, build_record, torso_axis

PILOTE = Path("data/derived/pilote")


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


UPPER = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
LOWER = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")


def coverage(mask: np.ndarray, points) -> float:
    """Share of the points the mask reaches within 8 px of: ankles and wrists sit on the
    outline, where a strict test drops them at random."""
    points = list(points)
    if not points:
        return 0.0
    return float(
        np.mean(
            [
                mask[
                    max(int(p[1]) - 8, 0) : int(p[1]) + 9, max(int(p[0]) - 8, 0) : int(p[0]) + 9
                ].any()
                for p in points
            ]
        )
    )


def figure_mask(model, processor, image, embeddings, figure: dict, others: list[dict]):
    """Cut one figure out of a busy ground.

    A box alone is enough on blank paper; against a hatched landscape SAM takes the whole
    box. So the prompt also carries the figure's own body points (inside), the other
    figures' points that fall in its box (outside), and a grid of points in the box far from
    its skeleton (outside: likely ground). The figure is cut whole, and also as an upper and
    a lower body joined afterwards -- legs in dark breeches are often in no whole cut.
    Returns every cut with the share of body points it covers and the share of its area far
    from the skeleton.
    """
    # Only joints that are seen guide the cut: a hidden joint lies on whatever hides it,
    # and telling SAM "keep this" there would pull the occluder into the figure.
    hidden = set(figure.get("occluded", []))
    k = {n: p for n, p in (figure.get("keypoints") or {}).items() if p and n not in hidden}
    h, w = image.size[1], image.size[0]
    distance = skeleton_distance((h, w), k)
    torso = torso_length(k)
    x0, y0, x1, y1 = figure["box"]
    negatives = [
        list(map(float, p))
        for other in others
        for p in (other.get("keypoints") or {}).values()
        if p and x0 <= p[0] <= x1 and y0 <= p[1] <= y1
    ]
    for gx in np.linspace(x0, x1, 7)[1:-1]:
        for gy in np.linspace(y0, y1, 7)[1:-1]:
            if k and distance[min(int(gy), h - 1), min(int(gx), w - 1)] > 0.7 * torso:
                negatives.append([float(gx), float(gy)])

    def cut(positives: list, box: list[float]) -> list[np.ndarray]:
        points = [list(map(float, p)) for p in positives] + negatives
        prompt = {"input_boxes": [[list(map(float, box))]]}
        if points:
            labels = [1] * len(positives) + [0] * len(negatives)
            prompt |= {"input_points": [[points]], "input_labels": [[labels]]}
        masks = masks_for(model, processor, image, embeddings, **prompt)
        return [m for m in (in_box(m, figure["box"]) for m in masks) if m.any()]

    def scored(mask: np.ndarray) -> tuple:
        stray = (mask & (distance > 0.6 * torso)).sum() / mask.sum()
        return (mask, coverage(mask, k.values()), float(stray))

    candidates = [scored(m) for m in cut(list(k.values()), figure["box"])]
    parts = []
    for names in (UPPER, LOWER):
        pts = [k[n] for n in names if n in k]
        if len(pts) < 2:
            continue
        pad = 0.4 * torso
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        box = [
            max(min(xs) - pad, x0),
            max(min(ys) - pad, y0),
            min(max(xs) + pad, x1),
            min(max(ys) + pad, y1),
        ]
        part = [
            (m, coverage(m, pts), float((m & (distance > 0.6 * torso)).sum() / m.sum()))
            for m in cut(pts, box)
        ]
        if part:
            parts.append(choose(part))
    if len(parts) == 2:
        candidates.append(scored(parts[0] | parts[1]))
    return candidates


def choose(candidates: list[tuple]) -> np.ndarray | None:
    """The body first, then tidiness: among the cuts that cover (nearly) as many body points
    as the best one, the one with the least area far from the skeleton. Trading coverage
    against stray area in one score kept tidy half-figures (children without legs)."""
    return rank(candidates)[0] if candidates else None


def rank(candidates: list[tuple]) -> list[np.ndarray]:
    """All distinct cuts, ``choose``'s pick first, then by coverage. Near-duplicates (IoU
    above 0.9) are dropped, so each alternative a person flips through actually differs."""
    if not candidates:
        return []
    top = max(c[1] for c in candidates)
    first = min((c for c in candidates if c[1] >= top - 0.05), key=lambda c: c[2])
    rest = sorted((c for c in candidates if c is not first), key=lambda c: (-c[1], c[2]))
    kept: list[np.ndarray] = []
    for mask, _, _ in [first, *rest]:
        if all((mask & k).sum() / max((mask | k).sum(), 1) < 0.9 for k in kept):
            kept.append(mask)
    return kept


def segment(image: Image.Image, description: dict, models: list, detail: float) -> dict:
    """Outline every figure and object, keeping every figure's distinct cuts, best first.

    Neither model wins everywhere: full SAM separates a figure hidden behind another
    (E014801) and a figure on a cluttered ground (E014782) where SlimSAM merges them, but
    loses half a skeleton (E014784) or a child's legs (E003797) that SlimSAM keeps, and
    cutting upper and lower body apart recovers dark legs while losing some upper bodies. No
    rule tried picks the right cut on all twenty pictures, so each figure keeps its distinct
    cuts as ``{"cuts": [...]}``, ranked by ``rank``; the viewer lets a person pick another.
    """
    figures = description.get("figures", [])
    candidates: dict[str, list] = {f["id"]: [] for f in figures}
    for model, processor in models:
        with torch.no_grad():
            embeddings = model.get_image_embeddings(
                processor(image, return_tensors="pt")["pixel_values"]
            )
        for f in figures:
            candidates[f["id"]] += figure_mask(
                model, processor, image, embeddings, f, [o for o in figures if o is not f]
            )
    polygons = {}
    for f in figures:
        cuts = [outline(m, detail) for m in rank(candidates[f["id"]])]
        polygons[f["id"]] = {"cuts": [c for c in cuts if c]}
    # Objects are compact and either model cuts them alike: the last model's cut is kept.
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


WEIGHTS = [Path("data/models/slimsam-77"), Path("data/models/sam-vit-base")]
_models: dict[tuple, list] = {}


def load_models(weights=WEIGHTS) -> list:
    """Both SAM models, loaded once per process (the viewer's server reuses them)."""
    key = tuple(str(w) for w in weights)
    if key not in _models:
        _models[key] = [
            (SamModel.from_pretrained(w).eval(), SamProcessor.from_pretrained(w)) for w in weights
        ]
    return _models[key]


def cut_picture(
    name: str,
    description: dict,
    corrections: dict | None,
    *,
    models=None,
    images=Path("data/raw/emblematica/full"),
    detail: float = 0.004,
) -> dict:
    """Outline one picture's figures and objects from its description as corrected."""
    image = Image.open(images / f"{name}.jpg").convert("RGB")
    polygons = segment(
        image, apply_corrections(description, corrections), models or load_models(), detail
    )
    (PILOTE / "masks" / f"{name}.json").write_text(json.dumps(polygons))
    return polygons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("data/raw/emblematica/full"))
    parser.add_argument("--weights", type=Path, nargs="+", default=WEIGHTS)
    parser.add_argument(
        "--detail", type=float, default=0.004, help="contour simplification, share of perimeter"
    )
    parser.add_argument("--resegment", action="store_true")
    args = parser.parse_args()

    (PILOTE / "masks").mkdir(parents=True, exist_ok=True)
    (PILOTE / "records").mkdir(parents=True, exist_ok=True)
    index = []
    for desc_path in sorted((PILOTE / "describe").glob("*.json")):
        name = desc_path.stem
        description = json.loads(desc_path.read_text())
        mask_path = PILOTE / "masks" / f"{name}.json"
        corr_path = PILOTE / "corrections" / f"{name}.json"
        corrections = json.loads(corr_path.read_text()) if corr_path.exists() else None
        if args.resegment or not mask_path.exists():
            cut_picture(
                name,
                description,
                corrections,
                models=load_models(args.weights),
                images=args.images,
                detail=args.detail,
            )
        polygons = json.loads(mask_path.read_text())
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
