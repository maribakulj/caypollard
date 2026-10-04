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


def outline(mask: np.ndarray, detail: float) -> list[list[list[int]]] | None:
    """The mask's outer contours as rings, simplified. Every piece at least 2% the area of
    the largest is kept: a figure crossed by another's arm or a sword falls into pieces, and
    keeping only the largest threw its legs away (E014784)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(cv2.contourArea(c) for c in contours)
    out = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 0.02 * largest:
            continue
        simple = cv2.approxPolyDP(contour, detail * cv2.arcLength(contour, True), True)
        if len(simple) >= 3:
            out.append(simple[:, 0, :].tolist())
    return out or None


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
    "crown",
    "chin",
    "nose",
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
    Returns every cut with the share of the body it covers (joints, and points along the
    bones) and its stray share: area far from its skeleton, or nearer another figure's seen
    skeleton than its own.
    """
    # Only joints that are seen guide the cut: a hidden joint lies on whatever hides it,
    # and telling SAM "keep this" there would pull the occluder into the figure.
    hidden = set(figure.get("occluded", []))
    k = {n: p for n, p in (figure.get("keypoints") or {}).items() if p and n not in hidden}
    h, w = image.size[1], image.size[0]
    distance = skeleton_distance((h, w), k)
    torso = torso_length(k)
    # A pixel nearer another figure's seen skeleton than this one's belongs to that figure.
    nearest_other = np.full((h, w), np.inf, np.float32)
    for other in others:
        seen = {
            n: p
            for n, p in (other.get("keypoints") or {}).items()
            if p and n not in set(other.get("occluded", []))
        }
        if seen:
            nearest_other = np.minimum(nearest_other, skeleton_distance((h, w), seen))
    elsewhere = (distance > 0.6 * torso) | (nearest_other < distance)
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

    # Points along every bone, hidden joints included: a cut must hold the body between its
    # joints too. Covering the visible joints alone let a cut keep head, sleeves and one
    # stocking and drop the robe in between (E014784, whose hips are under that robe).
    every = {n: p for n, p in (figure.get("keypoints") or {}).items() if p}
    along = []
    for a, b in BONES:
        if a in every and b in every:
            along += [
                (
                    every[a][0] + (every[b][0] - every[a][0]) * t,
                    every[a][1] + (every[b][1] - every[a][1]) * t,
                )
                for t in np.linspace(0.1, 0.9, 5)
            ]

    def scored(mask: np.ndarray) -> tuple:
        stray = (mask & elsewhere).sum() / mask.sum()
        body = coverage(mask, k.values())
        if along:
            body = (body + coverage(mask, along)) / 2
        return (mask, body, float(stray))

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
            (m, coverage(m, pts), float((m & elsewhere).sum() / m.sum())) for m in cut(pts, box)
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


def complete_limbs(
    mask: np.ndarray,
    figure: dict,
    others_masks: list,
    others_points: list,
    model,
    processor,
    image,
    embeddings,
) -> np.ndarray:
    """Add the thin limbs a whole-figure cut misses: a forearm reaching across, a shin
    between another figure's legs (E014784). For each bone of the skeleton that the mask
    covers poorly, SAM is asked for that bone alone -- a box around it, points along it --
    and what it returns within a narrow band around the bone is added. Nothing is added
    inside another figure's silhouette, so a bone passing behind someone stays hidden."""
    k = {n: p for n, p in (figure.get("keypoints") or {}).items() if p}
    torso = torso_length(k)
    h, w = mask.shape
    blocked = np.zeros_like(mask)
    for m in others_masks:
        blocked |= m
    out = mask.copy()
    for a, b in BONES:
        if a not in k or b not in k:
            continue
        samples = [
            (k[a][0] + (k[b][0] - k[a][0]) * t, k[a][1] + (k[b][1] - k[a][1]) * t)
            for t in np.linspace(0.15, 0.85, 5)
        ]
        visible = [p for p in samples if not blocked[min(int(p[1]), h - 1), min(int(p[0]), w - 1)]]
        if len(visible) < 2 or coverage(out, visible) >= 0.6:
            continue
        pad = 0.25 * torso
        xs, ys = [k[a][0], k[b][0]], [k[a][1], k[b][1]]
        box = [
            max(min(xs) - pad, 0),
            max(min(ys) - pad, 0),
            min(max(xs) + pad, w),
            min(max(ys) + pad, h),
        ]
        negatives = [
            list(map(float, p))
            for p in others_points
            if box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]
        ]
        points = [list(map(float, p)) for p in visible] + negatives
        labels = [1] * len(visible) + [0] * len(negatives)
        proposals = masks_for(
            model,
            processor,
            image,
            embeddings,
            input_boxes=[[box]],
            input_points=[[points]],
            input_labels=[[labels]],
        )
        best = max(proposals, key=lambda m: coverage(m, visible) - m.sum() / (h * w))
        band = skeleton_distance((h, w), {a: k[a], b: k[b]}) <= 0.25 * torso
        # A limb lies mostly within the band and is thinner than it; a proposal that spills
        # far beyond the band, or fills it, is a robe or a whole body (E003840, E014786,
        # E014810), and clipping it leaves a straight strip.
        inside = (best & band).sum()
        if inside < 0.5 * best.sum() or inside > 0.6 * band.sum():
            continue
        out |= best & band & ~blocked
    return out


def segment(image: Image.Image, description: dict, models: list, detail: float) -> dict:
    """Outline every figure and object, keeping every figure's distinct cuts, best first.

    Neither model wins everywhere: full SAM separates a figure hidden behind another
    (E014801) and a figure on a cluttered ground (E014782) where SlimSAM merges them, but
    loses half a skeleton (E014784) or a child's legs (E003797) that SlimSAM keeps, and
    cutting upper and lower body apart recovers dark legs while losing some upper bodies;
    thin limbs still missed are completed bone by bone (``complete_limbs``). No
    rule tried picks the right cut on all twenty pictures, so each figure keeps its distinct
    cuts as ``{"cuts": [...]}``, ranked by ``rank``; the viewer lets a person pick another.
    """
    figures = description.get("figures", [])
    candidates: dict[str, list] = {f["id"]: [] for f in figures}
    model = processor = embeddings = None
    for model, processor in models:
        with torch.no_grad():
            embeddings = model.get_image_embeddings(
                processor(image, return_tensors="pt")["pixel_values"]
            )
        for f in figures:
            candidates[f["id"]] += figure_mask(
                model, processor, image, embeddings, f, [o for o in figures if o is not f]
            )
    ranked = {f["id"]: rank(candidates[f["id"]]) for f in figures}
    # Where two figures' cuts overlap, each pixel goes to the figure whose seen skeleton is
    # nearer. SAM cannot always part two bodies (E014782: every cut of the count spills over
    # Death), but Death's own cut is right, and the nearer skeleton settles the overlap.
    h, w = image.size[1], image.size[0]
    near = {}
    for f in figures:
        seen = {
            n: p
            for n, p in (f.get("keypoints") or {}).items()
            if p and n not in set(f.get("occluded", []))
        }
        near[f["id"]] = skeleton_distance((h, w), seen) if seen else np.full((h, w), np.inf)
    chosen = {fid: cuts[0] for fid, cuts in ranked.items() if cuts}
    seen_points = {
        f["id"]: [
            p
            for n, p in (f.get("keypoints") or {}).items()
            if p and n not in set(f.get("occluded", []))
        ]
        for f in figures
    }
    polygons = {}
    for f in figures:
        carved = []
        for mask in ranked[f["id"]]:
            for other, theirs in chosen.items():
                if other != f["id"]:
                    mask = mask & ~(theirs & (near[other] < near[f["id"]]))
            if mask.any():
                carved.append(mask)
        # Thin limbs are completed on the first cut only (the one shown), with the last
        # model loaded -- full SAM, the better of the two on thin bony shapes.
        if carved:
            others = [m for fid, m in chosen.items() if fid != f["id"]]
            points = [p for fid, pts in seen_points.items() if fid != f["id"] for p in pts]
            carved[0] = complete_limbs(
                carved[0], f, others, points, model, processor, image, embeddings
            )
        cuts = [outline(m, detail) for m in carved]
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
