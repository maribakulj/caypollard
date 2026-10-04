#!/usr/bin/env python3
"""Does a figure's pose find the copies of a scene that pixels miss?

Holbein's Dance of Death was copied for three centuries: the woodcuts of 1526, the
*Icones mortis* of 1648 (the pilot's ten), Bewick's wood engravings of 1789, Mechel's
etchings of 1803, Hollar's of 1816. One scene keeps its composition across them, while the
hand, the medium and often the side (copies are mirrored) change. So the right answers are
known in advance: a query should find the other editions of its scene before any other
scene. ``data/derived/copies/manifest.json`` lists the pictures and their scene.

Every picture, the pilot's included, gets the same single pass of Opus
(``scripts/pilot_describe.py``), with no review and no person's correction: the scalable
setting, not the curated one.

The two channels, fixed before any result was seen:

- **pose**: each figure's skeleton normalised to its hips, torso up, unit torso
  (``normalised_skeleton``); two figures differ by the mean distance between their joints,
  joints present in one only counting as a full miss; two pictures by the best matching of
  their figures (Hungarian), each pair also paying for the gap between where the two
  figures stand in their frame; an unmatched figure costs a full miss. A picture is also
  compared mirrored (x reversed, left and right swapped) and the closer reading is kept.
- **pixels**: DINOv2-base CLS embeddings (the viewer's pixel channel), cosine, the query
  also compared mirrored and the closer reading kept.

Run: ``uv run python scripts/pose_copies.py describe`` (Opus, ~1 min a picture), then
``uv run python scripts/pose_copies.py evaluate``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment

from caypollard.figures import KEYPOINTS, normalised_skeleton

COPIES = Path("data/derived/copies")
MISS = 1.0  # a joint, or a figure, that has no counterpart
LAYOUT = 0.5  # weight of where a figure stands in its frame, against its pose


def mirrored_keypoints(k: dict, width: float) -> dict:
    """The skeleton seen in a mirror: x reversed, the figure's left and right swapped."""
    out = {}
    for name, p in k.items():
        other = (
            name.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")
            if name.startswith(("left_", "right_"))
            else name
        )
        out[other] = None if not p else [width - p[0], p[1]]
    return out


def figures_of(description: dict, mirror: bool) -> list[dict]:
    w, h = description["size"]
    out = []
    for f in description.get("figures", []):
        k = {n: p for n, p in (f.get("keypoints") or {}).items() if n in KEYPOINTS and p}
        box = f.get("box") or [0, 0, w, h]
        if mirror:
            k = mirrored_keypoints(k, w)
            box = [w - box[2], box[1], w - box[0], box[3]]
        out.append(
            {
                "skeleton": normalised_skeleton(k),
                "centre": ((box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h),
                "height": (box[3] - box[1]) / h,
            }
        )
    return out


def figure_distance(a: dict, b: dict) -> float:
    sa, sb = a["skeleton"], b["skeleton"]
    names = set(sa) | set(sb)
    if not names:
        return MISS
    gaps = [min(math.dist(sa[n], sb[n]), 1.5) if n in sa and n in sb else MISS for n in names]
    pose = sum(gaps) / len(gaps)
    layout = math.dist(a["centre"], b["centre"]) + abs(a["height"] - b["height"])
    return pose + LAYOUT * layout


def scene_distance(fa: list[dict], fb: list[dict]) -> float:
    if not fa or not fb:
        return MISS * 2
    cost = np.array([[figure_distance(a, b) for b in fb] for a in fa])
    rows, cols = linear_sum_assignment(cost)
    unmatched = len(fa) + len(fb) - 2 * len(rows)
    return float(cost[rows, cols].sum() + MISS * unmatched) / max(len(fa), len(fb))


def pose_matrix(descriptions: list[dict]) -> np.ndarray:
    plain = [figures_of(d, False) for d in descriptions]
    flipped = [figures_of(d, True) for d in descriptions]
    n = len(descriptions)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = min(
                scene_distance(plain[i], plain[j]), scene_distance(flipped[i], plain[j])
            )
    return out


def pixel_matrix(paths: list[Path]) -> np.ndarray:
    from caypollard.vision.encoders import DEFAULT_MODELS, HuggingFaceVisionEncoder

    encoder = HuggingFaceVisionEncoder(DEFAULT_MODELS["dinov2-base"], device="auto")
    images = [Image.open(p).convert("RGB") for p in paths]
    plain = encoder.encode(images)
    flipped = encoder.encode([ImageOps.mirror(im) for im in images])
    return -np.maximum(flipped @ plain.T, plain @ plain.T)  # a distance: lower is closer


def score(dist: np.ndarray, scenes: list[str]) -> dict:
    """Per query: is the nearest other picture the same scene; average precision."""
    n = len(scenes)
    top1, aps, firsts = [], [], []
    for i in range(n):
        if sum(s == scenes[i] for s in scenes) < 2:
            continue
        order = [j for j in np.argsort(dist[i], kind="stable") if j != i]
        hits = [scenes[j] == scenes[i] for j in order]
        top1.append(hits[0])
        firsts.append(hits.index(True) + 1)
        found, precisions = 0, []
        for rank, hit in enumerate(hits, 1):
            if hit:
                found += 1
                precisions.append(found / rank)
        aps.append(sum(precisions) / len(precisions))
    return {
        "queries": len(top1),
        "top1": round(float(np.mean(top1)), 3),
        "mAP": round(float(np.mean(aps)), 3),
        "median_first_hit": float(np.median(firsts)),
    }


def chance(scenes: list[str]) -> dict:
    """What a random ranking scores, on average over many shuffles."""
    rng = np.random.default_rng(0)
    runs = [score(rng.random((len(scenes), len(scenes))), scenes) for _ in range(500)]
    return {k: round(float(np.mean([r[k] for r in runs])), 3) for k in ("top1", "mAP")}


def describe_all(model: str, timeout: float, part: int, parts: int) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from pilot_describe import describe

    manifest = json.loads((COPIES / "manifest.json").read_text())
    (COPIES / "describe").mkdir(parents=True, exist_ok=True)
    for item in manifest[part::parts]:
        target = COPIES / "describe" / f"{item['id']}.json"
        if target.exists():
            continue
        data = describe(COPIES / "images" / f"{item['id']}.jpg", model, timeout)
        if str(data.get("_raw", "")).startswith("API Error"):
            print(f"{item['id']}: not described — {data['_raw'][:80]}", flush=True)
            continue
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        print(f"{item['id']}: {len(data.get('figures', []))} figures", flush=True)


def evaluate() -> None:
    manifest = json.loads((COPIES / "manifest.json").read_text())
    items = [m for m in manifest if (COPIES / "describe" / f"{m['id']}.json").exists()]
    descriptions = []
    for m in items:
        d = json.loads((COPIES / "describe" / f"{m['id']}.json").read_text())
        d["size"] = Image.open(COPIES / "images" / f"{m['id']}.jpg").size
        descriptions.append(d)
    scenes = [m["scene"] for m in items]
    matrices = {
        "pose": pose_matrix(descriptions),
        "pixels": pixel_matrix([COPIES / "images" / f"{m['id']}.jpg" for m in items]),
    }
    report = {"pictures": len(items), "scenes": len(set(scenes)), "chance": chance(scenes)}
    for name, dist in matrices.items():
        report[name] = score(dist, scenes)
        # Per query: the nearest other picture, to look at rather than to average.
        nearest = {}
        for i in range(len(items)):
            order = [j for j in np.argsort(dist[i], kind="stable") if j != i]
            nearest[items[i]["id"]] = [items[j]["id"] for j in order[:5]]
        report[f"{name}_nearest"] = nearest
        np.save(COPIES / f"{name}-distances.npy", dist)
    (COPIES / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if "nearest" not in k}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["describe", "evaluate"])
    parser.add_argument("--model", default="opus")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--part", default="0/1", help="i/n: describe every n-th picture from i")
    args = parser.parse_args()
    if args.step == "describe":
        part, parts = map(int, args.part.split("/"))
        describe_all(args.model, args.timeout, part, parts)
    else:
        evaluate()


if __name__ == "__main__":
    main()
