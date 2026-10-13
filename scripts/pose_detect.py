#!/usr/bin/env python3
"""Superseded by the workshop (``scripts/atelier.py``), which reproduces its numbers exactly;
kept to rebuild ``results/pose-detect``.

Find the figures without Opus: Grounding DINO boxes, then ViTPose points in them.

``scripts/pose_vitpose.py`` showed that ViTPose (Human-Art) places points as well as
Opus inside Opus's boxes. This removes Opus altogether: Grounding DINO (base, zero-shot)
is asked for the figures these pictures hold -- the prompt, fixed before any result,
is "a person. a skeleton. an angel. a child." -- boxes above 0.3 are kept after
non-maximum suppression, and ViTPose places the points in each. Both pose tests are
then rerun unchanged.

Run under the heavy-jobs queue: ``~/outils-seg/lourd.sh uv run python
scripts/pose_detect.py``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).parent))
import pose_copies
import pose_subjects
import pose_vitpose

PROMPT = "a person. a skeleton. an angel. a child."
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
NMS_IOU = 0.5


def load_detector(device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    repo = "IDEA-Research/grounding-dino-base"
    return (
        AutoModelForZeroShotObjectDetection.from_pretrained(repo).to(device).eval(),
        AutoProcessor.from_pretrained(repo),
    )


def detect(image: Image.Image, detector, processor, device: str) -> list[list[float]]:
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detector(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes, scores = result["boxes"].cpu(), result["scores"].cpu()
    if not len(boxes):
        return []
    keep = nms(boxes, scores, NMS_IOU)
    return [[round(v, 1) for v in boxes[i].tolist()] for i in keep]


def describe(root: Path, items: list[dict], models) -> list[dict]:
    detector, det_processor, pose_model, pose_processor, device = models
    (root / "detected").mkdir(exist_ok=True)
    out = []
    for it in items:
        cache = root / "detected" / f"{it['id']}.json"
        image = Image.open(root / "images" / f"{it['id']}.jpg").convert("RGB")
        if cache.exists():
            d = json.loads(cache.read_text())
        else:
            start = time.time()
            boxes = detect(image, detector, det_processor, device)
            points = pose_vitpose.estimate(image, boxes, pose_model, pose_processor, device)
            d = {
                "figures": [{"box": b, "keypoints": k} for b, k in zip(boxes, points, strict=True)],
                "seconds": round(time.time() - start, 2),
            }
            cache.write_text(json.dumps(d))
        d["size"] = image.size
        out.append(d)
    return out


def main() -> None:
    pose_model, pose_processor, device = pose_vitpose.load()
    detector, det_processor = load_detector(device)
    models = (detector, det_processor, pose_model, pose_processor, device)
    report = {"prompt": PROMPT, "box_threshold": BOX_THRESHOLD}

    copies = json.loads((pose_copies.COPIES / "manifest.json").read_text())
    scenes = [m["scene"] for m in copies]
    descriptions = describe(pose_copies.COPIES, copies, models)
    report["copies"] = pose_copies.score(pose_copies.pose_matrix(descriptions), scenes)
    seconds = [d["seconds"] for d in descriptions]
    figures = [len(d["figures"]) for d in descriptions]

    subjects = pose_subjects.manifest()
    descriptions = describe(pose_subjects.SUBJECTS, subjects, models)
    dist = pose_copies.pose_matrix(descriptions)
    for query in ("emblem", "museum"):
        report[f"subjects_{query}_queries"] = pose_subjects.cross_score(dist, subjects, query)
    seconds += [d["seconds"] for d in descriptions]
    figures += [len(d["figures"]) for d in descriptions]
    report["seconds_per_picture_median"] = sorted(seconds)[len(seconds) // 2]
    report["figures_per_picture"] = {
        "median": sorted(figures)[len(figures) // 2],
        "none": sum(f == 0 for f in figures),
    }
    Path("results/pose-detect").mkdir(parents=True, exist_ok=True)
    Path("results/pose-detect/report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
