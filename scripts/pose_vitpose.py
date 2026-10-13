#!/usr/bin/env python3
"""Superseded by the workshop (``scripts/atelier.py``), which reproduces its numbers exactly;
kept to rebuild ``results/pose-vitpose``.

Replace Opus's body points by ViTPose-B trained on Human-Art, and rerun both pose tests.

Opus places the points well but at about a minute a picture, which does not scale. A
pose estimator trained on depicted people (Human-Art: paintings, sculpture, drawings)
is the standard alternative: ViTPose-B reaches 0.759 AP on Human-Art with given boxes,
against 0.555 when trained on photographs only (IDEA-Research/HumanArt). Its mmpose
checkpoint is converted here to the ``transformers`` implementation, with no mmpose.

Boxes: each figure's box from the Opus description, so that this measures the pose
estimator alone (the "given boxes" setting); finding the figures is a separate step.
ViTPose gives the 17 COCO points: nose, shoulders, elbows, wrists, hips, knees, ankles
are kept (eyes and ears have no counterpart in the pilot). As a control, Opus's points
are scored again restricted to the same joints, so that a difference between the two
comes from the estimator and not from the joint set.

Run: ``uv run python scripts/pose_vitpose.py`` (local, ~1 s a figure on mps).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import pose_copies
import pose_subjects

CHECKPOINT = Path("data/models/vitpose/vitpose-b-humanart.pth")
COCO = (
    "nose",
    None,
    None,
    None,
    None,
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
KEPT = {n for n in COCO if n}
MIN_SCORE = 0.3


def convert(mm: dict) -> dict:
    """mmpose ViTPose (classic decoder) names to transformers' VitPose names."""
    out = {
        "backbone.embeddings.position_embeddings": mm["backbone.pos_embed"],
        "backbone.embeddings.patch_embeddings.projection.weight": mm[
            "backbone.patch_embed.projection.weight"
        ],
        "backbone.embeddings.patch_embeddings.projection.bias": mm[
            "backbone.patch_embed.projection.bias"
        ],
        "backbone.layernorm.weight": mm["backbone.ln1.weight"],
        "backbone.layernorm.bias": mm["backbone.ln1.bias"],
    }
    for i in range(12):
        src, dst = f"backbone.layers.{i}.", f"backbone.encoder.layer.{i}."
        for part in ("weight", "bias"):
            q, k, v = mm[src + f"attn.qkv.{part}"].chunk(3, dim=0)
            out[dst + f"attention.attention.query.{part}"] = q
            out[dst + f"attention.attention.key.{part}"] = k
            out[dst + f"attention.attention.value.{part}"] = v
            out[dst + f"attention.output.dense.{part}"] = mm[src + f"attn.proj.{part}"]
            out[dst + f"layernorm_before.{part}"] = mm[src + f"ln1.{part}"]
            out[dst + f"layernorm_after.{part}"] = mm[src + f"ln2.{part}"]
            out[dst + f"mlp.fc1.{part}"] = mm[src + f"ffn.layers.0.0.{part}"]
            out[dst + f"mlp.fc2.{part}"] = mm[src + f"ffn.layers.1.{part}"]
    out["head.deconv1.weight"] = mm["head.deconv_layers.0.weight"]
    out["head.deconv2.weight"] = mm["head.deconv_layers.3.weight"]
    for src, dst in (("1", "batchnorm1"), ("4", "batchnorm2")):
        for part in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            out[f"head.{dst}.{part}"] = mm[f"head.deconv_layers.{src}.{part}"]
    out["head.conv.weight"] = mm["head.final_layer.weight"]
    out["head.conv.bias"] = mm["head.final_layer.bias"]
    return out


def load():
    from huggingface_hub import hf_hub_download
    from transformers import VitPoseConfig, VitPoseForPoseEstimation, VitPoseImageProcessor

    repo = "usyd-community/vitpose-base-simple"
    config = VitPoseConfig.from_json_file(hf_hub_download(repo, "config.json"))
    config.use_simple_decoder = False
    model = VitPoseForPoseEstimation(config)
    mm = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(convert(mm), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device).eval(), VitPoseImageProcessor.from_pretrained(repo), device


def estimate(image: Image.Image, boxes: list, model, processor, device) -> list[dict]:
    """Points for each box (x0, y0, x1, y1), ours names, low-confidence points dropped."""
    if not boxes:
        return []
    coco_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    inputs = processor(image, boxes=[coco_boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])[0]
    out = []
    for res in results:
        k = {}
        for name, p, s in zip(COCO, res["keypoints"].tolist(), res["scores"].tolist(), strict=True):
            if name and s >= MIN_SCORE:
                k[name] = [round(p[0], 1), round(p[1], 1)]
        out.append(k)
    return out


def describe(root: Path, items: list[dict], model, processor, device) -> list[dict]:
    """ViTPose descriptions, cached under ``root / 'vitpose'``, in the Opus format."""
    (root / "vitpose").mkdir(exist_ok=True)
    out = []
    for it in items:
        cache = root / "vitpose" / f"{it['id']}.json"
        image = Image.open(root / "images" / f"{it['id']}.jpg").convert("RGB")
        if cache.exists():
            d = json.loads(cache.read_text())
        else:
            opus = json.loads((root / "describe" / f"{it['id']}.json").read_text())
            boxes = [f["box"] for f in opus.get("figures", []) if f.get("box")]
            points = estimate(image, boxes, model, processor, device)
            d = {
                "figures": [{"box": b, "keypoints": k} for b, k in zip(boxes, points, strict=True)]
            }
            cache.write_text(json.dumps(d))
        d["size"] = image.size
        out.append(d)
    return out


def opus(root: Path, items: list[dict], joints: set | None) -> list[dict]:
    out = []
    for it in items:
        d = json.loads((root / "describe" / f"{it['id']}.json").read_text())
        if joints is not None:
            for f in d.get("figures", []):
                f["keypoints"] = {
                    n: p for n, p in (f.get("keypoints") or {}).items() if n in joints
                }
        d["size"] = Image.open(root / "images" / f"{it['id']}.jpg").size
        out.append(d)
    return out


def main() -> None:
    model, processor, device = load()
    report = {}

    copies = json.loads((pose_copies.COPIES / "manifest.json").read_text())
    scenes = [m["scene"] for m in copies]
    block = {"chance": pose_copies.chance(scenes)}
    for name, descriptions in (
        ("opus", opus(pose_copies.COPIES, copies, None)),
        ("opus_same_joints", opus(pose_copies.COPIES, copies, KEPT)),
        ("vitpose_humanart", describe(pose_copies.COPIES, copies, model, processor, device)),
    ):
        block[name] = pose_copies.score(pose_copies.pose_matrix(descriptions), scenes)
    report["copies"] = block

    subjects = pose_subjects.manifest()
    for name, descriptions in (
        ("opus", opus(pose_subjects.SUBJECTS, subjects, None)),
        ("opus_same_joints", opus(pose_subjects.SUBJECTS, subjects, KEPT)),
        ("vitpose_humanart", describe(pose_subjects.SUBJECTS, subjects, model, processor, device)),
    ):
        dist = pose_copies.pose_matrix(descriptions)
        for query in ("emblem", "museum"):
            report.setdefault(f"subjects_{query}_queries", {})[name] = pose_subjects.cross_score(
                dist, subjects, query
            )
    for query in ("emblem", "museum"):
        report[f"subjects_{query}_queries"]["chance"] = pose_subjects.chance(subjects, query)
    found = [
        len(f["keypoints"])
        for root in (pose_copies.COPIES, pose_subjects.SUBJECTS)
        for p in (root / "vitpose").glob("*.json")
        for f in json.loads(p.read_text())["figures"]
    ]
    report["vitpose_points_per_figure"] = {
        "figures": len(found),
        "median": float(np.median(found)),
        "under_6": int(sum(n < 6 for n in found)),
    }
    Path("results/pose-vitpose").mkdir(parents=True, exist_ok=True)
    Path("results/pose-vitpose/report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
