#!/usr/bin/env python3
"""A pose channel, after Impett and Süsstrunk's reading of Warburg's Pathosformel.

Division 3 -- the human body and its actions -- is the second largest block in
the corpus, 1 913 cross-medium queries, and the shape record leaves it at median
rank 1 835 of 12 479. Composition takes it to 948 and the full record to 698, but
none of those channels knows what a body is.

Impett and Süsstrunk clustered a third of Warburg's Bilderatlas on the relative
angles of limbs alone and recovered pose groups corresponding to Pathosformeln.
Relative angles are exactly the right shape of feature here: they discard the
figure's size, its position in the frame, its orientation, and everything about
the support, and keep the one thing a print and a painting of the same gesture
share.

Angles are taken against the torso axis rather than against the vertical, so a
figure leaning or lying down is described by what its limbs do relative to its
own body rather than to the picture's edge.

The stop condition is measured, not assumed. A keypoint detector is trained on
photographs and inherits their statistics, which is the cross-depiction problem
applied to the instrument itself; if it finds figures in paintings and not in
engravings, the channel is measuring the medium and must be abandoned. The
detection rate per corpus is therefore the first thing this script reports.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import save_embedding_table
from caypollard.provenance import read_jsonl

# COCO keypoint order, as torchvision emits it.
NOSE, L_SHO, R_SHO = 0, 5, 6
L_ELB, R_ELB, L_WRI, R_WRI = 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

SEGMENTS = (
    ("bras gauche", L_SHO, L_ELB),
    ("avant-bras gauche", L_ELB, L_WRI),
    ("bras droit", R_SHO, R_ELB),
    ("avant-bras droit", R_ELB, R_WRI),
    ("cuisse gauche", L_HIP, L_KNE),
    ("jambe gauche", L_KNE, L_ANK),
    ("cuisse droite", R_HIP, R_KNE),
    ("jambe droite", R_KNE, R_ANK),
    ("tête", L_SHO, NOSE),
)


def pose_features(
    keypoints: np.ndarray, scores: np.ndarray, *, threshold: float
) -> np.ndarray | None:
    """Relative limb angles for one figure, or None if too little of it is visible."""
    visible = scores >= threshold
    if not (visible[L_SHO] and visible[R_SHO] and (visible[L_HIP] or visible[R_HIP])):
        return None
    shoulder = (keypoints[L_SHO] + keypoints[R_SHO]) / 2.0
    hips = [keypoints[i] for i in (L_HIP, R_HIP) if visible[i]]
    hip = np.mean(hips, axis=0)
    torso = shoulder - hip
    norm = float(np.hypot(*torso))
    if norm < 1e-6:
        return None
    reference = math.atan2(float(torso[1]), float(torso[0]))

    features: list[float] = []
    for _name, start, end in SEGMENTS:
        if not (visible[start] and visible[end]):
            # Absent is a state of its own, not a zero angle: the cosine and sine
            # go to zero and a presence flag says why.
            features.extend((0.0, 0.0, 0.0))
            continue
        vector = keypoints[end] - keypoints[start]
        angle = math.atan2(float(vector[1]), float(vector[0])) - reference
        features.extend((math.cos(angle), math.sin(angle), 1.0))
    # Shoulder width against torso length: a foreshortened or frontal figure.
    width = float(np.hypot(*(keypoints[L_SHO] - keypoints[R_SHO])))
    features.append(width / norm)
    return np.asarray(features, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--image-dir", default="/")
    parser.add_argument("--corpus", required=True, help="Name recorded in the detection report")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--score-threshold", type=float, default=0.75)
    parser.add_argument("--keypoint-threshold", type=float, default=3.0)
    parser.add_argument("--max-figures", type=int, default=6)
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Longest side fed to the detector. A 960x1280 tensor costs several times "
             "what the detector needs -- it resizes internally anyway -- and on MPS the "
             "allocations accumulate until the run is killed. Keypoints are read in "
             "relative angles, so the scale is irrelevant to the features.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Persist partial results every N images so a kill costs minutes, not hours.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-new",
        type=int,
        help="Stop after this many newly processed images and exit cleanly. A long "
             "run was killed three times for memory even with cache clearing; a short "
             "process returns everything to the system when it exits, so a shell loop "
             "of bounded runs finishes where one unbounded run does not.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights,
        keypointrcnn_resnet50_fpn,
    )
    from torchvision.transforms import functional

    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights).to(args.device).eval()

    records = read_jsonl(args.manifest)
    if args.limit:
        records = records[: args.limit]

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    with_figure = 0
    processed = 0
    figures_total = 0
    checkpoint = Path(str(args.output) + ".partial.npz")
    stopped_early = False
    done: set[str] = set()
    if args.resume and checkpoint.is_file():
        saved = np.load(checkpoint, allow_pickle=False)
        ids = [str(value) for value in saved["ids"]]
        vectors = list(saved["vectors"])
        done = set(ids)
        with_figure = len(ids)
        print(f"reprise : {len(ids)} images déjà posées", flush=True)

    def write_checkpoint() -> None:
        if not vectors:
            return
        staging = checkpoint.with_suffix(".tmp.npz")
        with staging.open("wb") as handle:
            np.savez(handle, ids=np.array(ids, dtype=np.str_), vectors=np.stack(vectors))
        staging.replace(checkpoint)

    for record in records:
        if str(record["id"]) in done:
            continue
        recorded = record.get("image_path")
        path = Path(recorded) if recorded else Path(args.image_dir) / str(record["filename"])
        if not path.is_file():
            continue
        if args.max_new and processed >= args.max_new:
            stopped_early = True
            break
        processed += 1
        try:
            image = Image.open(path).convert("RGB")
            if max(image.size) > args.max_side:
                scale = args.max_side / max(image.size)
                image = image.resize(
                    (max(int(image.width * scale), 1), max(int(image.height * scale), 1)),
                    Image.LANCZOS,
                )
            tensor = functional.to_tensor(image).to(args.device)
            with torch.inference_mode():
                output = model([tensor])[0]
            detections = [
                (
                    float(output["scores"][i]),
                    output["keypoints"][i].detach().cpu().numpy()[:, :2],
                    output["keypoints_scores"][i].detach().cpu().numpy(),
                )
                for i in range(len(output["scores"]))
            ]
            del output, tensor
        except Exception:
            continue

        if processed % 50 == 0 and args.device == "mps":
            # MPS does not return allocations on its own across many small graphs,
            # and a run of ten thousand images is killed long before it finishes.
            torch.mps.empty_cache()
        if args.checkpoint_every and processed % args.checkpoint_every == 0:
            write_checkpoint()

        kept = []
        for score, keypoints, confidence in detections:
            if score < args.score_threshold:
                continue
            features = pose_features(
                keypoints, confidence, threshold=args.keypoint_threshold
            )
            if features is not None:
                kept.append(features)
            if len(kept) >= args.max_figures:
                break
        if not kept:
            continue
        with_figure += 1
        figures_total += len(kept)
        stack = np.stack(kept)
        # A picture is described by the mean pose of its figures and by their
        # spread: one gesture repeated and two opposed gestures are different
        # pictures with the same mean.
        vector = np.concatenate(
            [stack.mean(axis=0), stack.std(axis=0), [math.log1p(len(kept))]]
        ).astype(np.float32)
        ids.append(str(record["id"]))
        vectors.append(vector)

    report = {
        "corpus": args.corpus,
        "images_processed": processed,
        "images_with_a_usable_figure": with_figure,
        "detection_rate": round(with_figure / max(processed, 1), 4),
        "figures_found": figures_total,
        "score_threshold": args.score_threshold,
        "keypoint_threshold": args.keypoint_threshold,
        "max_side": args.max_side,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))

    if stopped_early:
        # A bounded run has not finished the corpus, and writing the final table
        # would tell a caller it had. Only the checkpoint is updated, so the next
        # invocation resumes and the shell loop keeps going.
        write_checkpoint()
        print("interrompu par --max-new ; point de contrôle écrit", flush=True)
        return

    if ids:
        save_embedding_table(
            args.output,
            ids=tuple(ids),
            vectors=np.stack(vectors),
            metadata={
                "family": "pose",
                "method": "relative limb angles against the torso axis, after Impett",
                "detector": "keypointrcnn_resnet50_fpn",
                "detection_rate": report["detection_rate"],
            },
            normalize=True,
        )


if __name__ == "__main__":
    main()
