#!/usr/bin/env python3
"""Place each head as three points that can be seen: top of the skull, chin, tip of the nose.

The first description used the photo convention for the head (nose, eyes, ears); on an
engraving the eyes are a few pixels apart and the ears under hair, and the points fell
almost at random. This pass shows Claude (Opus) the existing figures and asks only for the
three head points; the rest of the skeleton is not touched. Merged into
``data/derived/pilote/describe/<id>.json``: ``crown``, ``chin`` and ``nose`` replace the old
head points, which are kept under ``_photo_head``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PROMPT = """Read the image file {path} with the Read tool. It is an engraving of {w}x{h} \
pixels (x to the right, y downward, origin at the top-left).

These figures were already described (box and body points):

{figures}

For each figure's head give three points, in pixel coordinates of that image:
  "crown": the top of the skull (where the head's axis leaves the top of the head, under \
any hat, hair or halo);
  "chin": the bottom of the chin (or of the jaw, for a skull);
  "nose": the tip of the nose (for a skull, the nasal opening).
Crown and chin together give the axis of the head, so place them on that axis even when \
the head is bowed, raised or turned. If a point is hidden but can be inferred, give it and \
list its name in "occluded". Use null only if the head is not in the picture.

Answer ONLY with JSON: {{"figures": [{{"id": "F1", "crown": [x, y], "chin": [x, y], \
"nose": [x, y], "occluded": []}}]}}"""

HEAD = ("crown", "chin", "nose")
PHOTO_HEAD = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def ask(path: Path, description: dict, model: str, timeout: float) -> dict:
    w, h = description["_size"]
    figures = json.dumps(
        [
            {"id": f["id"], "name": f.get("name"), "box": f["box"], "keypoints": f["keypoints"]}
            for f in description.get("figures", [])
        ],
        ensure_ascii=False,
    )
    run = subprocess.run(
        [
            "claude",
            "-p",
            "--model",
            model,
            "--allowedTools",
            "Read",
            "--output-format",
            "json",
            PROMPT.format(path=path.resolve(), w=w, h=h, figures=figures),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    text = str(json.loads(run.stdout).get("result", ""))
    start = text.find("{")
    try:
        answer, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else ({}, 0)
    except json.JSONDecodeError:
        answer = {}
    return answer


def merge(description: dict, answer: dict) -> int:
    placed = 0
    by_id = {f["id"]: f for f in answer.get("figures", []) if isinstance(f, dict)}
    for f in description.get("figures", []):
        got = by_id.get(f["id"])
        if got is None:
            continue
        k = f.setdefault("keypoints", {})
        f["_photo_head"] = {n: k.pop(n, None) for n in PHOTO_HEAD}
        occluded = set(f.get("occluded", [])) - set(PHOTO_HEAD)
        for name in HEAD:
            p = got.get(name)
            ok = isinstance(p, list) and len(p) == 2
            k[name] = [round(float(p[0])), round(float(p[1]))] if ok else None
            placed += ok
            if ok and name in got.get("occluded", []):
                occluded.add(name)
        f["occluded"] = sorted(occluded)
    return placed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("data/raw/emblematica/full"))
    parser.add_argument("--describe", type=Path, default=Path("data/derived/pilote/describe"))
    parser.add_argument("--model", default="opus")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    for path in sorted(args.describe.glob("*.json")):
        description = json.loads(path.read_text())
        if "_head_pass" in description:
            continue
        answer = ask(args.images / f"{path.stem}.jpg", description, args.model, args.timeout)
        placed = merge(description, answer)
        description["_head_pass"] = {"model": args.model, "raw": answer}
        path.write_text(json.dumps(description, ensure_ascii=False, indent=1))
        print(f"{path.stem}: {placed} head points placed", flush=True)


if __name__ == "__main__":
    main()
