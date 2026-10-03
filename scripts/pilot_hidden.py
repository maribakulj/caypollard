#!/usr/bin/env python3
"""Add the joints a figure hides, without moving the ones it shows.

The first description asked for null on hidden joints, so a leg behind another figure
lost its knee and ankle: the skeleton -- which, unlike the silhouette, is meant to be the
whole body -- had a hole where the engraving implies a limb. This pass shows Claude
(Opus) the existing skeletons and asks only for the joints that are hidden but can be
inferred, marked as such. Visible points are not touched, so corrections made on them
stay valid. The answer is merged into ``data/derived/pilote/describe/<id>.json``:
filled keypoints, and ``occluded``, the names of joints placed rather than seen.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PROMPT = """Read the image file {path} with the Read tool. It is an engraving of {w}x{h} \
pixels (x to the right, y downward, origin at the top-left).

These figures were already described, with body points placed where each joint is SEEN \
(null where it was not seen). Left and right are the figure's own:

{figures}

For each figure, find the joints that are HIDDEN -- behind another figure, an object, \
drapery or the figure's own body -- but whose position can be inferred from the rest of \
the body (a leg that continues behind another leg, an arm behind a back). Give each such \
joint's inferred [x,y]. Do not include joints that are visible, and do not move any \
existing point. If an existing point actually sits on a joint that is hidden, list its \
name in "occluded" as well, without coordinates. Leave out joints outside the picture or \
that cannot be inferred.

Answer ONLY with JSON: {{"figures": [{{"id": "F1", "hidden": {{"left_knee": [x, y]}}, \
"occluded": ["right_ankle"]}}]}}"""


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
    """Fill hidden joints (only where no point was seen) and record them as occluded."""
    added = 0
    by_id = {f["id"]: f for f in answer.get("figures", []) if isinstance(f, dict)}
    for f in description.get("figures", []):
        got = by_id.get(f["id"], {})
        k = f.setdefault("keypoints", {})
        occluded = set(f.get("occluded", []))
        for name, p in (got.get("hidden") or {}).items():
            if name in k and k[name] is None and isinstance(p, list) and len(p) == 2:
                k[name] = [round(float(p[0])), round(float(p[1]))]
                occluded.add(name)
                added += 1
        occluded |= {n for n in got.get("occluded", []) if k.get(n)}
        f["occluded"] = sorted(occluded)
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("data/raw/emblematica/full"))
    parser.add_argument("--describe", type=Path, default=Path("data/derived/pilote/describe"))
    parser.add_argument("--model", default="opus")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    for path in sorted(args.describe.glob("*.json")):
        description = json.loads(path.read_text())
        if "_hidden_pass" in description:
            continue
        answer = ask(args.images / f"{path.stem}.jpg", description, args.model, args.timeout)
        added = merge(description, answer)
        description["_hidden_pass"] = {"model": args.model, "raw": answer}
        path.write_text(json.dumps(description, ensure_ascii=False, indent=1))
        occluded = sum(len(f.get("occluded", [])) for f in description["figures"])
        print(f"{path.stem}: {added} joints added, {occluded} marked hidden", flush=True)


if __name__ == "__main__":
    main()
