#!/usr/bin/env python3
"""Ask Claude, headless, for each pilot picture's figures, skeletons and objects.

This is the proposal step of the pilot: a vision-language model names the figures, places
seventeen body points on each, names the objects that matter and outlines them with a box,
and says in one sentence who does what to whom. The geometry is then refined (masks) and
the relations are computed from it, never taken from the model; the model's own sentence
and its ``held_by`` claims are kept apart, as the verbal representation to compare with.

Keypoints from photo-trained pose detectors were tried first and fail on these engravings
(Death, a skeleton, is not a person to them), which is why a reader that can see is used.
Its points are proposals: the viewer lets a person drag them, and the corrections are kept.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

from caypollard.figures import KEYPOINTS

PROMPT = """Read the image file {path} with the Read tool. It is an engraving of {w}x{h} pixels \
(x to the right, y downward, origin at the top-left).

Describe it as JSON, in pixel coordinates of that {w}x{h} image.

"figures": every acting figure (person, skeleton/Death, angel, winged child, personification, \
animal acting as a character). For each:
  "id": "F1", "F2"...; "name": a short, specific name in French ("la Mort (squelette)", \
"l'Amour divin (enfant ailé, auréolé)", "l'Âme (enfant auréolé)", "jeune homme");
  "box": [x0,y0,x1,y1] tight around the whole figure;
  "attributes": what belongs to the body itself, in French ("ailes", "auréole", "couronne \
de laurier", "bandeau sur les yeux") -- these are NOT objects;
  "keypoints": an object with exactly these keys: {keys}. Each is [x,y] placed ON that body \
part, or null if hidden. Left and right are the FIGURE's own left and right, not the viewer's.

"objects": only the things that matter to the meaning: held, worn as an attribute, pointed \
at, given, or symbolic (an hourglass, a bow, a heart, a crown). Not walls, ground, sky, \
ordinary clothing or decoration, not wings or haloes. Look closely at what each \
object actually is before naming it. For each: "id": "O1"...; "name" in French; "box" tight \
around it; "held_by": a figure id or null.

"phrase": one French sentence saying who does what to whom in the scene, as you see it.

Answer ONLY with the JSON object {{"figures": [...], "objects": [...], "phrase": "..."}}."""


def describe(path: Path, model: str, timeout: float) -> dict:
    w, h = Image.open(path).size
    prompt = PROMPT.format(path=path.resolve(), w=w, h=h, keys=", ".join(KEYPOINTS))
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
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    payload = json.loads(run.stdout)
    text = str(payload.get("result", ""))
    # The answer is the first JSON object in the text, whatever prose follows it.
    start = text.find("{")
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else ({}, 0)
    except json.JSONDecodeError:
        data = {}
    data = {"figures": [], "objects": [], "phrase": "", **data}
    data["_model"] = ", ".join(payload.get("modelUsage", {}).keys()) or model
    data["_size"] = [w, h]
    data["_raw"] = text
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("--images", type=Path, default=Path("data/raw/emblematica/full"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/pilote/describe"))
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for item_id in json.loads(args.selection.read_text()):
        name = item_id.split(":")[1]
        image, target = args.images / f"{name}.jpg", args.output / f"{name}.json"
        if target.exists() or not image.exists():
            continue
        data = describe(image, args.model, args.timeout)
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        print(
            f"{name}: {len(data['figures'])} figures, {len(data['objects'])} objects"
            f" — {data.get('phrase', '')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
