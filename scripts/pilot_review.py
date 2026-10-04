#!/usr/bin/env python3
"""Have Claude (Opus) review each pilot picture against its current cut, and fix the points.

Opus sees the engraving and the same engraving with the current silhouettes and points
drawn on it, and is asked what is wrong -- a limb given to the wrong figure, a limb
missing, a point off its joint -- and for corrected points. The first such review, asked by
Marcel on E014784, found Death's front leg passing behind the nobleman with its bare foot
given to him. Corrections replace the proposal's points (logged under ``_reviews``); the
pictures whose points changed are re-cut by ``scripts/pilot_build.py --resegment``.

A model reviewing the same model's work can repeat its own errors: this is a pass before a
person's review, not instead of it. Run with ``uv run --with opencv-python-headless``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from caypollard.figures import KEYPOINTS, rings

PILOTE = Path("data/derived/pilote")
COLOURS = [(209, 73, 91), (46, 134, 171), (63, 143, 41), (201, 138, 0), (120, 80, 160)]

PROMPT = """Read two image files with the Read tool: {image} (an engraving, {w}x{h} pixels, \
x to the right, y downward) and {overlay} (the same engraving with the current analysis \
drawn on it: each figure's silhouette in its colour -- {legend} -- and its body points as \
dots, hollow where the joint is hidden).

Current figures and body points (left/right are the figure's own; crown = top of skull, \
chin = bottom of chin, nose = tip of nose):
{figures}

Review the analysis against the engraving, figure by figure and limb by limb. Look for:
a limb or foot given to the wrong figure; a limb missing (a joint left null, or placed on \
another body part); a point clearly off its joint; a joint marked hidden that is visible, \
or visible that is hidden; parts of a figure left out of every silhouette.

First explain briefly what is wrong, if anything. Then end with a fenced ```json block: \
{{"problems": ["..."], "figures": [{{"id": "F1", "keypoints": {{only the joints to change: \
"name": [x, y] or null}}, "occluded": [the full list of hidden joint names for this figure]}}]}}. \
List only figures with changes; an empty "figures" list means the analysis is right."""


def overlay(record: dict, path: Path) -> None:
    img = Image.open(record["image"]).convert("RGB")
    layer = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(layer)
    for i, f in enumerate(record["figures"]):
        c = COLOURS[i % len(COLOURS)]
        for ring in rings(f.get("polygon")):
            draw.polygon([tuple(p) for p in ring], fill=(*c, 90), outline=(*c, 255))
        for name, p in f["keypoints"].items():
            if p:
                hidden = name in f.get("occluded", [])
                box = [p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7]
                draw.ellipse(
                    box, fill=None if hidden else (*c, 255), outline=(255, 255, 0, 255), width=3
                )
    Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB").save(path)


def review(name: str, model: str, timeout: float) -> dict:
    record = json.loads((PILOTE / "records" / f"{name}.json").read_text())
    description = json.loads((PILOTE / "describe" / f"{name}.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        drawn = Path(tmp) / f"{name}-overlay.jpg"
        overlay(record, drawn)
        names = ["red", "blue", "green", "orange", "purple"]
        legend = ", ".join(
            f"{names[i % 5]} = {f['id']} {f['name']}" for i, f in enumerate(record["figures"])
        )
        figures = json.dumps(
            [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "box": f["box"],
                    "keypoints": {k: f["keypoints"].get(k) for k in KEYPOINTS},
                    "occluded": f.get("occluded", []),
                }
                for f in description["figures"]
            ],
            ensure_ascii=False,
        )
        w, h = record["size"]
        prompt = PROMPT.format(
            image=Path(record["image"]).resolve(),
            overlay=drawn,
            w=w,
            h=h,
            legend=legend,
            figures=figures,
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
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            cwd=Path.cwd(),
            env=None,
        )
    text = str(json.loads(run.stdout).get("result", ""))
    start = text.rfind("```json")
    block = text[start + 7 :] if start >= 0 else text[text.find("{") :]
    try:
        answer, _ = json.JSONDecoder().raw_decode(block.strip())
    except json.JSONDecodeError:
        answer = {"problems": ["(réponse illisible)"], "figures": []}
    answer["_text"] = text
    return answer


def apply(description: dict, answer: dict, model: str) -> int:
    """Write the reviewed points into the proposal; returns how many points changed."""
    changed = 0
    by_id = {f["id"]: f for f in description["figures"]}
    for fix in answer.get("figures", []):
        f = by_id.get(fix.get("id"))
        if f is None:
            continue
        for joint, p in (fix.get("keypoints") or {}).items():
            if joint in KEYPOINTS and (p is None or (isinstance(p, list) and len(p) == 2)):
                new = None if p is None else [round(float(p[0])), round(float(p[1]))]
                if f["keypoints"].get(joint) != new:
                    f["keypoints"][joint] = new
                    changed += 1
        if isinstance(fix.get("occluded"), list):
            f["occluded"] = sorted(n for n in fix["occluded"] if n in KEYPOINTS)
    description.setdefault("_reviews", []).append(
        {"model": model, "problems": answer.get("problems", []), "answer": answer}
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="pictures to review (default: all)")
    parser.add_argument("--model", default="opus")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    names = args.names or sorted(p.stem for p in (PILOTE / "describe").glob("*.json"))
    for name in names:
        path = PILOTE / "describe" / f"{name}.json"
        description = json.loads(path.read_text())
        answer = review(name, args.model, args.timeout)
        if answer["_text"].startswith("API Error"):
            print(f"{name}: not reviewed — {answer['_text'][:80]}", flush=True)
            continue
        changed = apply(description, answer, args.model)
        path.write_text(json.dumps(description, ensure_ascii=False, indent=1))
        problems = " | ".join(answer.get("problems", [])) or "rien à corriger"
        print(f"{name}: {changed} points changed — {problems}", flush=True)


if __name__ == "__main__":
    main()
