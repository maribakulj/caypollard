#!/usr/bin/env python3
"""Does a non-human reader still name the object once the picture is reduced to strokes?

A pictogram of a house is read as a house by anyone who already holds the schema "house".
The question here is whether a machine reader holds enough of it: the same closed-vocabulary
naming question is put to the namer on four versions of each picture -- the original, its
contour drawing, the sketch of forty strokes, the pictogram of twelve -- and the names given
at each level are compared with the names given on the original.

The floor is what agreement looks like by accident: the names of a level compared with the
names given on the original of a *different* picture, over a seeded permutation. A level
whose agreement sits at that floor tells the reader nothing about this picture; a level well
above it still carries the object. Both the agreement on the full name set (Jaccard) and the
survival of the original's first name are reported, per level and per corpus.

The namer is Claude (Sonnet) through Claude Code, as in the demo; it is not deterministic,
so even the original against itself would not score 1.0, and that ceiling is measured too by
naming the original twice.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from caypollard import sketch as sketching
from caypollard.demo.namer import name_nodes

LEVELS = ("original", "original_bis", "contours", "croquis", "pictogramme")


def names_of(nodes: list[dict]) -> list[str]:
    return [n["name"] for n in nodes]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "items", type=Path, default=Path("data/derived/viewer/items.json"), nargs="?"
    )
    parser.add_argument("--museums", type=int, default=30)
    parser.add_argument("--emblems", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=6)
    parser.add_argument("--images", type=Path, default=Path("data/derived/sketch/naming"))
    parser.add_argument("--output", type=Path, default=Path("results/wide/pictogram-naming.json"))
    args = parser.parse_args()
    args.images.mkdir(parents=True, exist_ok=True)

    items = json.loads(args.items.read_text(encoding="utf-8"))["items"]
    rng = random.Random(args.seed)
    museums = [it for it in items if it["c"] == "musée" and it["ic"] and it["typ"]]
    emblems = [it for it in items if it["c"] == "emblème" and it["ic"]]
    picked = rng.sample(museums, args.museums) + rng.sample(emblems, args.emblems)

    # 1. the four renderings of each picture, on disk, so the namer reads files as it always does
    jobs: list[tuple[str, str, Path]] = []
    for it in picked:
        stem = it["id"].replace(":", "_")
        with Image.open(it["img"]) as image:
            image = image.convert("RGB")
            sk = sketching.sketch(image)
            original = args.images / f"{stem}-original.png"
            small = image.copy()
            small.thumbnail((512, 512))
            small.save(original, "PNG")
            sketching.edges_image(sk).save(args.images / f"{stem}-contours.png", "PNG")
            sketching.render(sk.sketch, sk.size, width=2).save(
                args.images / f"{stem}-croquis.png", "PNG"
            )
            sketching.render(sk.pictogram, sk.size, width=3).save(
                args.images / f"{stem}-pictogramme.png", "PNG"
            )
        for level in LEVELS:
            file = level if level != "original_bis" else "original"
            jobs.append((it["id"], level, (args.images / f"{stem}-{file}.png").resolve()))
    print(f"{len(picked)} pictures, {len(jobs)} naming calls", file=sys.stderr)

    # 2. the namer, a few calls at a time
    t = time.time()
    answers: dict[tuple[str, str], dict] = {}

    def ask(job: tuple[str, str, Path]) -> tuple[tuple[str, str], dict]:
        item_id, level, path = job
        return (item_id, level), name_nodes(path, model="sonnet", max_nodes=args.max_nodes)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, (key, answer) in enumerate(pool.map(ask, jobs), 1):
            answers[key] = answer
            print(f"  {n}/{len(jobs)} ({time.time() - t:.0f}s)", file=sys.stderr, end="\r")
    print(file=sys.stderr)

    # 3. agreement per level, against the original, and the floor by permutation
    ids = [it["id"] for it in picked]
    corpus = {it["id"]: it["c"] for it in picked}
    per_item = []
    for it in picked:
        row = {"id": it["id"], "corpus": it["c"], "type": it["typ"][:2], "iconclass": it["ic"][:4]}
        for level in LEVELS:
            a = answers[(it["id"], level)]
            row[level] = names_of(a["nodes"])
            if a.get("error"):
                row[level + "_error"] = a["error"]
        per_item.append(row)

    def score(level: str, pairing: list[tuple[str, str]]) -> dict:
        j, first, empty = [], [], 0
        for a_id, b_id in pairing:
            got = names_of(answers[(a_id, level)]["nodes"])
            ref = names_of(answers[(b_id, "original")]["nodes"])
            if not got:
                empty += 1
            j.append(jaccard(got, ref))
            first.append(1.0 if ref and ref[0] in got else 0.0)
        return {
            "jaccard_with_original": round(float(np.mean(j)), 3),
            "first_name_survives": round(float(np.mean(first)), 3),
            "empty_answers": empty,
            "n": len(pairing),
        }

    permutation = ids[:]
    rng.shuffle(permutation)
    while any(a == b for a, b in zip(ids, permutation, strict=True)):
        rng.shuffle(permutation)
    summary: dict = {"levels": {}, "floor": {}, "by_corpus": {}}
    for level in LEVELS:
        summary["levels"][level] = score(level, [(i, i) for i in ids])
        summary["floor"][level] = score(level, list(zip(ids, permutation, strict=True)))
        for c in ("musée", "emblème"):
            sub = [i for i in ids if corpus[i] == c]
            summary["by_corpus"].setdefault(c, {})[level] = score(level, [(i, i) for i in sub])

    out = {
        "question": "does the namer still name the object at each level of abstraction?",
        "namer": next(iter(answers.values()))["model"],
        "pictures": {"musées": args.museums, "emblèmes": args.emblems, "seed": args.seed},
        "sketch_params": sketching.DEFAULTS,
        "summary": summary,
        "items": per_item,
        "reading": "jaccard_with_original: overlap of the name sets between a level and the "
        "original (original_bis is the same original named a second time: the ceiling of a "
        "non-deterministic namer). first_name_survives: the original's first name is among "
        "the level's names. floor: the same scores against another picture's original.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"levels": summary["levels"], "floor": summary["floor"]}, ensure_ascii=False, indent=1
        )
    )


if __name__ == "__main__":
    main()
