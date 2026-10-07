#!/usr/bin/env python3
"""Does a figure's pose find the same subject across media, where pixels do not?

The copies test (``scripts/pose_copies.py``) was the easy case: a copy keeps the whole
composition, and pixels found it as well as pose did. Here the composition changes and
only the subject stays: an emblem print against museum paintings, sculptures, glass and
prints that Iconclass gives the same subject -- the Crucifixion, the Carrying of the
Cross, the Temptation of Adam and Eve, the Visitation, with the Baptism of Christ and
the Flagellation as museum-only distractors. These subjects were chosen because their
iconography all but fixes the figures' poses.

Pictures were kept by looking at them, on one criterion: the subject is the main scene
(not one panel of an altarpiece, an object, a page with a tiny cut, or a symbol such as
an apple for the Fall). Of 139 emblems carrying these notations, 14 qualify; of 120
museum works, 68. The selection is ``data/derived/subjects/selection.json``.

Every picture gets one pass of Opus (``scripts/pilot_describe.py``), no review. The two
channels are those of the copies test, unchanged: no weight was tuned here.

The question asked: an emblem searches the museum works only; a hit is a work of the
same subject. The reverse (a museum work searching the emblems) is reported beside it.

Run: ``uv run python scripts/pose_subjects.py describe --part 0/3`` (three in parallel),
then ``uv run python scripts/pose_subjects.py evaluate``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from pose_copies import pixel_matrix, pose_matrix

SUBJECTS = Path("data/derived/subjects")


def manifest() -> list[dict]:
    """The pictures, copied under ``images/`` with their subject and corpus."""
    path = SUBJECTS / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    selection = json.loads((SUBJECTS / "selection.json").read_text())
    museums = {}
    for line in Path("data/derived/museums-v0.2/manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        museums[row["wikidata"]] = row["image_path"]
    (SUBJECTS / "images").mkdir(parents=True, exist_ok=True)
    items = []
    for subject, ids in selection["emblems"].items():
        for e in ids:
            items.append({"id": e, "subject": subject, "corpus": "emblem"})
            shutil.copy(f"data/raw/emblematica/full/{e}.jpg", SUBJECTS / "images" / f"{e}.jpg")
    for subject, ids in selection["museums"].items():
        for q in ids:
            items.append({"id": q, "subject": subject, "corpus": "museum"})
            shutil.copy(museums[q], SUBJECTS / "images" / f"{q}.jpg")
    path.write_text(json.dumps(items, indent=1))
    return items


def cross_score(dist: np.ndarray, items: list[dict], query: str) -> dict:
    """Queries from one corpus search the other only; a hit shares the subject."""
    top1, aps, firsts, per_subject = [], [], [], {}
    for i, it in enumerate(items):
        if it["corpus"] != query:
            continue
        gallery = [j for j, other in enumerate(items) if other["corpus"] != query]
        hits = [
            items[j]["subject"] == it["subject"] for j in sorted(gallery, key=lambda j: dist[i, j])
        ]
        if not any(hits):
            continue
        top1.append(hits[0])
        firsts.append(hits.index(True) + 1)
        found, precisions = 0, []
        for rank, hit in enumerate(hits, 1):
            if hit:
                found += 1
                precisions.append(found / rank)
        aps.append(sum(precisions) / len(precisions))
        per_subject.setdefault(it["subject"], []).append(hits[0])
    return {
        "queries": len(top1),
        "top1": round(float(np.mean(top1)), 3),
        "mAP": round(float(np.mean(aps)), 3),
        "median_first_hit": float(np.median(firsts)),
        "top1_by_subject": {s: f"{sum(v)}/{len(v)}" for s, v in sorted(per_subject.items())},
    }


def chance(items: list[dict], query: str) -> dict:
    rng = np.random.default_rng(0)
    runs = [cross_score(rng.random((len(items), len(items))), items, query) for _ in range(500)]
    return {k: round(float(np.mean([r[k] for r in runs])), 3) for k in ("top1", "mAP")}


def describe_all(model: str, timeout: float, part: int, parts: int) -> None:
    from pilot_describe import describe

    (SUBJECTS / "describe").mkdir(parents=True, exist_ok=True)
    for item in manifest()[part::parts]:
        target = SUBJECTS / "describe" / f"{item['id']}.json"
        if target.exists():
            continue
        data = describe(SUBJECTS / "images" / f"{item['id']}.jpg", model, timeout)
        if str(data.get("_raw", "")).startswith("API Error"):
            print(f"{item['id']}: not described — {data['_raw'][:80]}", flush=True)
            continue
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        print(f"{item['id']}: {len(data.get('figures', []))} figures", flush=True)


def evaluate() -> None:
    items = [m for m in manifest() if (SUBJECTS / "describe" / f"{m['id']}.json").exists()]
    descriptions = []
    for m in items:
        d = json.loads((SUBJECTS / "describe" / f"{m['id']}.json").read_text())
        d["size"] = Image.open(SUBJECTS / "images" / f"{m['id']}.jpg").size
        descriptions.append(d)
    matrices = {
        "pose": pose_matrix(descriptions),
        "pixels": pixel_matrix([SUBJECTS / "images" / f"{m['id']}.jpg" for m in items]),
    }
    report = {
        "pictures": len(items),
        "emblems": sum(m["corpus"] == "emblem" for m in items),
        "museum_works": sum(m["corpus"] == "museum" for m in items),
    }
    for query in ("emblem", "museum"):
        block = {"chance": chance(items, query)}
        for name, dist in matrices.items():
            block[name] = cross_score(dist, items, query)
        report[f"{query}_queries"] = block
    for name, dist in matrices.items():
        np.save(SUBJECTS / f"{name}-distances.npy", dist)
        nearest = {}
        for i, it in enumerate(items):
            if it["corpus"] != "emblem":
                continue
            gallery = [j for j, o in enumerate(items) if o["corpus"] == "museum"]
            order = sorted(gallery, key=lambda j: dist[i, j])[:5]
            nearest[it["id"]] = [f"{items[j]['id']} ({items[j]['subject']})" for j in order]
        report[f"{name}_nearest_museum_works"] = nearest
    (SUBJECTS / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
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
