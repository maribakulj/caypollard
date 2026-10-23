"""Run bricks on a bench, keep every intermediate, score, and write the common table.

A *combination* is either a pose chain (a figures brick, then a pose brick) or a vector
brick, or a sum of combinations already run. Its descriptions are cached under
``data/derived/atelier/<bench>/<combination>/`` and its distance matrix beside them, so a
combination is computed once and can be fused with others later. Each run writes
``results/atelier/<bench>/<combination>.json`` (scores, chance, bricks, timing, commit);
``table`` gathers them in ``results/atelier/TABLEAU.md``.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .benches import BENCHES, Bench, top1_detail
from .bricks import COCO_KEPT, brick
from .measure import (
    dense_matrix,
    pose_matrix,
    relation_words,
    scene_phrases,
    scene_words,
    set_matrix,
    vector_matrix,
    words_matrix,
    zsum,
)

CACHE = Path("data/derived/atelier")
RESULTS = Path("results/atelier")


def pose_name(figures: str, pose: str, joints: str | None = None) -> str:
    return f"pose--{figures}--{pose}" + (f"--{joints}" if joints else "")


def describe(bench: Bench, figures: str, pose: str) -> tuple[list[dict], list[float]]:
    """Boxes then points for every picture of the bench, cached one file per picture."""
    find, place = brick("figures", figures), brick("pose", pose)
    folder = CACHE / bench.name / pose_name(figures, pose)
    folder.mkdir(parents=True, exist_ok=True)
    out, seconds = [], []
    for it in bench.items():
        cache = folder / f"{it['id']}.json"
        image = Image.open(bench.image(it["id"])).convert("RGB")
        if cache.exists():
            d = json.loads(cache.read_text())
        else:
            start = time.time()
            boxes = find.boxes(image, it["id"], bench)
            points = place.points(image, boxes, it["id"], bench)
            d = {
                "figures": [{"box": b, "keypoints": k} for b, k in zip(boxes, points, strict=True)],
                "seconds": round(time.time() - start, 2),
            }
            cache.write_text(json.dumps(d))
        seconds.append(d.get("seconds", 0.0))
        d["size"] = image.size
        out.append(d)
    return out, seconds


def run_pose(bench_name: str, figures: str, pose: str, joints: str | None = None) -> dict:
    bench = BENCHES[bench_name]
    descriptions, seconds = describe(bench, figures, pose)
    kept = {"coco": set(COCO_KEPT)}.get(joints) if joints else None
    dist = pose_matrix(descriptions, joints=kept)
    figures_found = [len(d.get("figures", [])) for d in descriptions]
    extra = {
        "seconds_per_picture_median": float(np.median(seconds)),
        "figures_per_picture_median": float(np.median(figures_found)),
        "pictures_without_figure": int(sum(n == 0 for n in figures_found)),
    }
    bricks = [brick("figures", figures).meta(), brick("pose", pose).meta()]
    return _save(bench, pose_name(figures, pose, joints), dist, bricks, extra)


def run_relations(bench_name: str, figures: str, pose: str, relations_too: bool = True) -> dict:
    """Bags of postures and relations, from a pose chain's cached descriptions."""
    bench = BENCHES[bench_name]
    descriptions, _ = describe(bench, figures, pose)
    bags = [relation_words(d, relations_too) for d in descriptions]
    kind = "relations" if relations_too else "postures"
    extra = {
        "words_per_picture_median": float(np.median([sum(b.values()) for b in bags])),
        "vocabulary": len({w for b in bags for w in b}),
    }
    bricks = [brick("figures", figures).meta(), brick("pose", pose).meta()]
    return _save(bench, f"{kind}--{figures}--{pose}", words_matrix(bags), bricks, extra)


def scenes(bench: Bench, items: list[dict], name: str, workers: int = 4) -> list[dict]:
    """A scene brick's reading of every picture, cached one file per picture."""
    from concurrent.futures import ThreadPoolExecutor

    reader = brick("scene", name)
    folder = CACHE / bench.name / f"scene--{name}"
    folder.mkdir(parents=True, exist_ok=True)

    def one(item_id: str) -> dict:
        cache = folder / f"{item_id}.json"
        if not cache.exists():
            start = time.time()
            d = reader.scene(bench.image(item_id))
            d["seconds"] = round(time.time() - start, 1)
            cache.write_text(json.dumps(d, ensure_ascii=False))
            print(f"  {item_id}: {len(d.get('relations', []))} relations", flush=True)
        return json.loads(cache.read_text())

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, [it["id"] for it in items]))


def run_scene(bench_name: str, name: str, relations_too: bool = True) -> dict:
    bench = BENCHES[bench_name]
    items = bench.items()
    read = scenes(bench, items, name)
    bags = [scene_words(d, relations_too) for d in read]
    kind = "scene" if relations_too else "scene-figures"
    extra = {
        "seconds_per_picture_median": float(np.median([d.get("seconds", 0) for d in read])),
        "words_per_picture_median": float(np.median([sum(b.values()) for b in bags])),
        "vocabulary": len({w for b in bags for w in b}),
        "subjects_named": {
            it["id"]: d.get("subject", "") for it, d in zip(items, read, strict=True)
        },
    }
    dist = words_matrix(bags)
    if bench.rule == "all":
        # Sensitivity only, fixed before the LoRA result (Codex review, 23/10/2026): ties at
        # the best distance taken as equally likely, and every reading without figures
        # counted as a failed search. The main score stays the historic one.
        detail = top1_detail(dist, items)
        invalid = [not any(isinstance(f, dict) for f in d.get("figures") or []) for d in read]
        extra["sensitivity"] = {
            "invalid_readings": int(sum(invalid)),
            "top1_ties_equally_likely": round(float(np.mean([q["expected"] for q in detail])), 4),
            "queries_tied_at_best": int(sum(q["tied"] > 1 for q in detail)),
            "top1_invalid_as_failure": round(
                float(
                    np.mean([q["hit"] and not bad for q, bad in zip(detail, invalid, strict=True)])
                ),
                4,
            ),
        }
    return _save(bench, f"{kind}--{name}", dist, [brick("scene", name).meta()], extra)


def run_scene_vectors(bench_name: str, name: str, text: str = "minilm-multilingual") -> dict:
    """A scene brick's graphs compared phrase by phrase through a text encoder."""
    from caypollard.text.encoders import DEFAULT_MODELS, HuggingFaceTextEncoder

    bench = BENCHES[bench_name]
    items = bench.items()
    read = scenes(bench, items, name)
    phrases = [scene_phrases(d) for d in read]
    encoder = HuggingFaceTextEncoder(DEFAULT_MODELS[text], device="cpu")
    flat = sorted({p for ps in phrases for p in ps})
    vectors = dict(zip(flat, encoder.encode(flat), strict=True)) if flat else {}
    sets = [np.array([vectors[p] for p in ps]) for ps in phrases]
    extra = {
        "phrases_per_picture_median": float(np.median([len(p) for p in phrases])),
        "text": text,
    }
    bricks = [brick("scene", name).meta()]
    return _save(bench, f"scene-vecteurs--{name}--{text}", set_matrix(sets), bricks, extra)


def run_vector(bench_name: str, vector: str) -> dict:
    bench = BENCHES[bench_name]
    encoder = brick("vector", vector)
    images = [Image.open(bench.image(it["id"])).convert("RGB") for it in bench.items()]
    start = time.time()
    # In batches: a whole bench at once filled 18 GB of memory on 1,410 pictures.
    plain = np.concatenate([encoder.encode(images[i : i + 32]) for i in range(0, len(images), 32)])
    flipped = np.concatenate(
        [
            encoder.encode([ImageOps.mirror(im) for im in images[i : i + 32]])
            for i in range(0, len(images), 32)
        ]
    )
    extra = {"seconds_per_picture": round((time.time() - start) / len(images) / 2, 3)}
    return _save(bench, f"vector--{vector}", vector_matrix(plain, flipped), [encoder.meta()], extra)


def run_regions(bench_name: str, name: str) -> dict:
    bench = BENCHES[bench_name]
    encoder = brick("regions", name)
    images = [Image.open(bench.image(it["id"])).convert("RGB") for it in bench.items()]
    start = time.time()
    plain = np.concatenate([encoder.patches(images[i : i + 32]) for i in range(0, len(images), 32)])
    flipped = np.concatenate(
        [
            encoder.patches([ImageOps.mirror(im) for im in images[i : i + 32]])
            for i in range(0, len(images), 32)
        ]
    )
    encoded = time.time()
    dist = dense_matrix(plain, flipped, encoder.symmetric)
    extra = {
        "seconds_per_picture": round((encoded - start) / len(images) / 2, 3),
        "seconds_matching": round(time.time() - encoded, 1),
    }
    return _save(bench, f"regions--{name}", dist, [encoder.meta()], extra)


def run_fusion(bench_name: str, combinations: list[str]) -> dict:
    bench = BENCHES[bench_name]
    matrices = [np.load(CACHE / bench.name / f"{c}.npy") for c in combinations]
    name = "sum--" + "--and--".join(combinations)
    return _save(bench, name, zsum(*matrices), [], {"of": combinations})


def _save(bench: Bench, name: str, dist: np.ndarray, bricks: list, extra: dict) -> dict:
    items = bench.items()
    (CACHE / bench.name).mkdir(parents=True, exist_ok=True)
    np.save(CACHE / bench.name / f"{name}.npy", dist)
    result = {
        "bench": bench.name,
        "combination": name,
        "date": datetime.date.today().isoformat(),
        "commit": _commit(),
        "bricks": bricks,
        "scores": bench.score(dist, items),
        "chance": bench.chance(items),
        **extra,
    }
    folder = RESULTS / bench.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    table()
    return result


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def table() -> Path:
    """One Markdown table per bench: every combination run so far, best first."""
    lines = [
        "# Atelier — tableau commun",
        "",
        "Generated by `scripts/atelier.py`; do not edit by hand. top1 = the nearest answer is",
        "right; mAP = average precision over all right answers. Settings of each brick are in",
        "`src/caypollard/atelier/bricks.py`; each result's JSON keeps scores per label.",
        "",
    ]
    for bench in BENCHES.values():
        runs = [json.loads(p.read_text()) for p in sorted((RESULTS / bench.name).glob("*.json"))]
        if not runs:
            continue
        directions = list(runs[0]["scores"])
        lines += [f"## {bench.name}", "", bench.description, ""]
        header = "| combination | " + " | ".join(f"{d} top1 | {d} mAP" for d in directions)
        lines += [header + " | s/picture |", "|" + "---|" * (1 + 2 * len(directions) + 1)]
        chance = runs[0]["chance"]
        cells = " | ".join(f"{chance[d]['top1']} | {chance[d]['mAP']}" for d in directions)
        lines.append(f"| *chance* | {cells} | |")
        runs.sort(key=lambda r: -r["scores"][directions[0]]["mAP"])
        for r in runs:
            cells = " | ".join(
                f"{r['scores'][d]['top1']} | {r['scores'][d]['mAP']}" for d in directions
            )
            speed = r.get("seconds_per_picture_median", r.get("seconds_per_picture", ""))
            lines.append(f"| {r['combination']} | {cells} | {speed} |")
        lines.append("")
        for d in directions:
            media = sorted({m for r in runs for m in r["scores"][d].get("by_medium", {})})
            if not media:
                continue
            counts = {m: runs[0]["scores"][d]["by_medium"].get(m, {}).get("queries") for m in media}
            lines += [
                f"Top1 by medium ({d}; queries: "
                + ", ".join(f"{m} {n}" for m, n in counts.items())
                + ")",
                "",
                "| combination | " + " | ".join(media) + " |",
                "|" + "---|" * (1 + len(media)),
            ]
            for r in runs:
                by = r["scores"][d].get("by_medium", {})
                cells = " | ".join(str(by.get(m, {}).get("top1", "")) for m in media)
                lines.append(f"| {r['combination']} | {cells} |")
            lines.append("")
    path = RESULTS / "TABLEAU.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
