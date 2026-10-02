#!/usr/bin/env python3
"""Build what the viewer reads: thumbnails, one item table, and neighbour lists.

The viewer (`viewer/`) is a page that shows hundreds of pictures at once and, for
any one of them, the neighbours each representation returns. It computes
nothing: everything it displays is written here, once, from the frozen manifests
and the embedding tables the experiments used. Rerun this when a table changes.

Outputs, under `data/derived/viewer/`:

  thumbs/<id>.jpg      one thumbnail per picture, longest side --thumb pixels
  items.json           every picture of the pool with the attributes the regimes
                       read (corpus, collection, kind, genre, century, creator,
                       material, notations, motto, named nodes), labelled
  tables.json          the representations, their pool and their provenance
  ids-<slug>.json      row order of one table, as indices into items
  nb-<slug>.bin        uint32 [rows, k]  neighbour rows, nearest first
  sc-<slug>.bin        float32 [rows, k] cosine similarity of each neighbour

A table's neighbours are computed among the rows that are also in the pool, so a
picture with no image on disk can neither be shown nor be a neighbour.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from caypollard.embeddings.store import load_embedding_table

BASE = re.compile(r"[(\[]")


def slug(name: str) -> str:
    text = name.lower()
    text = re.sub(r"[àâä]", "a", text)
    text = re.sub(r"[éèêë]", "e", text)
    text = re.sub(r"[îï]", "i", text)
    text = re.sub(r"[ôö]", "o", text)
    text = re.sub(r"[ùûü]", "u", text)
    text = text.replace("œ", "oe").replace("ç", "c")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def thumb_name(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", item_id) + ".jpg"


def make_thumb(job: tuple[str, str, int]) -> tuple[str, bool]:
    source, target, size = job
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((size, size), Image.LANCZOS)
            image.save(target, "JPEG", quality=84, optimize=True)
        return source, True
    except Exception:
        return source, False


def first(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value or None


def century_of(record: dict) -> int | None:
    year = str(record.get("inception") or "")[:4]
    return int(year) // 100 + 1 if year.isdigit() else None


def load_records(paths: list[Path]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                records[str(row["id"])] = row
    return records


def load_nodes(paths: list[Path]) -> dict[str, list[str]]:
    nodes: dict[str, list[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                found = row.get("nodes", [])
                nodes[str(row["id"])] = [n["name"] for n in found if isinstance(n, dict)]
    return nodes


def iconclass_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            hits = row.get("wikidata") or []
            if hits:
                labels[row["notation"]] = hits[0].get("label", "")
    return labels


def neighbours(vectors: np.ndarray, k: int, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    n = vectors.shape[0]
    k = min(k, n - 1)
    rows = np.empty((n, k), dtype=np.uint32)
    scores = np.empty((n, k), dtype=np.float32)
    transposed = vectors.T.copy()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = vectors[start:stop] @ transposed
        sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        rows[start:stop] = np.take_along_axis(part, order, axis=1)
        scores[start:stop] = np.take_along_axis(part_scores, order, axis=1)
        print(f"  {stop}/{n}", file=sys.stderr, end="\r")
    print(file=sys.stderr)
    return rows, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument("--table", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--nodes", action="append", default=[], type=Path, help="named-node transcriptions (jsonl)"
    )
    parser.add_argument("--labels", type=Path, default=Path("data/derived/wikidata-labels.json"))
    parser.add_argument("--genre-labels", type=Path, default=Path("data/derived/genre-labels.json"))
    parser.add_argument(
        "--iconclass-labels", type=Path, default=Path("data/derived/iconclass-wikidata.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("data/derived/viewer"))
    parser.add_argument("--thumb", type=int, default=320)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument(
        "--hub-fraction",
        type=float,
        default=0.032,
        help="a base notation carried by more than this share of the pool is a category, not a "
        "subject, and sharing it says nothing; the same cut the regime scripts apply",
    )
    parser.add_argument("--skip-thumbs", action="store_true")
    args = parser.parse_args()

    labels: dict[str, str] = {}
    if args.labels.exists():
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if args.genre_labels.exists():
        labels.update(json.loads(args.genre_labels.read_text(encoding="utf-8")))
    ic_labels = iconclass_labels(args.iconclass_labels)
    nodes = load_nodes(args.nodes)

    def named(values) -> list[str]:
        return [labels.get(v, v) for v in (values or [])]

    records = load_records(args.manifest)
    thumbs_dir = args.output / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    index_of: dict[str, int] = {}
    jobs: list[tuple[str, str, int]] = []
    used_notations: set[str] = set()
    for item_id in sorted(records):
        row = records[item_id]
        image_path = row.get("image_path")
        if not image_path or not Path(image_path).exists():
            continue
        is_emblem = item_id.startswith("emblematica:")
        notations = [str(x) for x in row.get("iconclass") or []]
        for notation in notations:
            used_notations.add(BASE.split(notation)[0].strip())
        thumb = thumb_name(item_id)
        item = {
            "id": item_id,
            "c": "emblème" if is_emblem else "musée",
            "col": str(row.get("collection") or ""),
            "typ": ["estampe"] if is_emblem else named(row.get("type")),
            "gen": named(row.get("genre")),
            "cen": century_of(row),
            "cre": named(row.get("creator")),
            "mat": named(row.get("material")),
            "ic": notations,
            "th": f"thumbs/{thumb}",
            "img": image_path,
            "url": row.get("source_url") or row.get("pictura_url") or "",
        }
        if is_emblem:
            item["mot"] = html.unescape(row.get("motto") or "")
            item["book"] = row.get("book_id") or ""
        if item_id in nodes:
            item["nodes"] = nodes[item_id]
        index_of[item_id] = len(items)
        items.append(item)
        target = thumbs_dir / thumb
        if not target.exists():
            jobs.append((image_path, str(target), args.thumb))

    print(
        f"{len(items)} pictures with an image on disk, {len(jobs)} thumbnails to make",
        file=sys.stderr,
    )
    if jobs and not args.skip_thumbs:
        failed = 0
        with ProcessPoolExecutor() as pool:
            for done, (_, ok) in enumerate(pool.map(make_thumb, jobs, chunksize=64), 1):
                failed += 0 if ok else 1
                if done % 500 == 0:
                    print(f"  {done}/{len(jobs)} thumbnails", file=sys.stderr, end="\r")
        print(f"\n  {failed} images could not be read", file=sys.stderr)

    # Drop pictures whose thumbnail does not exist after all, so the page never asks for a
    # missing file.
    kept = [it for it in items if (args.output / it["th"]).exists()]
    for it in kept:
        # The page lays thumbnails out in justified rows, which needs each one's proportions.
        with Image.open(args.output / it["th"]) as thumb:
            it["w"], it["h"] = thumb.size
    index_of = {it["id"]: i for i, it in enumerate(kept)}
    items = kept

    labels_used = {n: ic_labels[n] for n in used_notations if n in ic_labels}
    carried: dict[str, int] = {}
    for it in items:
        for notation in {BASE.split(n)[0].strip() for n in it["ic"]}:
            carried[notation] = carried.get(notation, 0) + 1
    hubs = sorted(n for n, c in carried.items() if c > args.hub_fraction * len(items))
    print(
        f"{len(hubs)} hub notations above {args.hub_fraction:.1%} of the pool: {hubs}",
        file=sys.stderr,
    )
    (args.output / "items.json").write_text(
        json.dumps(
            {"items": items, "iconclass": labels_used, "hubs": hubs},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    tables_meta = []
    for spec in args.table:
        name, _, path = spec.partition("=")
        table = load_embedding_table(path)
        keep = [i for i, item_id in enumerate(table.ids) if item_id in index_of]
        vectors = table.vectors[keep]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = (vectors / norms).astype(np.float32)
        print(f"{name}: {len(keep)} of {len(table.ids)} rows are in the pool", file=sys.stderr)
        rows, scores = neighbours(vectors, args.k)
        s = slug(name)
        (args.output / f"ids-{s}.json").write_text(
            json.dumps([index_of[table.ids[i]] for i in keep], separators=(",", ":")),
            encoding="utf-8",
        )
        rows.tofile(args.output / f"nb-{s}.bin")
        scores.tofile(args.output / f"sc-{s}.bin")
        tables_meta.append(
            {
                "slug": s,
                "name": name,
                "path": path,
                "rows": len(keep),
                "rows_in_table": len(table.ids),
                "dimension": int(vectors.shape[1]),
                "k": int(rows.shape[1]),
                "method": table.metadata.get("method"),
                "family": table.metadata.get("family"),
            }
        )
    (args.output / "tables.json").write_text(
        json.dumps(tables_meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
