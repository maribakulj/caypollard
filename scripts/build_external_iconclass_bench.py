#!/usr/bin/env python3
"""Candidates for an external Iconclass bench from the local Emblematica Online picturae, away from the Iconclass AI test set.

Every local pictura with catalogue notations is compared, by a 64-bit difference hash of its image, with every image of the
Iconclass AI test set (ICAI), which the published VLMs and our head learn from. A pictura within 10 bits of an ICAI image is a
copy; a book with any copy is left out whole, since ICAI may hold other pages or editions of it. Nothing is drawn here: the
frozen dev/test lists are drawn by a later, reviewed step. Writes data/derived/external-iconclass/candidates.jsonl and
results/iconclass-externe/candidats.json (counts).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from caypollard.datasets.emblematica import book_id_from_pictura, parse_emblem_xml

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data/raw/emblematica"
ICAI = ROOT / "data/raw/iconclass-testset"
OUT = ROOT / "data/derived/external-iconclass"
RES = ROOT / "results/iconclass-externe"


def dhash(path: Path) -> int:
    try:
        g = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    except Exception:
        return -1
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    index = {}
    for line in open(EMB / "index.jsonl"):
        row = json.loads(line)
        index[row["emblemID"]] = row
    records = []
    for img in sorted((EMB / "images").glob("*.jpg")):
        eid = img.stem
        xml = EMB / "xml" / f"{eid}.xml"
        if not xml.exists():
            continue
        url = index.get(eid, {}).get("Pictura")
        r = parse_emblem_xml(xml.read_text(errors="replace"), emblem_id=eid, book_id=book_id_from_pictura(url),
                             pictura_url=url)
        if r.iconclass:
            records.append(r)
    icai = sorted(p for p in ICAI.glob("*.jpg"))
    with ProcessPoolExecutor(4) as pool:
        h_icai = list(pool.map(dhash, icai, chunksize=256))
        h_emb = list(pool.map(dhash, [EMB / "images" / f"{r.emblem_id}.jpg" for r in records], chunksize=256))
    ok_icai = np.array([h for h in h_icai if h >= 0], dtype=np.uint64)

    def nearest(h: int) -> int:
        return int(np.bitwise_count(np.bitwise_xor(ok_icai, np.uint64(h))).min()) if len(ok_icai) else 64

    dist = [nearest(h) if h >= 0 else -1 for h in h_emb]
    copies_books = {r.book_id for r, d in zip(records, dist, strict=True) if 0 <= d <= 10}
    rows = []
    for r, d in zip(records, dist, strict=True):
        rows.append({**r.as_manifest_record(), "dhash_min_bits_to_icai": d, "copy_of_icai": 0 <= d <= 10,
                     "book_has_icai_copy": r.book_id in copies_books,
                     "kept": d > 10 and r.book_id not in copies_books and r.book_id is not None})
    with open(OUT / "candidates.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    kept = [r for r in rows if r["kept"]]
    summary = {"picturae_with_notations_and_image": len(rows), "icai_images_hashed": len(ok_icai),
               "copies_of_icai": sum(r["copy_of_icai"] for r in rows), "books_with_a_copy": len(copies_books),
               "kept": len(kept), "kept_books": len({r["book_id"] for r in kept}),
               "notations_per_kept_picture_median": float(np.median([len(r["iconclass"]) for r in kept])) if kept else None,
               "distance_histogram": {str(b): sum(1 for r in rows if r["dhash_min_bits_to_icai"] == b) for b in range(0, 16)}}
    json.dump(summary, open(RES / "candidats.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
