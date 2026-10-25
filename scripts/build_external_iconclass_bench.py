#!/usr/bin/env python3
"""Candidates for an external Iconclass bench from the local Emblematica Online picturae, away from the Iconclass AI test set.

Every local pictura with catalogue notations is compared, by a 256-bit difference hash of its image (16 x 16 gradients), with
every image of the Iconclass AI test set (ICAI), which the published VLMs and our head learn from. The threshold was set by eye on
pairs (25/10/2026, sheets kept in results/iconclass-externe/etalonnage/): under 45 bits the pairs were the same emblem scanned twice,
and one copy was seen at 52. A book with any pictura under 50 bits of an ICAI image is left out whole, since ICAI may hold other pages
or editions of it. (A first 64-bit version, threshold 10, flagged 8,513 copies among look-alike prints: dropped.) Nothing is drawn here: the
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


def dhash(path: Path):
    try:
        g = np.asarray(Image.open(path).convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    except Exception:
        return None
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


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
    ok = [i for i, h in enumerate(h_icai) if h is not None]
    H = np.stack([h_icai[i] for i in ok])

    def nearest(h) -> tuple[int, str | None]:
        if h is None:
            return -1, None
        d = np.unpackbits(np.bitwise_xor(H, h), axis=1).sum(1)
        j = int(d.argmin())
        return int(d[j]), icai[ok[j]].name

    near = [nearest(h) for h in h_emb]
    dist = [d for d, _ in near]
    copies_books = {r.book_id for r, d in zip(records, dist, strict=True) if 0 <= d < 50}
    rows = []
    for r, (d, n) in zip(records, near, strict=True):
        rows.append({**r.as_manifest_record(), "dhash256_min_bits_to_icai": d, "nearest_icai": n,
                     "copy_of_icai": 0 <= d < 50, "book_has_icai_copy": r.book_id in copies_books,
                     "kept": d >= 50 and r.book_id not in copies_books and r.book_id is not None})
    with open(OUT / "candidates.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    kept = [r for r in rows if r["kept"]]
    summary = {"picturae_with_notations_and_image": len(rows), "icai_images_hashed": len(ok),
               "copies_of_icai": sum(r["copy_of_icai"] for r in rows), "books_with_a_copy": len(copies_books),
               "kept": len(kept), "kept_books": len({r["book_id"] for r in kept}),
               "notations_per_kept_picture_median": float(np.median([len(r["iconclass"]) for r in kept])) if kept else None,
               "distance_histogram": np.histogram([r["dhash256_min_bits_to_icai"] for r in rows],
                                                   bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 128])[0].tolist()}
    json.dump(summary, open(RES / "candidats.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
