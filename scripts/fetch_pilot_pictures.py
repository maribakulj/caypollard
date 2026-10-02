#!/usr/bin/env python3
"""Fetch the pilot's picturae at the resolution of the scan, not the 200 px access copy.

The pool's images under ``data/raw/emblematica/images/`` are Emblematica's ``_access``
copies, 200 pixels wide: enough for a thumbnail, not for a hand or a face. Each emblem's
XML record names the scanned leaf it was cut from (``<book>_0114-0115.jp2``); Emblematica's
own IIIF server no longer answers, but the books are Internet Archive items, so the leaf
is fetched there at full size and the pictura is found on it by matching the 200 px copy
at every scale. The page itself is kept too, since its borders are ornament.

Run with ``uv run --with opencv-python-headless``; OpenCV is not a project dependency.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HEADERS = {"User-Agent": "Mozilla/5.0 (caypollard research)"}
RECORD = "http://emblemimages.library.illinois.edu/{book}/emblematica/emblem{num}.xml"
LEAF = "https://archive.org/download/{book}/page/n{leaf}.jpg"


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def locate(page: np.ndarray, small: np.ndarray) -> tuple[float, tuple[int, int, int, int]]:
    """Best (score, box) of ``small`` on ``page``, searched over the scale of the pictura."""
    work = 600 / page.shape[1]
    page_g = cv2.cvtColor(cv2.resize(page, None, fx=work, fy=work), cv2.COLOR_BGR2GRAY)
    small_g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    best = (-1.0, (0, 0, 0, 0))
    for width in range(120, page_g.shape[1] + 1, 4):
        h = round(small_g.shape[0] * width / small_g.shape[1])
        if h > page_g.shape[0]:
            break
        templ = cv2.resize(small_g, (width, h))
        score = cv2.matchTemplate(page_g, templ, cv2.TM_CCOEFF_NORMED)
        _, peak, _, (x, y) = cv2.minMaxLoc(score)
        if peak > best[0]:
            best = (peak, (round(x / work), round(y / work), round(width / work), round(h / work)))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("--items", type=Path, default=Path("data/derived/viewer/items.json"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/emblematica/full"))
    parser.add_argument("--pause", type=float, default=3.0)
    parser.add_argument("--accept", type=float, default=0.6, help="lowest match taken as found")
    args = parser.parse_args()

    items = {it["id"]: it for it in json.loads(args.items.read_text())["items"]}
    pages = args.output / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "crops.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for item_id in json.loads(args.selection.read_text()):
        name = item_id.split(":")[1]
        if name in log:
            continue
        item = items[item_id]
        xml = get(RECORD.format(book=item["book"], num=name[1:])).decode("utf-8")
        leaves = re.search(r"_(\d{4})(?:-(\d{4}))?\.jp2", xml)
        small = cv2.imread(item["img"])
        # The Archive's page index is not always the scan's file number: try the named
        # leaves first, then their neighbours, and stop at the first convincing match.
        named = [int(g) for g in leaves.groups() if g]
        candidates = list(dict.fromkeys(named + [n + d for d in (1, -1, 2, -2) for n in named]))
        found = []
        for leaf in candidates:
            page_file = pages / f"{item['book']}_{leaf:04d}.jpg"
            if not page_file.exists():
                page_file.write_bytes(get(LEAF.format(book=item["book"], leaf=leaf)))
                time.sleep(args.pause)
            page = cv2.imread(str(page_file))
            score, box = locate(page, small)
            found.append((score, box, f"{leaf:04d}", page))
            if score >= args.accept:
                break
        score, (x, y, w, h), leaf, page = max(found, key=lambda f: f[0])
        if score < args.accept:
            print(f"{name}: no convincing match (best {score:.3f} on leaf {leaf})", flush=True)
            continue
        cv2.imwrite(
            str(args.output / f"{name}.jpg"),
            page[y : y + h, x : x + w],
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        log[name] = {
            "book": item["book"],
            "leaf": leaf,
            "box": [x, y, w, h],
            "match": round(score, 3),
        }
        log_path.write_text(json.dumps(log, indent=1))
        print(f"{name}: leaf {leaf}, {w}x{h}, match {score:.3f}", flush=True)


if __name__ == "__main__":
    main()
