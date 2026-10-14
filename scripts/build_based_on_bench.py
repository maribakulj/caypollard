#!/usr/bin/env python3
"""Build the ``derivees`` bench from Wikidata's "based on" pairs.

From ``data/derived/based-on/pairs.jsonl`` (``scripts/harvest_based_on.py``):

- pairs whose two sides use the same image file are dropped (the answer is given), and so
  are derived works that are photographs of the painting itself rather than works made
  after it (types "archival resource", "glass plate negative", "photograph");
- every pair whose derived work is not a painting is kept (prints, drawings, tapestries,
  sculpture...: the change of medium), and as many painted copies are drawn at random
  (seed 0), so the two situations can be compared;
- each derived work gets a medium class from its Wikidata types, used to report scores
  per medium.

Images are fetched from Wikimedia Commons at 512 px (``Special:FilePath``; Commons
refuses arbitrary thumbnail widths on upload.wikimedia.org), 1.5 s apart. A derived work
searches the source paintings; a hit is its own source.

Run: ``uv run python scripts/build_based_on_bench.py``.
"""

from __future__ import annotations

import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("data/derived/based-on")
HEADERS = {
    "User-Agent": "caypollard/0.1 (research prototype; https://github.com/maribakulj/caypollard)"
}
NOT_DERIVED = {"archival resource", "glass plate negative", "photograph"}
MEDIA = (
    (
        "estampe",
        ("engraving", "print", "etching", "mezzotint", "lithograph", "woodcut", "aquatint"),
    ),
    ("dessin", ("drawing", "watercolor", "watercolour", "pastel", "gouache", "sketch")),
    ("textile", ("tapestry", "textile", "embroidery")),
    ("sculpture", ("sculpture", "relief", "statue", "bust")),
    ("copie peinte", ("painting", "artwork copy", "fresco", "altarpiece", "miniature")),
)


def medium(types: list[str]) -> str:
    text = " ".join(types).lower()
    for name, words in MEDIA:
        if any(w in text for w in words):
            return name
    return "autre"


def fetch(url: str, path: Path) -> bool:
    if path.exists():
        return True
    name = urllib.parse.unquote(url.rsplit("/", 1)[1])
    thumb = (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        + urllib.parse.quote(name)
        + "?width=512"
    )
    for attempt in range(4):
        try:
            request = urllib.request.Request(thumb, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
            time.sleep(1.5)
            return True
        except Exception as exc:
            wait = 5 * 2**attempt
            print(f"  {path.name}: retry in {wait}s ({exc})", flush=True)
            time.sleep(wait)
    return False


def main() -> None:
    pairs = [json.loads(line) for line in (ROOT / "pairs.jsonl").read_text().splitlines()]
    pairs = [
        p
        for p in pairs
        if p["work_image"] != p["source_image"]
        and not NOT_DERIVED & set(p["work_info"].get("types", []))
        and p["work"] != p["source"]
    ]
    for p in pairs:
        types = p["work_info"].get("types", [])
        # A work typed both painting and print is a print: the non-painting medium wins.
        media = [medium([t]) for t in types] or ["autre"]
        p["medium"] = next((m for m in media if m != "copie peinte"), media[0])
    other = [p for p in pairs if p["medium"] != "copie peinte"]
    painted = [p for p in pairs if p["medium"] == "copie peinte"]
    chosen = other + random.Random(0).sample(painted, min(len(other), len(painted)))
    (ROOT / "images").mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor

    # Four downloads at a time: some thumbnails are rendered on request and take long.
    jobs = {p["work"]: p["work_image"] for p in chosen} | {
        p["source"]: p["source_image"] for p in chosen
    }
    with ThreadPoolExecutor(4) as pool:
        ok = dict(
            zip(
                jobs,
                pool.map(lambda q: fetch(jobs[q], ROOT / "images" / f"{q}.jpg"), jobs),
                strict=True,
            )
        )
    print(f"{sum(ok.values())}/{len(ok)} images", flush=True)
    items: dict[str, dict] = {}
    for p in chosen:
        if ok[p["work"]] and ok[p["source"]]:
            items.setdefault(
                p["source"],
                {"id": p["source"], "label": p["source"], "corpus": "source", "medium": "peinture"},
            )
            items[p["work"]] = {
                "id": p["work"],
                "label": p["source"],
                "corpus": "derived",
                "medium": p["medium"],
            }
    (ROOT / "manifest.json").write_text(json.dumps(list(items.values()), indent=1))
    derived = [i for i in items.values() if i["corpus"] == "derived"]
    counts = {
        m: sum(i["medium"] == m for i in derived) for m in sorted({i["medium"] for i in derived})
    }
    print(
        f"{len(derived)} derived works, {len(items) - len(derived)} sources; {counts}", flush=True
    )


if __name__ == "__main__":
    main()
