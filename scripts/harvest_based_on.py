#!/usr/bin/env python3
"""Harvest Wikidata's documented derivations: works "based on" (P144) a painting.

Each pair is a transmission recorded by an institution or an editor: a print, a copy, a
tapestry, a sculpture... made after a painting. Both sides must have an image (P18). The
pairs are the raw material of the ``based-on`` bench: from the derived work, find its
source among many paintings, across media.

The query is kept small (Wikidata answers 502 to wide queries with many OPTIONAL
clauses); the types, creators, dates and collections are fetched afterwards in batches.
Output: ``data/derived/based-on/pairs.jsonl``, one line per pair.

Run: ``uv run python scripts/harvest_based_on.py``.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "caypollard/0.1 (research prototype; https://github.com/maribakulj/caypollard)",
}
OUT = Path("data/derived/based-on")

PAIRS = """
SELECT ?work ?source ?workImage ?sourceImage WHERE {
  ?work wdt:P144 ?source ; wdt:P18 ?workImage .
  ?source wdt:P31 wd:Q3305213 ; wdt:P18 ?sourceImage .
}"""

DETAILS = """
SELECT ?item ?label ?type ?typeLabel ?creatorLabel ?inception ?collectionLabel WHERE {
  VALUES ?item { %s }
  OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label) = "en") }
  OPTIONAL { ?item wdt:P31 ?type . ?type rdfs:label ?typeLabel . FILTER(LANG(?typeLabel) = "en") }
  OPTIONAL { ?item wdt:P170 ?creator . ?creator rdfs:label ?creatorLabel .
             FILTER(LANG(?creatorLabel) = "en") }
  OPTIONAL { ?item wdt:P571 ?inception }
}"""

COLLECTION = """
SELECT ?item ?collectionLabel WHERE {
  VALUES ?item { %s }
  ?item wdt:P195 ?collection . ?collection rdfs:label ?collectionLabel .
  FILTER(LANG(?collectionLabel) = "en")
}"""


def sparql(query: str, retries: int = 5) -> list[dict]:
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())["results"]["bindings"]
        except Exception as exc:  # 429, 502, timeouts: wait and retry
            wait = 5 * 2**attempt
            print(f"  retry in {wait}s ({exc})", flush=True)
            time.sleep(wait)
    raise RuntimeError("Wikidata does not answer")


def qid(binding: dict, key: str) -> str:
    return binding[key]["value"].rsplit("/", 1)[1]


def details(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {
        i: {"types": set(), "creators": set(), "collections": set()} for i in ids
    }
    for start in range(0, len(ids), 150):
        batch = ids[start : start + 150]
        values = " ".join(f"wd:{i}" for i in batch)
        for b in sparql(DETAILS % values):
            d = out[qid(b, "item")]
            if "label" in b:
                d["label"] = b["label"]["value"]
            if "typeLabel" in b:
                d["types"].add(b["typeLabel"]["value"])
            if "creatorLabel" in b:
                d["creators"].add(b["creatorLabel"]["value"])
            if "inception" in b:
                d["inception"] = b["inception"]["value"][:4]
        for b in sparql(COLLECTION % values):
            out[qid(b, "item")]["collections"].add(b["collectionLabel"]["value"])
        print(f"  details {start + len(batch)}/{len(ids)}", flush=True)
        time.sleep(2)
    return {
        i: {k: sorted(v) if isinstance(v, set) else v for k, v in d.items()} for i, d in out.items()
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sparql(PAIRS)
    pairs: dict[tuple[str, str], dict] = {}
    for b in rows:
        key = (qid(b, "work"), qid(b, "source"))
        # One image per item: the first Wikidata gives.
        pairs.setdefault(
            key,
            {
                "work": key[0],
                "source": key[1],
                "work_image": b["workImage"]["value"],
                "source_image": b["sourceImage"]["value"],
            },
        )
    print(f"{len(pairs)} pairs", flush=True)
    ids = sorted({i for k in pairs for i in k})
    info = details(ids)
    with (OUT / "pairs.jsonl").open("w") as f:
        for p in pairs.values():
            p["work_info"], p["source_info"] = info[p["work"]], info[p["source"]]
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"written {OUT / 'pairs.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
