#!/usr/bin/env python3
"""Fetch human-readable labels for the Wikidata ids a manifest carries.

The museum manifests record object kinds, creators, materials and genres as
Wikidata ids (`Q3305213`), which is right for a benchmark and useless for a
person looking at a picture. This asks Wikidata once, in batches of fifty, and
keeps the answer in a JSON file next to the other derived data, so the viewer
never touches the network and a label that was fetched once is never fetched
again. French first, English otherwise, and the bare id when neither exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "caypollard/0.7 (research code; viewer labels)"
FIELDS = ("type", "creator", "material", "genre", "depicts")


def ids_in(manifests: list[Path]) -> set[str]:
    wanted: set[str] = set()
    for path in manifests:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for field in FIELDS:
                for value in row.get(field) or []:
                    if isinstance(value, str) and value.startswith("Q"):
                        wanted.add(value)
    return wanted


def fetch_batch(ids: list[str], languages: str) -> dict[str, str]:
    query = urllib.parse.urlencode(
        {
            "action": "wbgetentities",
            "ids": "|".join(ids),
            "props": "labels",
            "languages": languages,
            "format": "json",
        }
    )
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    labels: dict[str, str] = {}
    preferred = languages.split("|")
    for item, entity in payload.get("entities", {}).items():
        found = entity.get("labels", {})
        for language in preferred:
            if language in found:
                labels[item] = found[language]["value"]
                break
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/derived/wikidata-labels.json"))
    parser.add_argument("--languages", default="fr|en")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between batches")
    args = parser.parse_args()

    known: dict[str, str] = {}
    if args.output.exists():
        known = json.loads(args.output.read_text(encoding="utf-8"))
    wanted = sorted(ids_in(args.manifest) - set(known))
    print(f"{len(known)} labels already known, {len(wanted)} to fetch", file=sys.stderr)

    for start in range(0, len(wanted), 50):
        batch = wanted[start : start + 50]
        fetched = None
        for attempt in range(6):
            try:
                fetched = fetch_batch(batch, args.languages)
                break
            except Exception as error:
                # Wikidata answers 429 to a fast client; waiting is the whole remedy.
                wait = 20 * (attempt + 1)
                print(f"batch at {start} failed ({error}), retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
        if fetched is None:
            print(f"giving up at {start}; rerun later to resume", file=sys.stderr)
            break
        known.update(fetched)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(known, ensure_ascii=False, indent=0, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"{start + len(batch)}/{len(wanted)}", file=sys.stderr, end="\r")
        time.sleep(args.pause)
    print(f"\n{len(known)} labels in {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
