#!/usr/bin/env python3
"""Fetch Wolfenbüttel METS records for the HAB shelfmarks in a benchmark manifest.

Unlike the Illinois catalogue, which is keyed by an internal book id we do not
have, HAB digitisations resolve directly from the shelfmark carried in the
filename: ``embhab_1224-20-theol`` is ``drucke/1224-20-theol``. The METS record
supplies a Dublin Core bibliographic citation of the form
``Author: Title - Place : Printer, Year``, plus a stable OPAC URI.

HAB matters here because the hard-positive pairs are emblem-book pairs, and
Wolfenbüttel is one of the emblem collections they are drawn from. Records are
cached so parsing stays replayable offline, and requests are serialised.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from caypollard.provenance import read_jsonl

METS_URL = "https://diglib.hab.de/{path}/mets.xml"
USER_AGENT = "caypollard-research/0.1 (cultural heritage retrieval research)"


def shelfmark_paths(book_id: str) -> list[str]:
    """Return candidate WDB paths for a filename-derived HAB identifier.

    The prefix is dropped and the remainder is the shelfmark. Some shelfmarks lose
    a hyphen in the filename (``xb2867`` for ``xb-2867``), so a hyphenated variant
    is tried as well. Both ``drucke`` and ``mss`` divisions are attempted because
    the filename does not say which one a shelfmark belongs to.
    """
    mark = book_id.split("_", 1)[1] if "_" in book_id else book_id
    variants = [mark]
    if "-" not in mark:
        for index, char in enumerate(mark):
            if char.isdigit() and index:
                variants.append(f"{mark[:index]}-{mark[index:]}")
                break
    return [f"{division}/{variant}" for variant in dict.fromkeys(variants)
            for division in ("drucke", "mss")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Canonical benchmark manifest JSONL")
    parser.add_argument("--cache-dir", default="data/raw/hab-mets")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    books = sorted(
        {
            str(row["book_id"])
            for row in read_jsonl(args.manifest)
            if str(row.get("book_id") or "").startswith(("embhab_", "hab_"))
        }
    )

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    pending = [b for b in books if not (cache / f"{b}.xml").is_file()]
    if args.limit:
        pending = pending[: args.limit]

    print(f"{len(books)} cotes HAB, {len(pending)} à récupérer", flush=True)
    resolved: dict[str, str] = {}
    failures: list[str] = []

    for index, book in enumerate(pending, start=1):
        for path in shelfmark_paths(book):
            request = urllib.request.Request(
                METS_URL.format(path=path), headers={"User-Agent": USER_AGENT}
            )
            try:
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    payload = response.read()
            except (urllib.error.URLError, TimeoutError):
                time.sleep(args.delay)
                continue
            if b"bibliographicCitation" not in payload:
                time.sleep(args.delay)
                continue
            (cache / f"{book}.xml").write_bytes(payload)
            resolved[book] = path
            break
        else:
            failures.append(book)
        if index % 20 == 0 or index == len(pending):
            print(f"  {index}/{len(pending)}  ok={len(resolved)}  échecs={len(failures)}",
                  flush=True)
        time.sleep(args.delay)

    provenance_path = cache / "provenance.json"
    previous = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {}
    )
    provenance = {
        "source": "Wolfenbütteler Digitale Bibliothek METS",
        "url_template": METS_URL,
        "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
        "n_books_requested": len(books),
        "n_books_cached": sum(1 for b in books if (cache / f"{b}.xml").is_file()),
        "resolved_paths": {**previous.get("resolved_paths", {}), **resolved},
        "unresolved": failures,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in provenance.items()
                      if k not in ("resolved_paths", "unresolved")}, indent=2))
    if failures:
        print(f"{len(failures)} non résolues, premières : {failures[:5]}")


if __name__ == "__main__":
    main()
