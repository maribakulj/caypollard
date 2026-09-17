#!/usr/bin/env python3
"""Fetch IIIF manifests for the BSB volumes present in a benchmark manifest.

The Iconclass AI filenames carry Munich digitisation identifiers (``bsb00024923``),
which resolve against the Münchener DigitalisierungsZentrum IIIF API. Those
manifests supply the creator, printer, place, and date that the filename-derived
``G1`` projection lacks, and — crucially — GND authority URIs, so two volumes by
one author link through an identifier rather than a spelling.

Raw manifests are cached to disk and parsed separately, so the network step runs
once and every later change to the parser is replayable offline. Requests are
serialised with a delay: this is a few hundred lookups against a public research
API, and it should stay that way.
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

MANIFEST_URL = "https://api.digitale-sammlungen.de/iiif/presentation/v2/{volume}/manifest"
USER_AGENT = "caypollard-research/0.1 (cultural heritage retrieval research)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Canonical benchmark manifest JSONL")
    parser.add_argument("--cache-dir", default="data/raw/bsb-manifests")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests.")
    parser.add_argument("--limit", type=int, help="Fetch at most N new volumes.")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    volumes = sorted(
        {
            str(row["book_id"])
            for row in read_jsonl(args.manifest)
            if str(row.get("book_id") or "").startswith("bsb")
            and not str(row["book_id"]).startswith("bsbink")
        }
    )

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    pending = [v for v in volumes if not (cache / f"{v}.json").is_file()]
    if args.limit:
        pending = pending[: args.limit]

    print(f"{len(volumes)} volumes, {len(pending)} à récupérer", flush=True)
    fetched = 0
    failures: dict[str, str] = {}

    for index, volume in enumerate(pending, start=1):
        request = urllib.request.Request(
            MANIFEST_URL.format(volume=volume), headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                payload = response.read()
            json.loads(payload)  # reject a truncated body before it reaches the cache
            (cache / f"{volume}.json").write_bytes(payload)
            fetched += 1
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            failures[volume] = f"{type(exc).__name__}: {exc}"
        if index % 25 == 0 or index == len(pending):
            print(f"  {index}/{len(pending)}  ok={fetched}  échecs={len(failures)}", flush=True)
        time.sleep(args.delay)

    provenance = {
        "source": "Münchener DigitalisierungsZentrum IIIF Presentation API v2",
        "url_template": MANIFEST_URL,
        "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
        "n_volumes_requested": len(volumes),
        "n_volumes_cached": sum(1 for v in volumes if (cache / f"{v}.json").is_file()),
        "failures": failures,
        "request_delay_seconds": args.delay,
    }
    (cache / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in provenance.items() if k != "failures"}, indent=2))
    if failures:
        print(f"{len(failures)} échecs, premiers : {list(failures)[:5]}")


if __name__ == "__main__":
    main()
