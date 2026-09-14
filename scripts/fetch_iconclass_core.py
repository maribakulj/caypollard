#!/usr/bin/env python3
"""Download a pinned Iconclass ``notations.txt`` hierarchy source."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.request
from pathlib import Path

from caypollard.provenance import sha256_file

DEFAULT_REF = "0aeced694cf0a57dd5ff5dfb4587da99f698b686"
RAW_TEMPLATE = "https://raw.githubusercontent.com/iconclass/data/{ref}/notations.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=DEFAULT_REF, help="Git commit/tag/branch to pin")
    parser.add_argument("--output", default="data/raw/iconclass/notations.txt")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = Path(args.output)
    if destination.exists() and not args.force:
        raise SystemExit(f"{destination} already exists; pass --force to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)

    url = RAW_TEMPLATE.format(ref=args.ref)
    request = urllib.request.Request(url, headers={"User-Agent": "caypollard"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)

    provenance = {
        "source_repository": "https://github.com/iconclass/data",
        "source_url": url,
        "ref": args.ref,
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sha256": sha256_file(destination),
        "license": "CC0-1.0",
    }
    provenance_path = destination.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(destination)
    print(provenance_path)


if __name__ == "__main__":
    main()
