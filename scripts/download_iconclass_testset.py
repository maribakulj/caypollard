#!/usr/bin/env python3
"""Explicit downloader for the ~3.1 GB Iconclass AI Test Set archive.

The archive is intentionally *not* downloaded by setup, tests, or notebooks.
Large external research data should require a deliberate user action.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from caypollard.datasets.iconclass import (
    ICONCLASS_TESTSET_ARCHIVE_URL,
    ICONCLASS_TESTSET_MD5,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/iconclass-ai-testset.zip")
    parser.add_argument(
        "--accept-large-download",
        action="store_true",
        help="Required safety flag: the official archive is approximately 3.1 GB.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.accept_large_download:
        raise SystemExit(
            "Refusing the ~3.1 GB download without --accept-large-download. "
            "See docs/ICONCLASS_DATA_CARD.md first."
        )

    destination = Path(args.output)
    if destination.exists() and not args.force:
        raise SystemExit(f"{destination} already exists; pass --force to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)

    md5 = hashlib.md5(usedforsecurity=False)
    request = urllib.request.Request(
        ICONCLASS_TESTSET_ARCHIVE_URL,
        headers={"User-Agent": "caypollard"},
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as target:
        while chunk := response.read(8 * 1024 * 1024):
            md5.update(chunk)
            target.write(chunk)

    digest = md5.hexdigest()
    if digest != ICONCLASS_TESTSET_MD5:
        destination.unlink(missing_ok=True)
        raise SystemExit(
            f"MD5 mismatch: expected {ICONCLASS_TESTSET_MD5}, got {digest}. "
            "Deleted incomplete/unverified archive."
        )
    print(f"Verified {destination} ({digest})")


if __name__ == "__main__":
    main()
