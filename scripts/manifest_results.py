#!/usr/bin/env python3
"""Attest every result artifact by digest, including the ones git does not carry.

Large artifacts -- embedding dumps, per-query rankings -- are regenerable and too
big to version, but a claim that rests on them needs a way to check that the file
on disk is the file the claim was computed from. This writes one line per artifact:
size, SHA-256, and whether the bytes are versioned or have to be rebuilt.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

VERSIONED_MAX_BYTES = 2_000_000


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="results")
    parser.add_argument("--output", default="results/MANIFEST.sha256")
    parser.add_argument("--max-versioned-bytes", type=int, default=VERSIONED_MAX_BYTES)
    args = parser.parse_args()

    output = Path(args.output)
    rows = []
    for path in sorted(Path(args.root).rglob("*")):
        if not path.is_file() or path == output:
            continue
        size = path.stat().st_size
        state = "versioned" if size <= args.max_versioned_bytes else "rebuild"
        rows.append(f"{digest(path)}  {size:>10}  {state:<9}  {path.as_posix()}")

    output.write_text(
        "# Result artifacts, one per line: sha256, bytes, git state, path.\n"
        "# 'rebuild' files exceed the versioning threshold and are reproduced by\n"
        "# rerunning the script that wrote them; their digest is the check.\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    versioned = sum(1 for row in rows if " versioned " in row)
    print(f"{len(rows)} artifacts, {versioned} versioned, {len(rows) - versioned} attested only")


if __name__ == "__main__":
    main()
