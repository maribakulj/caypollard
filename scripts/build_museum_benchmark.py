#!/usr/bin/env python3
"""Merge several Wikidata collections into one frozen benchmark.

The transfer and naming tests have been running on 1 864 objects from a single
museum against 2 723 emblem prints, and the binding constraint showed itself
plainly: only twelve Iconclass notations had enough support in both corpora to be
learnable, which is too narrow a base to conclude anything about whether a label
crosses a medium. Merging eight collections widens that base.

Merging is not concatenation. Three things have to be handled or the benchmark
lies about itself:

* an object can sit in two collections' query results, so identity is by
  Wikidata QID and duplicates are counted, not silently kept;
* leakage groups are the union of creator, exact bytes and confirmed near
  duplicate, closed transitively, and the creator now spans institutions -- the
  same painter's works in the Met and the Louvre must not straddle a split;
* institutions differ in what they hold, so the collection is recorded on every
  record and is the stratum every medium probe is run against.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from caypollard.datasets.near_duplicates import (
    assign_near_duplicate_groups,
    hash_image_file,
    near_duplicate_report,
)
from caypollard.provenance import read_jsonl, sha256_file
from caypollard.splitting import (
    combine_group_keys,
    find_checksum_leakage,
    find_group_leakage,
    split_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", action="append", required=True, metavar="NAME=DIR",
        help="A collection's fetch directory, holding records.jsonl and images/",
    )
    parser.add_argument("--output-dir", default="data/derived/museums-v0.1")
    parser.add_argument("--seed", default="museums-v0.1")
    parser.add_argument("--ratios", default="0.7,0.15,0.15")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    dropped: collections.Counter[str] = collections.Counter()
    seen: dict[str, str] = {}
    for spec in args.source:
        name, _, directory = spec.partition("=")
        base = Path(directory)
        for record in read_jsonl(base / "records.jsonl"):
            item = str(record["id"])
            if item in seen:
                dropped[f"doublon (déjà dans {seen[item]})"] += 1
                continue
            path = base / "images" / str(record["filename"])
            if not path.is_file() or path.stat().st_size == 0:
                dropped["image absente"] += 1
                continue
            try:
                coarse = hash_image_file(path)
                fine = hash_image_file(path, side=16)
            except Exception as exc:
                dropped[f"illisible ({type(exc).__name__})"] += 1
                continue
            seen[item] = name
            row = dict(record)
            row["collection"] = name
            row["image_path"] = str(path)
            row["sha256"] = sha256_file(path)
            row["phash"] = coarse
            row["phash_fine"] = fine
            creators = record.get("creator") or []
            row["creator_group"] = creators[0] if creators else item
            rows.append(row)

    if not rows:
        raise SystemExit("no usable record in any source")

    rows = assign_near_duplicate_groups(rows)
    rows = combine_group_keys(rows, keys=("creator_group", "sha256", "near_duplicate_group"))
    ratios = tuple(float(value) for value in args.ratios.split(","))
    assigned = split_records(rows, group_key="group_id", seed=args.seed, ratios=ratios)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in assigned) + "\n",
        encoding="utf-8",
    )

    notations = collections.Counter(
        str(label) for row in assigned for label in row.get("iconclass", [])
    )
    per_collection = collections.Counter(str(row["collection"]) for row in assigned)
    audit = {
        "records_kept": len(assigned),
        "dropped": dict(dropped),
        "per_collection": dict(per_collection.most_common()),
        "splits": dict(collections.Counter(str(row.get("split")) for row in assigned)),
        "distinct_notations": len(notations),
        "assignments": sum(notations.values()),
        "mean_notations_per_item": round(sum(notations.values()) / len(assigned), 3),
        "notations_on_at_least_25_items": sum(1 for c in notations.values() if c >= 25),
        "distinct_creator_groups": len({row["creator_group"] for row in assigned}),
        "near_duplicates": near_duplicate_report(assigned),
        "group_leakage": {g: sorted(s) for g, s in find_group_leakage(assigned).items()},
        "checksum_leakage": {d: sorted(s) for d, s in find_checksum_leakage(assigned).items()},
        "manifest_sha256": sha256_file(manifest),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
