#!/usr/bin/env python3
"""Turn the Rijksmuseum records into a benchmark in the schema the others use.

Everything downstream -- relevance, splitting, leakage checks, retrieval,
fusion -- already works on one manifest schema, so the transfer corpus is
written into that schema rather than given a pipeline of its own. What differs is
what plays the part of the volume: here it is the creator, because an artist's
works resemble one another the way plates of one book do, and a split that let
the same hand fall on both sides would leak.

Works whose image is missing or unreadable are dropped and counted. Near
duplicates are detected with the same two-stage perceptual hash as the Iconclass
benchmark, and folded into the grouping with creator and exact bytes.
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
    parser.add_argument("records", help="records.jsonl written by fetch_rijksmuseum.py")
    parser.add_argument("image_dir")
    parser.add_argument("--output-dir", default="data/derived/rijksmuseum-v0.1")
    parser.add_argument("--seed", default="rijksmuseum-v0.1")
    parser.add_argument(
        "--ratios",
        default="0.7,0.15,0.15",
        help="train,validation,test. Wider held-out splits than the 0.8/0.1/0.1 of the two "
             "larger benchmarks, because 1 864 items leave too few test queries otherwise.",
    )
    args = parser.parse_args()

    images = Path(args.image_dir)
    rows: list[dict[str, Any]] = []
    dropped: collections.Counter[str] = collections.Counter()
    for record in read_jsonl(args.records):
        path = images / str(record["filename"])
        if not path.is_file() or path.stat().st_size == 0:
            dropped["image absente"] += 1
            continue
        try:
            coarse = hash_image_file(path)
            fine = hash_image_file(path, side=16)
        except Exception as exc:
            dropped[f"illisible ({type(exc).__name__})"] += 1
            continue
        row = dict(record)
        row["sha256"] = sha256_file(path)
        row["phash"] = coarse
        row["phash_fine"] = fine
        # The creator stands in for the volume. A work with no creator recorded
        # becomes its own group rather than joining an "unknown artist" bucket
        # that would merge unrelated hands into one leakage unit.
        creators = record.get("creator") or []
        row["creator_group"] = creators[0] if creators else str(record["id"])
        rows.append(row)

    rows = assign_near_duplicate_groups(rows)
    rows = combine_group_keys(rows, keys=("creator_group", "sha256", "near_duplicate_group"))

    ratios = tuple(float(value) for value in args.ratios.split(","))
    if len(ratios) != 3:
        raise SystemExit("--ratios needs three comma-separated fractions")
    assigned = split_records(rows, group_key="group_id", seed=args.seed, ratios=ratios)
    leakage = find_group_leakage(assigned)
    checksum_leakage = find_checksum_leakage(assigned)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in assigned) + "\n",
        encoding="utf-8",
    )

    splits = collections.Counter(str(row.get("split")) for row in assigned)
    notations = collections.Counter(
        str(label) for row in assigned for label in row.get("iconclass", [])
    )
    audit = {
        "records_input": sum(splits.values()) + sum(dropped.values()),
        "records_kept": len(assigned),
        "dropped": dict(dropped),
        "splits": dict(splits),
        "distinct_notations": len(notations),
        "assignments": sum(notations.values()),
        "mean_notations_per_item": round(sum(notations.values()) / max(len(assigned), 1), 3),
        "most_common_notations": notations.most_common(15),
        "singleton_notations": sum(1 for count in notations.values() if count == 1),
        "distinct_creator_groups": len({row["creator_group"] for row in assigned}),
        "near_duplicates": near_duplicate_report(assigned),
        "group_leakage": {group: sorted(splits) for group, splits in leakage.items()},
        "checksum_leakage": {digest: sorted(s) for digest, s in checksum_leakage.items()},
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in audit.items() if k != "most_common_notations"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
