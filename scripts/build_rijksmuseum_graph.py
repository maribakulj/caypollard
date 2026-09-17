#!/usr/bin/env python3
"""Project the Rijksmuseum context graph, in the shape that worked on Emblematica.

The lesson from Emblematica was that a graph joining items through a hub node --
a volume holding ninety emblems -- gives every item its own neighbours and
nothing else, because the hub's degree drowns the relations that cross it. The
fix was to project the hub's attributes onto the items and drop the hub. The same
shape applies here with the creator in place of the volume.

Two variants are written so the comparison is available rather than assumed:

``creator``
    Items linked to their creator node, the hub-shaped graph.
``pure``
    Creator, period, material, genre and object type attached to each item
    directly, with no creator node to route through.

Nothing derived from ``depicts`` enters either graph: those statements are the
relevance ground truth, and putting them in the graph would be target leakage.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from caypollard.provenance import read_jsonl

# Iconclass notations come from depicts, so depicts is ground truth and must not
# appear in any projection. Every field below is bibliographic or material.
ATTRIBUTE_FIELDS = {
    "creator": "created_by",
    "material": "made_of",
    "genre": "genre",
    "type": "instance_of",
}


def decade(value: str | None) -> str | None:
    if not value or not value[:4].isdigit():
        return None
    return f"decade:{value[:3]}0s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", default="data/derived/rijksmuseum-v0.1/variants")
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    creator_triples: list[tuple[str, str, str]] = []
    pure_triples: list[tuple[str, str, str]] = []
    counts: collections.Counter[str] = collections.Counter()

    for row in records:
        item = str(row["id"])
        for field, predicate in ATTRIBUTE_FIELDS.items():
            for value in row.get(field) or ():
                node = f"{field}:{value}"
                pure_triples.append((item, predicate, node))
                counts[predicate] += 1
                if field == "creator":
                    creator_triples.append((item, predicate, node))
        period = decade(row.get("inception"))
        if period:
            pure_triples.append((item, "made_in", period))
            counts["made_in"] += 1

    # The hub-shaped variant additionally links every work of one creator to that
    # creator node and nothing more, which is the construction that failed before.
    by_creator: dict[str, list[str]] = collections.defaultdict(list)
    for item, _predicate, node in creator_triples:
        by_creator[node].append(item)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, triples in (("creator", creator_triples), ("pure", pure_triples)):
        path = output / f"{name}.tsv"
        path.write_text(
            "\n".join("\t".join(triple) for triple in sorted(set(triples))) + "\n",
            encoding="utf-8",
        )

    report = {
        "manifest": args.manifest,
        "n_items": len(records),
        "triples_creator": len(set(creator_triples)),
        "triples_pure": len(set(pure_triples)),
        "predicate_counts": dict(counts),
        "distinct_creators": len(by_creator),
        "largest_creator_group": max((len(v) for v in by_creator.values()), default=0),
        "items_without_any_attribute": sum(
            1
            for row in records
            if not any(row.get(field) for field in ATTRIBUTE_FIELDS)
            and not decade(row.get("inception"))
        ),
        "note": "depicts is ground truth and appears in no projection",
    }
    (output.parent / "graph-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
