#!/usr/bin/env python3
"""Compare what each modality retrieves for the same query, three ways.

Phase 7 asks for visual, graph and fused neighbours to be compared for emblem
queries. Overlap at 10 answers it directly: two rankings that agree on most of
their top ten are two views of one thing, and a fusion that agrees with its
visual arm everywhere has changed nothing.

The figure to read against is the overlap *between two visual encoders*, which
is how much two honest views of the same evidence agree.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import EmbeddingTable, l2_normalize, load_embedding_table
from caypollard.provenance import read_jsonl
from caypollard.retrieval import neighbor_overlap_at_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument(
        "--table",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="A named embedding table, repeatable",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tables = {}
    for spec in args.table:
        name, _, path = spec.partition("=")
        tables[name] = load_embedding_table(path)

    records = read_jsonl(args.manifest)
    common: set[str] | None = None
    for table in tables.values():
        ids = set(table.ids)
        common = ids if common is None else common & ids
    assert common is not None
    ids = sorted(
        str(row["id"])
        for row in records
        if str(row.get("split")) == args.split and str(row["id"]) in common
    )
    if len(ids) < args.k + 1:
        raise SystemExit("not enough aligned items in the requested split")

    # Restrict every table to the same items and the same order, so an overlap
    # figure compares neighbourhoods drawn from one candidate pool rather than
    # from whatever each modality happens to cover.
    matrices = {}
    for name, table in tables.items():
        row_of = {item: index for index, item in enumerate(table.ids)}
        matrices[name] = EmbeddingTable(
            ids=tuple(ids),
            vectors=l2_normalize(np.stack([table.vectors[row_of[item]] for item in ids])),
            metadata={"family": "restricted", "source": name},
        )

    overlaps = {}
    for left, right in itertools.combinations(sorted(matrices), 2):
        mean_overlap, _per_query = neighbor_overlap_at_k(
            matrices[left], matrices[right], k=args.k
        )
        overlaps[f"{left} ↔ {right}"] = round(float(mean_overlap), 4)

    report = {
        "label": args.label,
        "split": args.split,
        "k": args.k,
        "n_items": len(ids),
        "modalities": sorted(matrices),
        "mean_overlap_at_k": overlaps,
        "reading": (
            "Overlap between two visual encoders is the reference: it is how much two "
            "honest views of the same evidence agree. A modality pair well below it is "
            "seeing something different; a fused ranking well above its own visual arm "
            "has barely moved."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
