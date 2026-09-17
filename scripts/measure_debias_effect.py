#!/usr/bin/env python3
"""Does projecting out a confound actually change what the model groups together?

Nullspace projection makes a linear probe unable to read an attribute, and it is
tempting to report that as the attribute having been removed. It is not the same
claim. A property can be encoded redundantly across a space -- paper texture,
typography, ink density, page layout all covary with the holding library -- so a
linear readout can be destroyed while the geometry that readout was summarising
stays put.

This script measures the difference: how much the top-k neighbourhoods actually
move, and whether the composition of those neighbourhoods changes. A projection
that leaves 90% of neighbours in place has not removed a confound; it has hidden
it from one kind of classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import EmbeddingTable, l2_normalize, load_embedding_table
from caypollard.provenance import read_jsonl
from caypollard.retrieval import neighbor_overlap_at_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--attribute", action="append", default=["collection", "book_id"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = [row for row in read_jsonl(args.manifest) if row.get("split") == args.split]
    by_id = {str(row["id"]): row for row in records}
    ids = sorted(by_id)

    def restricted(path: str) -> EmbeddingTable:
        table = load_embedding_table(path)
        row_of = {item: index for index, item in enumerate(table.ids)}
        return EmbeddingTable(
            ids=tuple(ids),
            vectors=l2_normalize(np.stack([table.vectors[row_of[item]] for item in ids])),
            metadata={"family": "vision"},
        )

    before, after = restricted(args.before), restricted(args.after)
    overlap, _ = neighbor_overlap_at_k(before, after, k=args.k)

    def composition(table: EmbeddingTable) -> dict[str, float]:
        shares = dict.fromkeys(args.attribute, 0)
        total = 0
        for index, query in enumerate(ids):
            scores = table.vectors[index] @ table.vectors.T
            scores[index] = -np.inf
            for neighbour in np.argsort(-scores)[: args.k]:
                candidate = ids[int(neighbour)]
                for attribute in args.attribute:
                    shares[attribute] += int(
                        by_id[query].get(attribute) is not None
                        and by_id[query].get(attribute) == by_id[candidate].get(attribute)
                    )
                total += 1
        return {name: round(value / max(total, 1), 4) for name, value in shares.items()}

    report = {
        "manifest": args.manifest,
        "before": args.before,
        "after": args.after,
        "k": args.k,
        "n_items": len(ids),
        "neighbour_overlap_before_after": round(float(overlap), 4),
        "neighbourhood_composition": {
            "before": composition(before),
            "after": composition(after),
        },
        "reading": (
            "An overlap near 1 means the projection changed almost nothing about which "
            "items are close, whatever it did to a linear probe's accuracy. The attribute "
            "shares say the same thing from the other side: if neighbours are still drawn "
            "from the query's own collection at the same rate, the confound is still "
            "shaping the geometry and has only been made unreadable by a linear model."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
