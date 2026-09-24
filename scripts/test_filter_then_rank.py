#!/usr/bin/env python3
"""Neither channel alone: the name chooses the field, the pixels rank inside it.

The two representations fail in opposite directions, and the failures are
complementary rather than competing. A photographic embedding ranks beautifully
and selects badly: asked for a cat, sixty-five per cent of what it returns shares
the query's *object kind* and thirty-seven per cent its *collection*, while eight
per cent share the cat. It is a reader of the object and the institution, and the
motif rides along. A named record selects and cannot rank: it says ``cat`` of two
hundred and thirty pictures, many of them wrongly, but the field it picks out is
enormously enriched in cats compared to the corpus.

So one does what the other cannot. Restrict to the pictures whose named record
carries the motif, then order that field by the embedding. Measured on what the
whole exercise was built for -- the same motif on a *different kind of object* --
against the embedding used alone.

Recall is what matters in the filter and precision is not, which is worth
stating because it contradicts the instinct. The named channel is wrong about
the motif most of the time it speaks, and filtering on it still doubles or
triples the cross-kind hits, because a loose filter that keeps the right
pictures is worth more than a tight one that drops them. The same measurement
appears from the other side elsewhere in this record: the high-precision
intersection of two namers loses to either namer alone.

The failure case is the instructive one. A word the namer emits on four hundred
pictures filters nothing, and there the hybrid loses.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import load_embedding_table

BASE = re.compile(r"[(\[]")
MOTIFS = {
    "34B12": "cat", "46C21": "ship", "25F6": "fish", "25F36": "bird",
    "46C112": "bridge", "47I4223": "grape", "25F711": "insect",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+")
    parser.add_argument("--nodes", action="append", required=True,
                        help="A namer's jsonl; repeatable, and the union is used, "
                             "because recall is what a filter needs")
    parser.add_argument("--ranker", required=True, help="The embedding that orders the field")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-queries", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records: dict[str, dict] = {}
    for path in args.manifest:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                records[str(row["id"])] = row

    def notations(row: dict) -> set[str]:
        return {BASE.split(str(x))[0].strip() for x in row.get("iconclass", [])}

    def kind(row: dict) -> str:
        value = row.get("type")
        if isinstance(value, list):
            return str(value[0]) if value else "?"
        return str(value) if value else "?"

    named: dict[str, set[str]] = {}
    for path in args.nodes:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                named.setdefault(str(row["id"]), set()).update(
                    n["name"] for n in row.get("nodes") or []
                )

    table = load_embedding_table(args.ranker)
    row_of = {item: index for index, item in enumerate(table.ids)}
    ids = [i for i in table.ids if i in records and i in named]
    matrix = np.stack([table.vectors[row_of[i]] for i in ids]).astype(np.float64)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    position = {item: index for index, item in enumerate(ids)}

    report = {"pool": len(ids), "k": args.k, "motifs": {}}
    for code, word in MOTIFS.items():
        queries = [i for i in ids if code in notations(records[i])]
        if len(queries) < args.min_queries:
            continue
        field = [i for i in ids if word in named[i]]
        alone, hybrid = [], []
        for query in queries:
            index = position[query]
            scores = matrix[index] @ matrix.T
            scores[index] = -np.inf
            top = [ids[int(j)] for j in np.argsort(-scores)[: args.k]]
            alone.append(
                sum(
                    1 for o in top
                    if code in notations(records[o])
                    and kind(records[o]) != kind(records[query])
                )
            )
            candidates = [c for c in field if c != query]
            if candidates:
                rows = np.asarray([position[c] for c in candidates])
                inner = matrix[index] @ matrix[rows].T
                top = [candidates[int(j)] for j in np.argsort(-inner)[: args.k]]
                hybrid.append(
                    sum(
                        1 for o in top
                        if code in notations(records[o])
                        and kind(records[o]) != kind(records[query])
                    )
                )
            else:
                hybrid.append(0)
        report["motifs"][word] = {
            "queries": len(queries),
            "field_size": len(field),
            "ranker_alone": round(statistics.mean(alone), 3),
            "filter_then_rank": round(statistics.mean(hybrid), 3),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"bassin {len(ids)}, cible = même motif sur un AUTRE type d'objet, top-{args.k}\n")
    for word, value in report["motifs"].items():
        gain = value["filter_then_rank"] - value["ranker_alone"]
        print(f"   {word:8s} pixels {value['ranker_alone']:5.2f}  →  filtré "
              f"{value['filter_then_rank']:5.2f}  ({gain:+.2f})   champ {value['field_size']}")


if __name__ == "__main__":
    main()
