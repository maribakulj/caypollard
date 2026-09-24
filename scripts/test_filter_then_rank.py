#!/usr/bin/env python3
"""Query by image alone: the picture's own names choose the field, pixels rank it.

An earlier version of this measurement cheated, and the cheat is worth keeping in
view because it was invisible until someone asked the obvious question. It picked
the filter word from the *query's Iconclass notation* -- that is, from the answer
-- and filtered the corpus on it. That is retrieval by typing a word, and by
typing the right one.

The question is image to image: drop in any picture of a cat and get cats back,
on canvas, in stone, on a vase, without typing anything. So the filter comes from
the query's own transcription, which is what a system would actually have, and
the annotation is never consulted.

Shared names are weighted by rarity, because they must be: two pictures sharing
``man`` share almost nothing, two sharing ``windmill`` share a great deal, and
unweighted overlap fills the field with whatever the namer says most. Weighting
moves the cat from 0.11 to 0.19 and the ship from 1.59 to 1.91, which is most of
what the weighting can do.

It is not enough. Against the embedding used alone the hybrid wins three motifs
of six, ties one and loses two, and the earlier version's clean gains were the
cheat. The reason is measurable rather than mysterious: the query's own
transcription finds a cat 41% of the time at best and is right 7 to 9% of the
time it says so, so the field it selects is wrong about as often as it is right.
Whether better recall would rescue this is a prediction the record can carry --
the vocabulary has holes, the whole-picture namer returns three nodes and misses
what is small -- rather than a hope.

``--oracle`` restores the leaky version for comparison, and is never the result.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
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
    parser.add_argument("--field", type=int, default=500,
                        help="How many candidates the names hand to the ranker")
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Filter on the word for the query's own notation instead of on the "
             "query's transcription. This reads the answer and is kept only to show "
             "what the leak was worth; it is never the result.",
    )
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

    document_frequency = collections.Counter(n for i in ids for n in named[i])
    idf = {
        n: math.log(len(ids) / (1 + c)) for n, c in document_frequency.items()
    }
    holders_of_name: dict[str, set[str]] = collections.defaultdict(set)
    for item in ids:
        for name in named[item]:
            holders_of_name[name].add(item)

    report = {
        "pool": len(ids),
        "k": args.k,
        "oracle": args.oracle,
        "field": args.field,
        "motifs": {},
    }
    for code, word in MOTIFS.items():
        queries = [i for i in ids if code in notations(records[i])]
        if len(queries) < args.min_queries:
            continue
        oracle_field = [i for i in ids if word in named[i]]
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
            if args.oracle:
                candidates = [c for c in oracle_field if c != query]
            else:
                # Everything sharing a name with the query, scored by how rare the
                # shared names are: two pictures sharing "man" share almost nothing.
                weighted: dict[str, float] = collections.defaultdict(float)
                for name in named[query]:
                    weight = idf.get(name, 0.0)
                    for other in holders_of_name[name]:
                        if other != query:
                            weighted[other] += weight
                candidates = [
                    o for o, _ in sorted(weighted.items(), key=lambda kv: -kv[1])
                ][: args.field]
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
            "oracle_field_size": len(oracle_field),
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
        print(
            f"   {word:8s} pixels {value['ranker_alone']:5.2f}  →  filtré "
            f"{value['filter_then_rank']:5.2f}  ({gain:+.2f})"
        )


if __name__ == "__main__":
    main()
