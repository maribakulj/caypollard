#!/usr/bin/env python3
"""Can the system propose an annotation it was never shown? The non-circular test.

Retrieving items by the notation they carry measures the index, not the method:
a pair found *because* it shares a label says only that the label is there. The
question that matters for filling gaps in a catalogue is the opposite one — can a
missing annotation be predicted from everything except itself?

Two settings, both realistic populations in this corpus:

``partial``
    The item keeps all its notations but one, which is hidden and must be
    recovered. This is a catalogue entry that was filled in incompletely.
``unindexed``
    Every notation of the item is hidden. This is the 5 578 Emblematica records
    that carry no Iconclass at all, and it is the harder and more useful case.

Prediction is deliberately the simplest thing that could work: take the k nearest
neighbours in some representation and let them vote on notations, weighted by
similarity. A learned model would be premature — if neighbour voting cannot beat
the frequency prior, nothing built on the same representation will.

The frequency prior is the baseline that matters. A corpus where 500 items carry
the same notation can be scored well by a system that has learned nothing but
which labels are common, so that system is run alongside and reported.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import l2_normalize, load_embedding_table
from caypollard.provenance import read_jsonl


def average_precision(ranked: list[str], truth: set[str]) -> float:
    hits = 0
    total = 0.0
    for position, label in enumerate(ranked, start=1):
        if label in truth:
            hits += 1
            total += hits / position
    return total / len(truth) if truth else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument(
        "--table",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="A named representation to find neighbours in, repeatable",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--exclude-same-volume",
        action="store_true",
        help="Drop neighbours from the query's own volume. Other plates of one book "
             "share notations for reasons that have nothing to do with the method, "
             "so this is the setting that tests transfer between unrelated objects.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = [row for row in read_jsonl(args.manifest) if row.get("split") == args.split]
    by_id = {str(row["id"]): row for row in records}
    ids = sorted(by_id)
    labels = {item: {str(x) for x in by_id[item].get("iconclass", [])} for item in ids}
    usable = [item for item in ids if labels[item]]
    index = {item: position for position, item in enumerate(ids)}

    frequency = collections.Counter(label for item in ids for label in labels[item])
    prior = [label for label, _ in frequency.most_common()]

    same_volume = np.array(
        [[by_id[a].get("book_id") == by_id[b].get("book_id") for b in ids] for a in ids]
    )

    # A modality with partial coverage -- mottoes reach 94% of emblems -- would
    # otherwise be dropped or, worse, scored on a different query set from the
    # others. Every table is restricted to the items all of them carry, so the
    # comparison is on one population.
    loaded = {}
    for spec in args.table:
        name, _, path = spec.partition("=")
        table = load_embedding_table(path)
        loaded[name] = (table, {item: position for position, item in enumerate(table.ids)})
    covered = set(ids)
    for _table, row_of in loaded.values():
        covered &= set(row_of)
    dropped_for_coverage = len(ids) - len(covered)
    ids = [item for item in ids if item in covered]
    usable = [item for item in usable if item in covered]
    index = {item: position for position, item in enumerate(ids)}
    same_volume = np.array(
        [[by_id[a].get("book_id") == by_id[b].get("book_id") for b in ids] for a in ids]
    )
    frequency = collections.Counter(label for item in ids for label in labels[item])
    prior = [label for label, _ in frequency.most_common()]

    tables = {
        name: l2_normalize(np.stack([table.vectors[row_of[item]] for item in ids]))
        for name, (table, row_of) in loaded.items()
    }

    def vote(matrix: np.ndarray, query: str, *, visible: set[str]) -> list[str]:
        position = index[query]
        scores = matrix[position] @ matrix.T
        scores[position] = -np.inf
        if args.exclude_same_volume:
            scores[same_volume[position]] = -np.inf
        neighbours = np.argsort(-scores)[: args.k]
        tally: collections.Counter[str] = collections.Counter()
        for neighbour in neighbours:
            weight = float(scores[int(neighbour)])
            if not np.isfinite(weight):
                continue
            for label in labels[ids[int(neighbour)]]:
                tally[label] += max(weight, 0.0)
        # A notation the item already carries is not a prediction; in the partial
        # setting those are visible to the system and must not be scored.
        for label in visible:
            tally.pop(label, None)
        return [label for label, _ in tally.most_common()]

    results: dict[str, dict[str, dict[str, float]]] = {}
    for setting in ("unindexed", "partial"):
        per_method: dict[str, dict[str, float]] = {}
        candidates = (
            usable if setting == "unindexed" else [i for i in usable if len(labels[i]) >= 2]
        )
        for name, matrix in [*tables.items(), ("prior de fréquence", None)]:
            hits1 = hits5 = hits10 = 0
            reciprocal = []
            precisions = []
            for query in candidates:
                truth = set(labels[query])
                visible: set[str] = set()
                if setting == "partial":
                    hidden = sorted(truth)[0]
                    visible = truth - {hidden}
                    truth = {hidden}
                ranked = (
                    [label for label in prior if label not in visible]
                    if matrix is None
                    else vote(matrix, query, visible=visible)
                )
                top = ranked[: args.top]
                hits1 += int(bool(set(top[:1]) & truth))
                hits5 += int(bool(set(top[:5]) & truth))
                hits10 += int(bool(set(top) & truth))
                rank = next((p for p, label in enumerate(ranked, 1) if label in truth), None)
                reciprocal.append(1.0 / rank if rank else 0.0)
                precisions.append(average_precision(top, truth))
            n = max(len(candidates), 1)
            per_method[name] = {
                "n_queries": len(candidates),
                "hits_at_1": round(hits1 / n, 4),
                "hits_at_5": round(hits5 / n, 4),
                "hits_at_10": round(hits10 / n, 4),
                "mrr": round(float(np.mean(reciprocal)), 4),
                "map_at_10": round(float(np.mean(precisions)), 4),
            }
            print(f"  {setting:10s} {name:22s} hits@10={per_method[name]['hits_at_10']:.3f} "
                  f"MRR={per_method[name]['mrr']:.3f}", flush=True)
        results[setting] = per_method

    report = {
        "manifest": args.manifest,
        "split": args.split,
        "k_neighbours": args.k,
        "top": args.top,
        "exclude_same_volume": args.exclude_same_volume,
        "n_items": len(ids),
        "n_items_dropped_for_coverage": dropped_for_coverage,
        "n_items_with_a_notation": len(usable),
        "distinct_notations": len(frequency),
        "results": results,
        "reading": (
            "The frequency prior is the number to beat: a representation that does not "
            "clear it has learned nothing but which labels are common. 'unindexed' hides "
            "every notation and is the population that matters -- 5 578 emblems in this "
            "corpus carry none at all."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
