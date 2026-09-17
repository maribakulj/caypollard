#!/usr/bin/env python3
"""Does a grammar add anything to a lexicon, or is a bag of typed signs enough?

An isotype is a closed vocabulary *and* a grammar: a lion beneath a crown is not
a lion wearing one. Building a transcriber that recovers relations is far harder
than building one that recovers signs, so the question is worth answering before
the architecture is chosen rather than after.

Measured predictively, and therefore non-circularly. One notation is hidden from
each item; the item is represented by what remains; its neighbours vote the
hidden one back. Neighbours from the query's own volume are excluded, because
another plate of the same book carries the answer for reasons that have nothing
to do with the representation. The frequency prior runs alongside, since a corpus
whose commonest sign appears on hundreds of items rewards a system that has
learned only what is common.

Three representations, each a strictly richer reading of the same annotation:

``base``
    Notations stripped to their bare hierarchy code. A lexicon and nothing else.
``qualified``
    Iconclass's own modifiers kept as distinct signs: ``25G3(+361)`` and
    ``25G3`` are different words. 16.6% of this corpus carries such a key, and
    the project has been discarding them.
``pairs``
    ``base`` plus every co-occurring pair of signs as its own feature. The
    crudest possible grammar -- "these two appear together" -- with no spatial
    relation, since the corpus records none.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from caypollard.provenance import read_jsonl

BASE = re.compile(r"[(\[]")


def strip(label: str) -> str:
    return BASE.split(str(label))[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3, help="Truncation for pair features")
    parser.add_argument(
        "--hide",
        choices=("rarest", "random"),
        default="rarest",
        help="Which sign to hide. 'rarest' is the informative choice but floors every "
             "method near zero, since a rare sign has few neighbours carrying it; "
             "'random' leaves enough signal to rank the representations against "
             "each other.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = [row for row in read_jsonl(args.manifest) if row.get("split") == args.split]
    by_id = {str(row["id"]): row for row in records}
    ids = sorted(by_id)
    notations = {item: [str(x) for x in by_id[item].get("iconclass", [])] for item in ids}
    usable = [item for item in ids if len({strip(x) for x in notations[item]}) >= 2]

    # The hidden sign is the rarest one the item carries: hiding a sign shared by
    # hundreds of emblems would be recovered by the prior alone and would measure
    # nothing.
    frequency = collections.Counter(strip(x) for item in ids for x in notations[item])
    if args.hide == "rarest":
        hidden = {
            item: min({strip(x) for x in notations[item]}, key=lambda s: (frequency[s], s))
            for item in usable
        }
    else:
        generator = np.random.default_rng(args.seed)
        hidden = {}
        for item in usable:
            signs = sorted({strip(x) for x in notations[item]})
            hidden[item] = signs[int(generator.integers(0, len(signs)))]
    prior = [label for label, _ in frequency.most_common()]

    def features(item: str, *, representation: str, drop: str | None) -> set[str]:
        labels = [x for x in notations[item] if strip(x) != drop]
        base = {strip(x) for x in labels}
        if representation == "base":
            return base
        if representation == "qualified":
            return base | {x for x in labels if x != strip(x)}
        truncated = sorted({s[: args.depth] for s in base})
        conjunctions = {
            f"{a}&{b}" for index, a in enumerate(truncated) for b in truncated[index + 1 :]
        }
        return base | conjunctions

    same_volume = {
        item: {other for other in ids if by_id[other].get("book_id") == by_id[item].get("book_id")}
        for item in usable
    }

    results = {}
    for representation in ("base", "qualified", "pairs"):
        vocabulary: dict[str, int] = {}
        candidate_sets = {}
        for item in ids:
            signs = features(item, representation=representation, drop=None)
            candidate_sets[item] = signs
            for sign in signs:
                vocabulary.setdefault(sign, len(vocabulary))

        matrix = np.zeros((len(ids), len(vocabulary)), dtype=np.float32)
        for index, item in enumerate(ids):
            for sign in candidate_sets[item]:
                matrix[index, vocabulary[sign]] = 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.where(norms > 0, norms, 1.0)
        position = {item: index for index, item in enumerate(ids)}

        hits1 = hits10 = 0
        reciprocal = []
        for item in usable:
            query = np.zeros(len(vocabulary), dtype=np.float32)
            for sign in features(item, representation=representation, drop=hidden[item]):
                if sign in vocabulary:
                    query[vocabulary[sign]] = 1.0
            norm = np.linalg.norm(query)
            if norm == 0:
                reciprocal.append(0.0)
                continue
            scores = matrix @ (query / norm)
            scores[position[item]] = -np.inf
            for other in same_volume[item]:
                scores[position[other]] = -np.inf
            tally: collections.Counter[str] = collections.Counter()
            for neighbour in np.argsort(-scores)[: args.k]:
                weight = float(scores[int(neighbour)])
                if not np.isfinite(weight) or weight <= 0:
                    continue
                for label in notations[ids[int(neighbour)]]:
                    tally[strip(label)] += weight
            for sign in {strip(x) for x in notations[item]} - {hidden[item]}:
                tally.pop(sign, None)
            ranked = [label for label, _ in tally.most_common()]
            hits1 += int(bool(ranked[:1]) and ranked[0] == hidden[item])
            hits10 += int(hidden[item] in ranked[: args.top])
            rank = next((p for p, label in enumerate(ranked, 1) if label == hidden[item]), None)
            reciprocal.append(1.0 / rank if rank else 0.0)

        results[representation] = {
            "vocabulary": len(vocabulary),
            "n_queries": len(usable),
            "hits_at_1": round(hits1 / len(usable), 4),
            "hits_at_10": round(hits10 / len(usable), 4),
            "mrr": round(float(np.mean(reciprocal)), 4),
        }
        entry = results[representation]
        print(f"  {representation:10s} vocab {entry['vocabulary']:6d}  "
              f"hits@10 {entry['hits_at_10']:.3f}  MRR {entry['mrr']:.3f}", flush=True)

    prior_hits = sum(
        int(hidden[item] in [s for s in prior if s not in {strip(x) for x in notations[item]}
                             or s == hidden[item]][: args.top])
        for item in usable
    )
    results["prior de fréquence"] = {
        "hits_at_10": round(prior_hits / len(usable), 4),
        "n_queries": len(usable),
    }
    print(f"  {'prior':10s} {'':13s} hits@10 {results['prior de fréquence']['hits_at_10']:.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "manifest": args.manifest,
                "k": args.k,
                "hidden_sign": args.hide,
                "same_volume_neighbours": "excluded",
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
