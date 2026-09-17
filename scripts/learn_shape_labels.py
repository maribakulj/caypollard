#!/usr/bin/env python3
"""Learn to name a picture's shapes, and test whether the name crosses the medium.

Clustering cannot name. Atomic signs sit barely above chance on iconographic
coherence and composite ones not even that: a clump of mid-sized dark patches is
drapery in one picture and foliage in another, and geometry cannot tell them
apart. A name has to come from recognition.

In an evaluation, learning that recognition from Iconclass would be leakage --
the annotation is the ground truth. In a discovery system nothing is held out,
so the existing annotation is legitimate supervision, and a label learned this
way is contestable in the way a correlation is not: it claims that this picture
shows this, and a historian can say no.

The measurement that matters is not accuracy within a corpus. It is whether a
label learned on printed emblems applies to painted panels. Trained on one corpus
and tested on the other, a classifier that transfers has learned the motif; one
that does not has learned the workshop. Both directions are reported, and the
frequency prior runs alongside, because a corpus whose commonest notation covers
a tenth of it rewards a model that has learned only what is common.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from caypollard.embeddings.store import load_embedding_table
from caypollard.provenance import read_jsonl

BASE = re.compile(r"[(\[]")


def evaluate(scores: np.ndarray, truth: list[set[str]], vocabulary: list[str], k: int) -> dict:
    hits = 0
    reciprocal = []
    for row, actual in zip(scores, truth, strict=True):
        ranked = [vocabulary[i] for i in np.argsort(-row)]
        if set(ranked[:k]) & actual:
            hits += 1
        rank = next((p for p, label in enumerate(ranked, 1) if label in actual), None)
        reciprocal.append(1.0 / rank if rank else 0.0)
    return {
        "hits_at_k": round(hits / max(len(truth), 1), 4),
        "mrr": round(float(np.mean(reciprocal)), 4),
        "n": len(truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", help="The representation to learn from")
    parser.add_argument("--corpus", action="append", required=True, metavar="NAME=MANIFEST")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-support", type=int, default=25,
                        help="A notation needs this many examples in each corpus to be learnable")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    corpora: dict[str, dict[str, set[str]]] = {}
    for spec in args.corpus:
        name, _, path = spec.partition("=")
        corpora[name] = {
            str(row["id"]): {BASE.split(str(x))[0].strip() for x in row.get("iconclass", [])}
            for row in read_jsonl(path)
        }

    table = load_embedding_table(args.table)
    row_of = {item: index for index, item in enumerate(table.ids)}

    # A notation is learnable only where both corpora can show it; otherwise a
    # transfer result would measure which notations exist where, not whether a
    # shape means the same thing on paper and on panel.
    counts = {
        name: collections.Counter(
            notation
            for item, labels in members.items()
            if item in row_of
            for notation in labels
        )
        for name, members in corpora.items()
    }
    supported = [
        {notation for notation, count in counter.items() if count >= args.min_support}
        for counter in counts.values()
    ]
    vocabulary = sorted(set.intersection(*supported))
    if not vocabulary:
        raise SystemExit("no notation has enough support in every corpus")

    data = {}
    for name, members in corpora.items():
        known = set(vocabulary)
        ids = [
            item
            for item in sorted(members)
            if item in row_of and members[item] & known
        ]
        data[name] = {
            "X": np.stack([table.vectors[row_of[item]] for item in ids]),
            "y": [members[item] & set(vocabulary) for item in ids],
            "ids": ids,
        }

    def fit(train: str) -> np.ndarray | None:
        models = []
        for notation in vocabulary:
            target = np.array([notation in labels for labels in data[train]["y"]])
            if target.sum() < 5 or (~target).sum() < 5:
                models.append(None)
                continue
            models.append(
                LogisticRegression(max_iter=1200, class_weight="balanced").fit(
                    data[train]["X"], target
                )
            )
        return models

    results = {}
    names = sorted(corpora)
    for train in names:
        models = fit(train)
        for test in names:
            scores = np.zeros((len(data[test]["ids"]), len(vocabulary)), dtype=float)
            for index, model in enumerate(models):
                if model is None:
                    continue
                scores[:, index] = model.predict_proba(data[test]["X"])[:, 1]
            key = f"{train} → {test}"
            results[key] = evaluate(scores, data[test]["y"], vocabulary, args.k)

    prior_counts = collections.Counter(
        notation for name in names for labels in data[name]["y"] for notation in labels
    )
    order = [notation for notation, _ in prior_counts.most_common()]
    for test in names:
        ranked = np.tile(
            np.array([-order.index(n) for n in vocabulary], dtype=float),
            (len(data[test]["ids"]), 1),
        )
        results[f"prior → {test}"] = evaluate(ranked, data[test]["y"], vocabulary, args.k)

    report = {
        "table": args.table,
        "k": args.k,
        "vocabulary": len(vocabulary),
        "min_support": args.min_support,
        "items_per_corpus": {name: len(data[name]["ids"]) for name in names},
        "results": results,
        "reading": (
            "The diagonal is learning within a corpus; the off-diagonal is whether the label "
            "crosses the medium. A model that transfers has learned the motif, one that does "
            "not has learned the workshop. The prior is what a model knowing only which "
            "notations are common would score."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for key, value in results.items():
        line = f"  {key:26s} hits@{args.k} {value['hits_at_k']:.3f}"
        print(f"{line}  MRR {value['mrr']:.3f}  n={value['n']}")


if __name__ == "__main__":
    main()
