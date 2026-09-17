#!/usr/bin/env python3
"""Ask a representation what it still knows about a confound, two ways.

A linear probe is the usual referee and it is not a sufficient one. Projecting the
holding-library direction out of a DINOv2 space took a logistic regression from
94.8% to 43.7% while leaving 91% of the top-10 neighbourhoods in place: the
readout was destroyed, the geometry was not. Reporting only the linear number
would have called that a success.

So two probes run here. The linear one says whether the attribute can be read off
a direction. The k-nearest-neighbour one says whether the attribute still governs
who sits next to whom -- which is the only thing retrieval ever uses. When the two
disagree, the second is the one that matters.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from caypollard.embeddings.store import l2_normalize, load_embedding_table
from caypollard.provenance import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument(
        "--table", action="append", required=True, metavar="NAME=PATH",
        help="A named embedding table to probe, repeatable",
    )
    parser.add_argument("--attribute", default="collection")
    parser.add_argument("--neighbourhood-attribute", action="append",
                        default=["collection", "book_id"])
    parser.add_argument("--split", help="Restrict to one split; default is every record")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = [
        row
        for row in read_jsonl(args.manifest)
        if args.split is None or row.get("split") == args.split
    ]
    by_id = {str(row["id"]): row for row in records}

    results = {}
    for spec in args.table:
        name, _, path = spec.partition("=")
        table = load_embedding_table(path)
        row_of = {item: index for index, item in enumerate(table.ids)}
        ids = [item for item in sorted(by_id) if item in row_of]
        matrix = l2_normalize(np.stack([table.vectors[row_of[item]] for item in ids]))
        labels = np.array([str(by_id[item].get(args.attribute)) for item in ids])
        counts = collections.Counter(labels)
        majority = counts.most_common(1)[0][1] / len(labels)

        linear = float(
            cross_val_score(
                LogisticRegression(max_iter=1500), matrix, labels, cv=args.folds, n_jobs=2
            ).mean()
        )
        neighbour = float(
            cross_val_score(
                KNeighborsClassifier(n_neighbors=args.k, metric="cosine"),
                matrix,
                labels,
                cv=args.folds,
                n_jobs=2,
            ).mean()
        )

        shares = dict.fromkeys(args.neighbourhood_attribute, 0)
        total = 0
        for index, query in enumerate(ids):
            scores = matrix[index] @ matrix.T
            scores[index] = -np.inf
            for position in np.argsort(-scores)[: args.k]:
                candidate = ids[int(position)]
                for attribute in args.neighbourhood_attribute:
                    value = by_id[query].get(attribute)
                    shares[attribute] += int(
                        value is not None and value == by_id[candidate].get(attribute)
                    )
                total += 1

        results[name] = {
            "n_items": len(ids),
            "majority_baseline": round(majority, 4),
            "linear_probe_accuracy": round(linear, 4),
            "knn_probe_accuracy": round(neighbour, 4),
            "linear_above_majority": round(linear - majority, 4),
            "knn_above_majority": round(neighbour - majority, 4),
            "neighbourhood_share": {
                attribute: round(value / max(total, 1), 4) for attribute, value in shares.items()
            },
        }
        entry = results[name]
        print(
            f"  {name:26s} linéaire {entry['linear_probe_accuracy']:.1%}  "
            f"kNN {entry['knn_probe_accuracy']:.1%}  (majorité {majority:.1%})  "
            f"voisins même {args.attribute} {entry['neighbourhood_share'][args.attribute]:.1%}",
            flush=True,
        )

    report = {
        "manifest": args.manifest,
        "attribute": args.attribute,
        "split": args.split,
        "k": args.k,
        "results": results,
        "reading": (
            "A rendering has removed the confound only when the kNN probe falls to the "
            "majority baseline and the neighbourhood share falls with it. A linear probe "
            "dropping alone means the attribute was made unreadable, not absent."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
