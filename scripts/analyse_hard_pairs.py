#!/usr/bin/env python3
"""Test H2 and H3 on a mined hard-pair artifact, and report the confounds first.

The order matters. An AUC computed on pairs whose visual distance is really a
difference of scanning format measures the format, so this script reports the
composition of each pair class — which collections, which volumes, what the two
items share — before it reports any score, and prints the format check whether or
not the artifact was mined with a stratum.

H2 is read as hard positives against easy negatives: both are visually distant,
so only iconography separates them. H3 is hard positives against hard negatives.
The visual score is not a baseline for either — pair classes are *defined* by
visual thresholds, so a visual AUC near 0 is a tautology, printed only to make
that circularity visible rather than to be compared against.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import load_embedding_table
from caypollard.provenance import read_jsonl


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    a, b = np.array(positive), np.array(negative)
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", help="Hard-pair JSONL artifact")
    parser.add_argument("manifest", help="Manifest the pairs were mined from")
    parser.add_argument(
        "--score",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="A scoring embedding table, repeatable: --score graph=path.npz",
    )
    parser.add_argument("--stratum-key", default="collection")
    parser.add_argument("--output")
    args = parser.parse_args()

    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    records = {str(row["id"]): row for row in read_jsonl(args.manifest)}
    by_class: dict[str, list[dict]] = collections.defaultdict(list)
    for pair in pairs:
        by_class[pair["pair_class"]].append(pair)

    report: dict[str, object] = {"pairs_file": args.pairs, "n_pairs": len(pairs)}

    composition = {}
    for name, group in sorted(by_class.items()):
        strata = collections.Counter(
            tuple(sorted((
                str(records[p["query_id"]].get(args.stratum_key)),
                str(records[p["candidate_id"]].get(args.stratum_key)),
            )))
            for p in group
        )
        cross = sum(count for key, count in strata.items() if key[0] != key[1])
        shared_label = sum(
            1 for p in group
            if set(p["query_labels"]) & set(p["candidate_labels"])
        )
        composition[name] = {
            "n": len(group),
            "cross_stratum": cross,
            "cross_stratum_fraction": cross / len(group) if group else 0.0,
            "pairs_sharing_an_exact_notation": shared_label,
            "top_strata": [f"{a}+{b}:{c}" for (a, b), c in strata.most_common(3)],
        }
    report["composition"] = composition

    scores: dict[str, dict[str, list[float]]] = {}
    for spec in args.score:
        name, _, path = spec.partition("=")
        table = load_embedding_table(path)
        index = {item: i for i, item in enumerate(table.ids)}
        vectors = table.vectors / np.linalg.norm(table.vectors, axis=1, keepdims=True)
        scores[name] = {
            cls: [
                float(vectors[index[p["query_id"]]] @ vectors[index[p["candidate_id"]]])
                for p in group
                if p["query_id"] in index and p["candidate_id"] in index
            ]
            for cls, group in by_class.items()
        }

    report["auc"] = {
        name: {
            "H2_hard_positive_vs_easy_negative": _auc(
                per_class.get("hard_positive", []), per_class.get("easy_negative", [])
            ),
            "H3_hard_positive_vs_hard_negative": _auc(
                per_class.get("hard_positive", []), per_class.get("hard_negative", [])
            ),
            "control_easy_positive_vs_easy_negative": _auc(
                per_class.get("easy_positive", []), per_class.get("easy_negative", [])
            ),
        }
        for name, per_class in scores.items()
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
