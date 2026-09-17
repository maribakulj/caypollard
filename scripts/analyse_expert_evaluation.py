#!/usr/bin/env python3
"""Read filled rating sheets, unseal the key, and answer phase 10's question.

The question is not whether experts agree — that is a precondition — but whether
a metric improvement corresponds to useful scholarly retrieval. So the analysis
joins each rated pair back to the method that retrieved it and to the graded
relevance the automatic metric assigned, and reports:

* agreement per dimension, never collapsed into one score;
* the human rating per method, which is the blind comparison;
* rank correlation between each human dimension and the automatic relevance;
* the pairs raters disagreed on most, as items.

A dimension may be left unrated. Ratings are read as ordinal, so any cell that is
not one of 1-5 is treated as "cannot judge" and excluded rather than coerced to a
number.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from caypollard.human_evaluation import (
    compare_with_metric,
    disagreements,
    summarise_dimensions,
)
from caypollard.provenance import read_jsonl

VALID = {"1", "2", "3", "4", "5"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings", help="Filled rating-sheet CSV, or several concatenated")
    parser.add_argument("key", help="KEY-do-not-open.jsonl from the package")
    parser.add_argument("--relevance", help="Per-query artifact supplying graded relevance")
    parser.add_argument("--disagreement-threshold", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    key = {str(row["pair_id"]): row for row in read_jsonl(args.key)}

    dimensions: set[str] = set()
    judgements: dict[str, dict[str, dict[str, float]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    unrated = 0
    with open(args.ratings, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair_id = (row.get("pair_id") or "").strip()
            rater = (row.get("rater") or "").strip()
            if not pair_id or not rater or pair_id not in key:
                continue
            for column, value in row.items():
                if column in {"pair_id", "rater", "commentaire"} or column is None:
                    continue
                cleaned = (value or "").strip()
                if cleaned not in VALID:
                    unrated += 1
                    continue
                dimensions.add(column)
                judgements[column][pair_id][rater] = float(cleaned)

    if not judgements:
        raise SystemExit("no usable rating found; is the sheet filled and the rater column set?")

    per_dimension = summarise_dimensions(judgements)

    # The blind comparison. A pair may have been retrieved by several methods, so
    # a rating counts once for each method that returned it; that is the correct
    # reading of a pooled design, not double counting.
    by_method: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for dimension, units in judgements.items():
        for pair_id, ratings in units.items():
            mean = sum(ratings.values()) / len(ratings)
            for method in key[pair_id].get("retrieved_by", {}):
                by_method[dimension][method].append(mean)
    method_scores = {
        dimension: {
            method: {"n": len(values), "mean": round(sum(values) / len(values), 3)}
            for method, values in sorted(methods.items())
        }
        for dimension, methods in by_method.items()
    }

    relevance: dict[str, float] = {}
    if args.relevance:
        file = Path(args.relevance)
        rows = (
            list(read_jsonl(file))
            if file.suffix == ".jsonl"
            else json.loads(file.read_text(encoding="utf-8"))["test_queries"]
        )
        graded = {
            (str(row["query_id"]), str(item["item_id"])): float(
                item.get("hierarchical_relevance", 0.0)
            )
            for row in rows
            for item in row["top_results"]
        }
        for pair_id, entry in key.items():
            value = graded.get((str(entry["query_id"]), str(entry["candidate_id"])))
            if value is not None:
                relevance[pair_id] = value

    against_metric = {}
    if relevance:
        for dimension, units in judgements.items():
            human = {
                pair_id: sum(ratings.values()) / len(ratings)
                for pair_id, ratings in units.items()
            }
            against_metric[dimension] = compare_with_metric(human, relevance)

    bands: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for dimension, units in judgements.items():
        grouped: dict[str, list[float]] = collections.defaultdict(list)
        for pair_id, ratings in units.items():
            grouped[str(key[pair_id].get("band"))].append(sum(ratings.values()) / len(ratings))
        bands[dimension] = {
            band: round(sum(values) / len(values), 3) for band, values in sorted(grouped.items())
        }

    report = {
        "ratings_file": args.ratings,
        "n_pairs_rated": len({p for units in judgements.values() for p in units}),
        "n_raters": len({r for units in judgements.values() for u in units.values() for r in u}),
        "n_cells_not_rated": unrated,
        "agreement_per_dimension": per_dimension,
        "mean_rating_per_method": method_scores,
        "human_versus_automatic_relevance": against_metric,
        "mean_rating_per_disagreement_band": dict(bands),
        "largest_disagreements": {
            dimension: disagreements(units, threshold=args.disagreement_threshold, limit=10)
            for dimension, units in judgements.items()
        },
        "reading": (
            "Dimensions are never collapsed. A method's mean rating is the blind comparison; "
            "tau_b against graded relevance says whether the automatic metric tracks the "
            "judgement; the band breakdown says whether the answer depends on how much the "
            "methods disagreed in the first place."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "largest_disagreements"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
