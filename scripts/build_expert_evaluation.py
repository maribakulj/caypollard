#!/usr/bin/env python3
"""Build a blinded rating package for the expert evaluation, plus its sealed key.

Phase 10 asks for method labels to be blind during evaluation, which is easy to
promise and easy to break: an expert shown three columns headed "visual",
"graph" and "fused" is not judging retrievals, they are judging a hypothesis.
So the three rankings are pooled into one shuffled list per query, the pooling
order is destroyed by a seeded shuffle, and the mapping from item back to method
and rank is written to a separate key file that the analysis reads and the rater
never sees.

Queries are stratified by how much the methods disagree, in three bands, so the
sample contains the cases where the system changes nothing as well as the cases
where it changes everything. Sampling only the disagreements would measure the
method at its most conspicuous rather than at its most typical.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from caypollard.provenance import read_jsonl

DIMENSIONS = (
    ("ressemblance_visuelle", "Les deux images se ressemblent-elles formellement ?"),
    ("parente_iconographique", "Représentent-elles le même sujet ou motif ?"),
    ("pertinence_historique", "Un lien historique ou contextuel les relie-t-il ?"),
    ("utilite_pour_la_recherche", "Ce rapprochement serait-il utile à une recherche ?"),
    ("confiance", "Quelle confiance accordez-vous à vos réponses ci-dessus ?"),
)
SCALE = (
    "1 = pas du tout · 2 = un peu · 3 = moyennement · 4 = beaucoup · 5 = tout à fait · "
    "? = ne peut juger"
)


def load_rankings(path: str) -> dict[str, list[dict[str, Any]]]:
    file = Path(path)
    if file.suffix == ".jsonl":
        rows = list(read_jsonl(file))
    else:
        rows = json.loads(file.read_text(encoding="utf-8"))["test_queries"]
    return {str(row["query_id"]): list(row["top_results"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="A named per-query ranking artifact, repeatable",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--queries-per-band", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    methods: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for spec in args.method:
        name, _, path = spec.partition("=")
        methods[name] = load_rankings(path)
    if len(methods) < 2:
        raise SystemExit("at least two methods are needed for a blinded comparison")

    records = {str(row["id"]): row for row in read_jsonl(args.manifest)}
    shared = sorted(set.intersection(*(set(ranking) for ranking in methods.values())))

    # Disagreement is the size of the union of the top-k sets relative to k: when
    # every method returns the same items the union is k, when they share nothing
    # it is k times the number of methods.
    #
    # Bands are the observed terciles of that ratio, not fixed thresholds. Fixed
    # thresholds were tried first and put 2 456 of 2 804 queries in one band,
    # because a graph ranking and a visual ranking share about a twelfth of their
    # top-10 and so almost never agree by an absolute standard. Terciles make the
    # sample span this corpus's own range of agreement rather than an imported one.
    ratios = {}
    for query_id in shared:
        pooled = {
            str(item["item_id"])
            for ranking in methods.values()
            for item in ranking[query_id][: args.k]
        }
        ratios[query_id] = len(pooled) / max(args.k, 1)

    ordered = sorted(shared, key=lambda query_id: (ratios[query_id], query_id))
    third = max(len(ordered) // 3, 1)
    banded: dict[str, list[str]] = {
        "accord_relatif": ordered[:third],
        "partiel": ordered[third : 2 * third],
        "desaccord": ordered[2 * third :],
    }

    rng = random.Random(args.seed)
    selected: list[tuple[str, str]] = []
    for band, ids in banded.items():
        rng.shuffle(ids)
        selected.extend((query_id, band) for query_id in ids[: args.queries_per_band])
    selected.sort()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    package: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for query_id, band in selected:
        pooled: dict[str, dict[str, int]] = {}
        for method, ranking in methods.items():
            for rank, item in enumerate(ranking[query_id][: args.k], start=1):
                pooled.setdefault(str(item["item_id"]), {})[method] = rank
        candidates = sorted(pooled)
        rng.shuffle(candidates)
        for position, item_id in enumerate(candidates, start=1):
            pair_id = f"{query_id}__{position:02d}"
            package.append(
                {
                    "pair_id": pair_id,
                    "query_id": query_id,
                    "candidate_id": item_id,
                    "query_image": records.get(query_id, {}).get("pictura_url")
                    or records.get(query_id, {}).get("image_url"),
                    "candidate_image": records.get(item_id, {}).get("pictura_url")
                    or records.get(item_id, {}).get("image_url"),
                    "query_motto": records.get(query_id, {}).get("motto"),
                    "candidate_motto": records.get(item_id, {}).get("motto"),
                }
            )
            key.append(
                {
                    "pair_id": pair_id,
                    "query_id": query_id,
                    "candidate_id": item_id,
                    "band": band,
                    "retrieved_by": pooled[item_id],
                }
            )

    (output / "package.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in package) + "\n", encoding="utf-8"
    )
    (output / "KEY-do-not-open.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in key) + "\n", encoding="utf-8"
    )

    sheet = output / "rating-sheet.csv"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "rater", *[name for name, _ in DIMENSIONS], "commentaire"])
        for row in package:
            writer.writerow([row["pair_id"], "", *[""] * len(DIMENSIONS), ""])

    manifest = {
        "methods": sorted(methods),
        "k": args.k,
        "seed": args.seed,
        "queries_per_band": args.queries_per_band,
        "bands": {band: len(ids) for band, ids in banded.items()},
        "band_definition": "observed terciles of the top-k union ratio",
        "union_ratio_range": [
            round(min(ratios.values()), 3),
            round(max(ratios.values()), 3),
        ],
        "queries_selected": len(selected),
        "pairs_to_rate": len(package),
        "dimensions": [{"name": name, "question": question} for name, question in DIMENSIONS],
        "scale": SCALE,
        "blinding": (
            "package.jsonl carries no method label and no rank. The mapping back to method and "
            "rank is in KEY-do-not-open.jsonl, which the analysis reads and the rater does not."
        ),
    }
    (output / "package-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "dimensions"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
