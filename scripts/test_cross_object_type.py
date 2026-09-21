#!/usr/bin/env python3
"""Does a vase find an engraving of the same subject? Retrieval across object kinds.

Every cross-medium figure so far compared one corpus of prints against one of
museum works, which is a coarse contrast: both sides are largely flat images of
flat things. The question the record was built for is sharper. A motif that
travels moves between a painting, a printed page, a plate, a figurine and a
relief, and those differ not only in how they were made but in what a photograph
of them even shows -- a vase is curved, lit from one side, and photographed
against a plain ground.

Object kinds are read from Wikidata's `instance of` and grouped into families
that share a viewing condition rather than a technique:

``plat``
    painting, print, drawing, watercolour, woodcut, photograph -- a flat surface
    seen frontally.
``volume``
    sculpture, figurine, relief, statue -- a thing in the round, lit and shadowed.
``recipient``
    vase, plate, bowl, jug, cup, saucer, lekythos -- a curved surface whose
    decoration is partly hidden by its own form.

A pair counts when the two objects share a non-hub Iconclass notation and belong
to different families. If the record crosses that, it has crossed something
harder than print against painting.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import l2_normalize, load_embedding_table
from caypollard.provenance import read_jsonl

BASE = re.compile(r"[(\[]")

FAMILIES = {
    "plat": {
        "Q3305213",  # peinture
        "Q11060274",  # estampe
        "Q93184",  # dessin
        "Q18761202",  # aquarelle
        "Q18218093",  # eau-forte
        "Q11835431",  # gravure
        "Q18219090",  # xylogravure
        "Q125191",  # photographie
        "Q12043905",  # pastel
    },
    "volume": {
        "Q860861",  # sculpture
        "Q245117",  # relief
        "Q17489160",  # figurine
        "Q179700",  # statue
        "Q1935974",  # buste
    },
    "recipient": {
        "Q191851",  # vase
        "Q13417114",  # assiette
        "Q153988",  # bol
        "Q1049146",  # cruche
        "Q81727",  # tasse
        "Q1030963",  # soucoupe
        "Q1935452",  # lécythe
        "Q2712308",  # amphore
    },
}


def family_of(types: list[str]) -> str | None:
    for name, members in FAMILIES.items():
        if any(t in members for t in types):
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+")
    parser.add_argument("--table", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--hub-size", type=int, default=400)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records: dict[str, dict] = {}
    for path in args.manifest:
        for row in read_jsonl(path):
            records[str(row["id"])] = row

    tables = {}
    for spec in args.table:
        name, _, path = spec.partition("=")
        tables[name] = load_embedding_table(path)

    first = next(iter(tables.values()))
    covered = set(first.ids)
    for table in tables.values():
        covered &= set(table.ids)

    family: dict[str, str] = {}
    signs: dict[str, set[str]] = {}
    for item, row in records.items():
        if item not in covered:
            continue
        kind = family_of([str(t) for t in (row.get("type") or [])])
        labels = {BASE.split(str(x))[0].strip() for x in row.get("iconclass", [])}
        if kind and labels:
            family[item] = kind
            signs[item] = labels
    ids = sorted(family)
    if len(ids) < 50:
        raise SystemExit("too few items carry a recognised object family")
    position = {item: index for index, item in enumerate(ids)}

    frequency = collections.Counter(sign for item in ids for sign in signs[item])
    motifs = {sign for sign, count in frequency.items() if count <= args.hub_size}
    inverted: dict[str, set[str]] = collections.defaultdict(set)
    for item in ids:
        for sign in signs[item] & motifs:
            inverted[sign].add(item)

    targets: dict[str, set[str]] = {}
    pairs_by_families: collections.Counter[str] = collections.Counter()
    for item in ids:
        found: set[str] = set()
        for sign in signs[item] & motifs:
            for other in inverted[sign]:
                if family[other] != family[item]:
                    found.add(other)
                    pairs_by_families["+".join(sorted((family[item], family[other])))] += 1
        if found:
            targets[item] = found

    composition = collections.Counter(family[item] for item in ids)
    results = {}
    for name, table in tables.items():
        row_of = {item: index for index, item in enumerate(table.ids)}
        matrix = l2_normalize(np.stack([table.vectors[row_of[item]] for item in ids]))
        ranks = []
        crossing = 0
        for query, partners in targets.items():
            index = position[query]
            scores = matrix[index] @ matrix.T
            scores[index] = -np.inf
            order = np.argsort(-scores)
            rank_of = {ids[int(j)]: rank for rank, j in enumerate(order, start=1)}
            ranks.append(min(rank_of[partner] for partner in partners))
            crossing += sum(
                1 for j in order[:10] if family[ids[int(j)]] != family[query]
            )
        array = np.asarray(ranks)
        results[name] = {
            "queries": len(targets),
            "median_rank": int(np.median(array)),
            "share_in_top_10": round(float((array <= 10).mean()), 4),
            "share_in_top_100": round(float((array <= 100).mean()), 4),
            "share_of_top_10_from_another_family": round(crossing / (len(targets) * 10), 4),
        }
        entry = results[name]
        line = f"  {name:22s} rang médian {entry['median_rank']:5d}"
        line += f"  top-100 {entry['share_in_top_100']:.1%}"
        line += f"  voisins d'une autre famille {entry['share_of_top_10_from_another_family']:.1%}"
        print(line, flush=True)

    report = {
        "pool": len(ids),
        "chance_rank": len(ids) // 2,
        "families": dict(composition),
        "queries_with_a_cross_family_partner": len(targets),
        "pairs_by_family_couple": dict(pairs_by_families.most_common()),
        "hub_size": args.hub_size,
        "results": results,
        "reading": (
            "Families group viewing conditions rather than techniques: a flat surface seen "
            "frontally, a thing in the round lit and shadowed, and a curved vessel whose "
            "decoration its own form partly hides. Crossing those is harder than crossing "
            "print against painting, which this benchmark had been calling cross-medium."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
