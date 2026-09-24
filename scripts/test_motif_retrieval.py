#!/usr/bin/env python3
"""Does a painted cat find a carved one? The question asked literally.

Every other benchmark here aggregates, and aggregates are hard to believe. This
one asks the question anybody would ask first: take the pictures carrying a
concrete motif -- a cat, a ship, a dragon, a windmill -- put each in turn to the
representation, and count how many of the ten pictures it returns carry the same
motif. Then count how many of those are a *different kind of object*, because a
painted cat finding another painted cat is not what an intermediate record was
built for.

Two floors, as everywhere. What ten neighbours drawn at random would return, and
how many partners of another kind exist at all -- since a ratio is meaningless
if the corpus holds three of them. It holds sixty-three per query here, and
essentially every same-motif partner is of another kind, so scarcity explains
nothing and the measurement is about the representation.

Motifs are chosen for being things rather than scenes, and for occurring on many
kinds of object. A notation naming an episode would measure something else, and
the record already reports separately that it fails there.
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
    "34B12": "chat", "46C21": "navire", "25FF411": "dragon", "25F6": "poisson",
    "47I2111": "taureau", "25F36": "oiseau d'eau", "47I213": "mouton",
    "47D31": "moulin", "46C112": "pont", "25F711": "insecte",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+")
    parser.add_argument("--table", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--k", type=int, default=10)
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

    tables = {}
    for spec in args.table:
        name, _, path = spec.partition("=")
        tables[name] = load_embedding_table(path)

    ids = sorted(set.intersection(*(set(t.ids) for t in tables.values())) & set(records))
    position = {item: index for index, item in enumerate(ids)}
    queries = [
        (item, notation)
        for item in ids
        for notation in notations(records[item]) & set(MOTIFS)
    ]
    if not queries:
        raise SystemExit("no picture carries one of the motifs")

    holders = {
        notation: [i for i in ids if notation in notations(records[i])]
        for notation in MOTIFS
    }
    available, available_cross = [], []
    for item, notation in queries:
        same = [j for j in holders[notation] if j != item]
        available.append(len(same))
        available_cross.append(
            sum(1 for j in same if kind(records[j]) != kind(records[item]))
        )
    floor = args.k * statistics.mean(available) / len(ids)
    floor_cross = args.k * statistics.mean(available_cross) / len(ids)

    report = {
        "pool": len(ids),
        "queries": len(queries),
        "motifs": MOTIFS,
        "k": args.k,
        "partners_available_median": statistics.median(available),
        "of_another_kind_median": statistics.median(available_cross),
        "floor_same_motif": round(floor, 3),
        "floor_another_kind": round(floor_cross, 3),
        "representations": {},
    }
    for name, table in tables.items():
        row_of = {item: index for index, item in enumerate(table.ids)}
        matrix = np.stack([table.vectors[row_of[i]] for i in ids]).astype(np.float64)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        found, crossed = [], []
        for item, notation in queries:
            index = position[item]
            scores = matrix[index] @ matrix.T
            scores[index] = -np.inf
            top = np.argsort(-scores)[: args.k]
            hits = [
                ids[int(j)] for j in top if notation in notations(records[ids[int(j)]])
            ]
            found.append(len(hits))
            crossed.append(
                sum(1 for h in hits if kind(records[h]) != kind(records[item]))
            )
        report["representations"][name] = {
            "same_motif_per_k": round(statistics.mean(found), 3),
            "ratio_to_floor": round(statistics.mean(found) / max(floor, 1e-9), 1),
            "another_kind_per_k": round(statistics.mean(crossed), 3),
            "ratio_to_floor_another_kind": round(
                statistics.mean(crossed) / max(floor_cross, 1e-9), 1
            ),
        }
    report["reading"] = (
        "The second pair of numbers is the one the record was built for: a painted cat "
        "finding another painted cat is retrieval by object kind wearing the clothes of "
        "iconography."
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"bassin {len(ids)}, {len(queries)} requêtes, plancher {floor:.2f}/{args.k} "
          f"et {floor_cross:.2f}/{args.k} pour un autre type")
    for name, value in sorted(
        report["representations"].items(), key=lambda kv: -kv[1]["ratio_to_floor"]
    ):
        print(f"   {name:16s} même motif {value['same_motif_per_k']:.2f}/{args.k} "
              f"(x{value['ratio_to_floor']:.0f})   autre type "
              f"{value['another_kind_per_k']:.2f} (x{value['ratio_to_floor_another_kind']:.0f})")


if __name__ == "__main__":
    main()
