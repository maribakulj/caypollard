#!/usr/bin/env python3
"""Does the record match an object, a scene, or neither? Broken down by subject.

Every cross-medium figure so far pools pairs that share *any* Iconclass notation,
and a notation can be a cat or an Annunciation. "A painting of a cat finds an
engraving of a cat" has never actually been tested; what was tested is "two
pictures share a label", which is a much weaker and much vaguer claim.

Iconclass's first digit separates the kinds cleanly enough to ask the question.
Division 2 is Nature -- animals, plants, landscape -- where a notation names a
thing. Divisions 1 and 9 are Religion and Classical Mythology, where a notation
names an episode with several actors. Division 3 is the human body and its
actions, 4 society, 5 abstract ideas.

If a record matches objects and not scenes, the concrete divisions will rank far
better than the narrative ones, and the aggregate figure everything has been
reported on is an average over two different behaviours.
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
DIVISIONS = {
    "0": "0 · abstraction, art",
    "1": "1 · religion, magie (scènes)",
    "2": "2 · nature (objets : animaux, plantes)",
    "3": "3 · corps humain, action",
    "4": "4 · société, civilisation",
    "5": "5 · idées abstraites",
    "6": "6 · histoire",
    "7": "7 · Bible (scènes)",
    "8": "8 · littérature",
    "9": "9 · mythologie classique (scènes)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", action="append", required=True, metavar="NAME=MANIFEST")
    parser.add_argument("--table", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--hub-size", type=int, default=400)
    parser.add_argument("--min-pairs", type=int, default=25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    corpora = {}
    for spec in args.corpus:
        name, _, path = spec.partition("=")
        corpora[name] = {str(row["id"]): row for row in read_jsonl(path)}

    tables = {}
    for spec in args.table:
        name, _, path = spec.partition("=")
        tables[name] = load_embedding_table(path)

    first = next(iter(tables.values()))
    row_of = {item: index for index, item in enumerate(first.ids)}
    ids: list[str] = []
    corpus_of: dict[str, str] = {}
    signs: dict[str, set[str]] = {}
    for name, records in corpora.items():
        for item in sorted(records):
            labels = {BASE.split(str(x))[0].strip() for x in records[item].get("iconclass", [])}
            if item in row_of and labels:
                ids.append(item)
                corpus_of[item] = name
                signs[item] = labels
    position = {item: index for index, item in enumerate(ids)}

    frequency = collections.Counter(sign for item in ids for sign in signs[item])
    motifs = {sign for sign, count in frequency.items() if count <= args.hub_size}
    inverted: dict[str, set[str]] = collections.defaultdict(set)
    for item in ids:
        for sign in signs[item] & motifs:
            inverted[sign].add(item)

    # A pair is filed under the division of the notation that joins it, so a
    # picture carrying both a cat and a Nativity contributes to both rows.
    by_division: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for sign, members in inverted.items():
        if not sign or sign[0] not in DIVISIONS:
            continue
        for item in members:
            partners = [other for other in members if corpus_of[other] != corpus_of[item]]
            if partners:
                by_division[sign[0]].append((item, tuple(partners)))

    results: dict[str, dict[str, object]] = {}
    for name, table in tables.items():
        matrix = l2_normalize(
            np.stack([table.vectors[{i: n for n, i in enumerate(table.ids)}[item]] for item in ids])
        )
        per_division = {}
        for division, entries in sorted(by_division.items()):
            if len(entries) < args.min_pairs:
                continue
            ranks = []
            for item, partners in entries:
                index = position[item]
                scores = matrix[index] @ matrix.T
                scores[index] = -np.inf
                order = np.argsort(-scores)
                rank_of = {ids[int(j)]: rank for rank, j in enumerate(order, start=1)}
                ranks.append(min(rank_of[partner] for partner in partners))
            array = np.asarray(ranks)
            per_division[DIVISIONS[division]] = {
                "n_queries": len(entries),
                "median_rank": int(np.median(array)),
                "share_in_top_10": round(float((array <= 10).mean()), 4),
                "share_in_top_100": round(float((array <= 100).mean()), 4),
            }
            line = f"  {name:10s} {DIVISIONS[division]:36s} n={len(entries):5d}"
            line += f"  rang médian {int(np.median(array)):5d}"
            line += f"  top-100 {float((array <= 100).mean()):.1%}"
            print(line, flush=True)
        results[name] = per_division

    report = {
        "pool": len(ids),
        "hub_size": args.hub_size,
        "results": results,
        "reading": (
            "Divisions 2 and 4 name things; 1, 7 and 9 name episodes. A record that matches "
            "objects and not scenes separates them, and the aggregate figure reported "
            "elsewhere is then an average over two different behaviours."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
