#!/usr/bin/env python3
"""What is a representation actually keyed to? One question per regime.

A picture is several things at once and they are usually confounded. A Greek
vase painted with dancers is a *subject* (people dancing), an *object kind* (a
curved vessel whose own form arranges its decoration), a *period* (its century's
conventions), and a *museum photograph* (that institution's lighting, ground and
framing). Asking whether a representation "crosses the medium" bundles all four
and answers about the bundle.

This asks them one at a time. A regime names one attribute as the target and
*requires the others to differ*, so a partner can only be found for the reason
the regime is about. The subject regime wants the same Iconclass notation on a
different kind of object, from a different collection, of a different century --
the vase with dancers finding a modern painting of dancers rather than a vase
with athletes. The object-kind regime wants the same kind of thing sharing no
notation at all, and so on. Four regimes, one pool, one table, and what each
representation is keyed to becomes readable rather than assumed.

Two things make the numbers comparable, and both had to be got right.

The pool is identical across regimes, so a rank means the same thing in each.
Only the target set changes.

The floor is computed per query and not taken as half the pool. A query with one
partner in twelve thousand and a query with a thousand partners do not face the
same chance: under a random ordering the expected rank of the *best* of k
targets in a pool of n is (n+1)/(k+1), which is a thousandfold difference
between the tightest regime and the loosest. Reporting one floor for both would
make the easy regime look like a triumph and the hard one like a failure, when
the ratio to chance is what carries the meaning.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import load_embedding_table

BASE = re.compile(r"[(\[]")


def first(value) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None


def century_of(record: dict) -> int | None:
    year = str(record.get("inception") or "")[:4]
    return int(year) // 100 if year.isdigit() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+")
    parser.add_argument("--table", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--hub-size", type=int, default=400,
                        help="A notation on more items than this is a category, not a motif")
    parser.add_argument(
        "--hub-type",
        type=int,
        default=300,
        help="An object kind on more items than this is a category, not a kind. "
             "Without it 'painting' swallows the regime: the floor falls to 1.4 and "
             "every representation sits at rank 1, which measures nothing.",
    )
    parser.add_argument("--min-queries", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records: dict[str, dict] = {}
    for path in args.manifest:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                records[str(row["id"])] = row

    tables = {}
    for spec in args.table:
        name, _, path = spec.partition("=")
        tables[name] = load_embedding_table(path)

    # The pool is every item carrying all four attributes and present in every
    # table, so no regime and no representation is scored on a different set.
    shared = set.intersection(*(set(t.ids) for t in tables.values()))
    items: list[str] = []
    notation: dict[str, set[str]] = {}
    kind: dict[str, str] = {}
    collection: dict[str, str] = {}
    century: dict[str, int] = {}
    material: dict[str, str] = {}
    creator: dict[str, str] = {}
    corpus: dict[str, str] = {}
    for item in sorted(shared):
        row = records.get(item)
        if not row:
            continue
        notes = {BASE.split(str(x))[0].strip() for x in row.get("iconclass", [])}
        k, c, s = first(row.get("type")), first(row.get("collection")), century_of(row)
        # An emblem carries no Wikidata type and no date, but it is a print, and
        # that is a fact rather than a filler. It joins the pool and the regimes
        # whose attributes it has; the century regime is simply not open to it.
        origin = str(row.get("source_dataset") or "")
        if not k and origin:
            k = "estampe"
        if s is None:
            s = -1
        if notes and k and c:
            items.append(item)
            notation[item], kind[item], collection[item], century[item] = notes, k, c, s
            # Material and creator are not on every record, so their regimes run
            # on the subset that carries them rather than shrinking the pool for
            # everyone: a regime nobody can answer is not a harder regime.
            material[item] = first(row.get("material")) or ""
            creator[item] = first(row.get("creator")) or ""
            corpus[item] = origin or "musee"

    if len(items) < args.min_queries:
        raise SystemExit("too few items carry all four attributes")
    position = {item: index for index, item in enumerate(items)}

    kind_frequency = collections.Counter(kind[item] for item in items)
    rare_kinds = {k for k, count in kind_frequency.items() if count <= args.hub_type}
    material_frequency = collections.Counter(
        material[item] for item in items if material[item]
    )
    rare_materials = {
        m for m, count in material_frequency.items() if count <= args.hub_type
    }

    frequency = collections.Counter(n for item in items for n in notation[item])
    motifs = {n for n, count in frequency.items() if count <= args.hub_size}
    by_notation: dict[str, set[str]] = collections.defaultdict(set)
    for item in items:
        for n in notation[item] & motifs:
            by_notation[n].add(item)

    def shares_subject(a: str, b: str) -> bool:
        return bool((notation[a] & motifs) & (notation[b] & motifs))

    # Each regime: the attribute that must match, and the ones that must differ.
    regimes = {
        "sujet (autre objet, autre collection, autre siècle)": (
            lambda a, b: shares_subject(a, b)
            and kind[a] != kind[b]
            and collection[a] != collection[b]
            and century[a] > 0
            and century[b] > 0
            and century[a] != century[b]
        ),
        "type d'objet, rare (aucun sujet commun)": (
            lambda a, b: kind[a] == kind[b]
            and kind[a] in rare_kinds
            and not shares_subject(a, b)
            and collection[a] != collection[b]
        ),
        "siècle (aucun sujet commun, autre objet)": (
            lambda a, b: century[a] > 0
            and century[a] == century[b]
            and not shares_subject(a, b)
            and kind[a] != kind[b]
            and collection[a] != collection[b]
        ),
        "collection (aucun sujet commun, autre objet)": (
            lambda a, b: collection[a] == collection[b]
            and not shares_subject(a, b)
            and kind[a] != kind[b]
        ),
        "sujet, autre corpus et autre type (sans clause de siècle)": (
            lambda a, b: shares_subject(a, b)
            and corpus[a] != corpus[b]
            and kind[a] != kind[b]
        ),
        "matière (aucun sujet commun, autre collection)": (
            lambda a, b: bool(material[a])
            and material[a] == material[b]
            and material[a] in rare_materials
            and not shares_subject(a, b)
            and collection[a] != collection[b]
        ),
        "main (même créateur, aucun sujet commun)": (
            lambda a, b: bool(creator[a])
            and creator[a] == creator[b]
            and not shares_subject(a, b)
        ),
        # The same regime with the institution held apart. A maker's works are
        # usually gathered in one museum, so "same hand" and "same photographic
        # convention" are confounded until the collection is required to differ;
        # whichever of the two the pixels were reading, this separates them.
        "main, autre collection (aucun sujet commun)": (
            lambda a, b: bool(creator[a])
            and creator[a] == creator[b]
            and not shares_subject(a, b)
            and collection[a] != collection[b]
        ),
    }

    # Candidates are narrowed before the predicate runs, since a full pairwise
    # sweep over the pool is quadratic and most pairs fail on the first clause.
    def targets_for(name: str, predicate) -> dict[str, set[str]]:
        found: dict[str, set[str]] = {}
        if name.startswith("sujet"):
            for item in items:
                pool = set()
                for n in notation[item] & motifs:
                    pool |= by_notation[n]
                hits = {other for other in pool if other != item and predicate(item, other)}
                if hits:
                    found[item] = hits
            return found
        by_key: dict[object, list[str]] = collections.defaultdict(list)
        key = {
            "type": kind, "sièc": century, "coll": collection,
            "mati": material, "main": creator,
        }[name[:4]]
        for item in items:
            by_key[key[item]].append(item)
        for item in items:
            hits = {
                other for other in by_key[key[item]]
                if other != item and predicate(item, other)
            }
            if hits:
                found[item] = hits
        return found

    report = {"pool": len(items), "hub_size": args.hub_size, "regimes": {}}
    matrices = {
        name: table.vectors[[{i: n for n, i in enumerate(table.ids)}[item] for item in items]]
        for name, table in tables.items()
    }
    for name, matrix in matrices.items():
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrices[name] = matrix / norms

    for regime, predicate in regimes.items():
        targets = targets_for(regime, predicate)
        if len(targets) < args.min_queries:
            continue
        # Expected rank of the best of k targets under a random ordering.
        floors = [(len(items) + 1) / (len(hits) + 1) for hits in targets.values()]
        entry = {
            "queries": len(targets),
            "median_targets_per_query": statistics.median(
                len(hits) for hits in targets.values()
            ),
            "floor_median_rank_if_random": round(statistics.median(floors), 1),
            "representations": {},
        }
        for name, matrix in matrices.items():
            ranks = []
            for query, hits in targets.items():
                index = position[query]
                scores = matrix[index] @ matrix.T
                scores[index] = -np.inf
                order = np.argsort(-scores)
                place = {items[int(j)]: rank for rank, j in enumerate(order, start=1)}
                ranks.append(min(place[target] for target in hits))
            array = np.asarray(ranks)
            entry["representations"][name] = {
                "median_rank": int(np.median(array)),
                "ratio_to_floor": round(
                    float(statistics.median(floors) / max(np.median(array), 1)), 2
                ),
                "share_in_top_10": round(float((array <= 10).mean()), 4),
                "share_in_top_100": round(float((array <= 100).mean()), 4),
            }
        report["regimes"][regime] = entry

    report["reading"] = (
        "A ratio to floor above one means the representation finds the partner sooner than "
        "a random ordering would. Comparing ratios across regimes is the point: a "
        "representation keyed to the object's own form will do well in the object-kind "
        "regime and badly in the subject regime, and saying which is better is a category "
        "error -- they answer different questions."
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for regime, entry in report["regimes"].items():
        print(
            f"\n{regime}  ({entry['queries']} requêtes, "
            f"plancher {entry['floor_median_rank_if_random']})"
        )
        for name, value in sorted(
            entry["representations"].items(), key=lambda kv: -kv[1]["ratio_to_floor"]
        ):
            print(f"   {name:24s} rang {value['median_rank']:6d}  x{value['ratio_to_floor']:5.2f} "
                  f"top-100 {value['share_in_top_100']:.1%}")


if __name__ == "__main__":
    main()
