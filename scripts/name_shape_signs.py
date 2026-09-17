#!/usr/bin/env python3
"""Give the shape signs names a historian can read, and contest.

The vocabulary is found by clustering, so its signs are numbers. A number cannot
be argued with: an art historian shown "sign 312" has no way to say it is wrong.
The point of a closed vocabulary is that it is inspectable, and a sign is only
inspectable once it carries both a description of its form and an account of what
it tends to accompany.

Two halves, deliberately kept apart.

The **form** is read off the regions assigned to the sign -- how large, how
elongated, how solid, darker or lighter than the page, and where in the frame it
tends to sit. This is what the sign *is*, and it is true by construction.

The **association** is read off the Iconclass notations of the pictures carrying
it, ranked by lift over the corpus base rate rather than by raw frequency: a
notation on half the corpus will appear under every sign and mean nothing. This
is what the sign tends to *accompany*, which is a statistical fact about this
corpus and not a definition. A sign is never renamed into its notation; the
notation is offered as evidence for a human to accept or reject.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from caypollard.graphs.iconclass import (
    build_parent_index,
    child_edges,
    parse_notations,
    resolve_notation,
)
from caypollard.provenance import read_jsonl
from caypollard.statistics import compare_methods

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_shape_vocabulary import descriptor

BASE = re.compile(r"[(\[]")


def shape_words(rows: list[dict]) -> str:
    """A plain description of what the sign's regions look like."""
    area = float(np.median([r["area"] for r in rows]))
    elongation = float(np.median([r["elongation"] for r in rows]))
    solidity = float(np.median([r["solidity"] for r in rows]))
    y = float(np.median([r["centroid_y"] for r in rows]))
    dark = sum(1 for r in rows if r["tone"] == "sombre") / len(rows)

    size = "minuscule" if area < 0.01 else "petite" if area < 0.03 else (
        "moyenne" if area < 0.10 else "grande"
    )
    form = "très allongée" if elongation > 2.2 else "allongée" if elongation > 1.4 else "compacte"
    fill = "pleine" if solidity > 0.6 else "découpée" if solidity > 0.35 else "très découpée"
    place = "en haut" if y < 0.38 else "en bas" if y > 0.62 else "au centre"
    tone = "sombre" if dark > 0.6 else "claire" if dark < 0.4 else "de ton variable"
    return f"tache {size}, {form}, {fill}, {tone}, plutôt {place}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vocabulary_dir")
    parser.add_argument("--regions", action="append", required=True, metavar="CORPUS=PATH")
    parser.add_argument("--manifest", action="append", required=True, metavar="PATH")
    parser.add_argument("--min-support", type=int, default=40)
    parser.add_argument("--top-notations", type=int, default=5)
    parser.add_argument("--min-lift", type=float, default=1.5)
    parser.add_argument("--notations", help="Pinned notations.txt, for the coherence test")
    parser.add_argument("--labels", help="iconclass-wikidata.jsonl, for readable labels")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    signs = int(
        json.loads(
            (Path(args.vocabulary_dir) / "vocabulary-report.json").read_text(encoding="utf-8")
        )["signs"]
    )

    labels: dict[str, set[str]] = {}
    for path in args.manifest:
        for row in read_jsonl(path):
            labels[str(row["id"])] = {
                BASE.split(str(x))[0].strip() for x in row.get("iconclass", [])
            }

    rows = []
    for spec in args.regions:
        _name, _, path = spec.partition("=")
        rows.extend(
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    pool = np.stack([descriptor(region) for row in rows for region in row["regions"]])
    centre, scale = pool.mean(axis=0), pool.std(axis=0)
    scale[scale == 0] = 1.0
    model = MiniBatchKMeans(n_clusters=signs, random_state=42, n_init=8, batch_size=4096).fit(
        (pool - centre) / scale
    )

    by_sign_regions: dict[int, list[dict]] = collections.defaultdict(list)
    by_sign_items: dict[int, set[str]] = collections.defaultdict(set)
    for row in rows:
        if not row["regions"]:
            continue
        described = (np.stack([descriptor(r) for r in row["regions"]]) - centre) / scale
        assigned = model.predict(described)
        for sign, region in zip(assigned, row["regions"], strict=True):
            by_sign_regions[int(sign)].append(region)
            by_sign_items[int(sign)].add(str(row["id"]))

    annotated = [item for item in labels if labels[item]]
    base_rate = collections.Counter(
        notation for item in annotated for notation in labels[item]
    )
    total_items = len(annotated)

    entries = []
    for sign in sorted(by_sign_regions):
        items = [item for item in by_sign_items[sign] if labels.get(item)]
        if len(items) < args.min_support:
            continue
        carried = collections.Counter(
            notation for item in items for notation in labels[item]
        )
        associations = []
        for notation, count in carried.items():
            if count < 5:
                continue
            share = count / len(items)
            base = base_rate[notation] / total_items
            lift = share / base if base > 0 else 0.0
            if lift >= args.min_lift:
                associations.append(
                    {
                        "notation": notation,
                        "share_of_pictures_with_this_sign": round(share, 4),
                        "share_in_corpus": round(base, 4),
                        "lift": round(lift, 2),
                        "n": count,
                    }
                )
        associations.sort(key=lambda entry: -entry["lift"])
        entries.append(
            {
                "sign": sign,
                "form": shape_words(by_sign_regions[sign]),
                "n_regions": len(by_sign_regions[sign]),
                "n_pictures": len(items),
                "associated_notations": associations[: args.top_notations],
            }
        )

    if args.labels:
        # Labels come from a Wikidata entity carrying the notation, which is not the
        # same thing as a definition of it: notation 83 resolves to a particular
        # medieval poem rather than to "tales and romances". They are offered as an
        # aid to reading and marked as such.
        readable: dict[str, str] = {}
        for line in Path(args.labels).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            base = BASE.split(row["notation"])[0].strip()
            for entity in row["wikidata"]:
                if entity.get("label"):
                    readable.setdefault(base, entity["label"])
                    break
        for entry in entries:
            for association in entry["associated_notations"]:
                association["wikidata_label"] = readable.get(association["notation"])

    coherence = None
    if args.notations:
        # Are a sign's notations related to one another, or merely each enriched?
        # Jaccard over resolved ancestors, against sets drawn at random from the
        # same pool, which is the only fair control: the pool is already unusual.
        parents = build_parent_index(child_edges(parse_notations(args.notations)))

        def ancestors(notation: str) -> set[str]:
            node = resolve_notation(str(notation), parents)
            seen: set[str] = set()
            stack = [node] if node else []
            while stack:
                current = stack.pop()
                if current is None or current in seen:
                    continue
                seen.add(current)
                stack.extend(parents.get(current, ()))
            return seen

        def mean_similarity(notations: list[str]) -> float | None:
            values = []
            for left, right in itertools.combinations(notations, 2):
                a, b = ancestors(left), ancestors(right)
                if a and b:
                    values.append(len(a & b) / len(a | b))
            return float(np.mean(values)) if values else None

        observed = [
            value
            for entry in entries
            if len(entry["associated_notations"]) >= 3
            and (
                value := mean_similarity(
                    [x["notation"] for x in entry["associated_notations"][:5]]
                )
            )
            is not None
        ]
        pool_notations = sorted(
            {x["notation"] for entry in entries for x in entry["associated_notations"]}
        )
        generator = random.Random(42)
        control = []
        while len(control) < len(observed) and len(pool_notations) >= 5:
            value = mean_similarity(generator.sample(pool_notations, 5))
            if value is not None:
                control.append(value)
        if observed and control:
            size = min(len(observed), len(control))
            comparison = compare_methods(
                observed[:size], control[:size], first_name="signs", second_name="random", seed=42
            )
            coherence = {
                "mean_within_sign": round(float(np.mean(observed)), 4),
                "mean_random": round(float(np.mean(control)), 4),
                "difference": round(comparison["mean_difference"], 4),
                "p_value": comparison["p_value"],
                "cohens_d": round(comparison["cohens_d"], 3),
                "n_signs": size,
                "reading": (
                    "Above random means a sign's notations sit in somewhat related branches. "
                    "An absolute value near 0.06 means they share a sixteenth of their "
                    "ancestors, so a sign is a tendency and not a name."
                ),
            }

    named = sum(1 for entry in entries if entry["associated_notations"])
    report = {
        "vocabulary": args.vocabulary_dir,
        "signs": signs,
        "signs_with_enough_support": len(entries),
        "signs_with_an_association": named,
        "min_support": args.min_support,
        "min_lift": args.min_lift,
        "coherence_of_associations": coherence,
        "reading": (
            "A sign's form is true by construction; its associations are a fact about this "
            "corpus and an invitation to a historian, not a definition. Lift is the ratio of "
            "how often a notation accompanies the sign to how often it occurs at all, so a "
            "notation carried by half the corpus cannot rank highly by being common."
        ),
        "signs_table": entries,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("signs_table", "reading")},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
