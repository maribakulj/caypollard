#!/usr/bin/env python3
"""Do two objects of different media that share a motif come out as neighbours?

This is the question a cross-collection discovery system has to answer, and it
is not the question the project has been measuring. Retrieval within one corpus
compares objects made the same way, photographed the same way, held in one
place; the neighbourhoods it produces are dominated by the support. A printed
emblem and a painted panel that share a subject are the case that matters, and
the case every representation tested so far has had no occasion to fail at.

Two corpora are pooled: emblem prints and museum objects, overwhelmingly
paintings but with drawings, watercolours, etchings, pastels and reliefs behind
them. They are annotated by different people in different centuries and joined
only by Iconclass, so a pair sharing a notation across the pool is a genuine
cross-medium iconographic relation.

Each representation is asked two things. Can a probe still tell which corpus an
object came from -- that is the medium confound at its starkest. And where does
an object's best cross-corpus partner rank -- that is whether the relation is
reachable at all.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from caypollard.embeddings.store import l2_normalize, load_embedding_table
from caypollard.provenance import read_jsonl

BASE = re.compile(r"[(\[]")


def strip(label: str) -> str:
    return BASE.split(str(label))[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="NAME=MANIFEST",
        help="A named corpus manifest, repeatable; exactly two are expected",
    )
    parser.add_argument(
        "--visual",
        action="append",
        required=True,
        metavar="CORPUS=PATH",
        help="A visual embedding table per corpus, from the same encoder",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--hub-size",
        type=int,
        default=400,
        help="A notation carried by more items than this is a category, not a motif",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    corpora = {}
    for spec in args.corpus:
        name, _, path = spec.partition("=")
        corpora[name] = {str(row["id"]): row for row in read_jsonl(path)}
    visual_paths = dict(spec.split("=", 1) for spec in args.visual)

    ids: list[str] = []
    corpus_of: dict[str, str] = {}
    signs: dict[str, set[str]] = {}
    vectors: list[np.ndarray] = []
    for name, records in corpora.items():
        table = load_embedding_table(visual_paths[name])
        row_of = {item: index for index, item in enumerate(table.ids)}
        unit = l2_normalize(table.vectors)
        for item in sorted(records):
            if item not in row_of:
                continue
            labels = {strip(x) for x in records[item].get("iconclass", [])}
            if not labels:
                continue
            ids.append(item)
            corpus_of[item] = name
            signs[item] = labels
            vectors.append(unit[row_of[item]])

    position = {item: index for index, item in enumerate(ids)}
    pixels = l2_normalize(np.stack(vectors))

    vocabulary: dict[str, int] = {}
    for item in ids:
        for sign in signs[item]:
            vocabulary.setdefault(sign, len(vocabulary))
    symbolic = np.zeros((len(ids), len(vocabulary)), dtype=np.float32)
    for index, item in enumerate(ids):
        for sign in signs[item]:
            symbolic[index, vocabulary[sign]] = 1.0
    symbolic = l2_normalize(symbolic)

    frequency = collections.Counter(sign for item in ids for sign in signs[item])
    motifs = {sign for sign, count in frequency.items() if count <= args.hub_size}

    partners: dict[str, set[str]] = {}
    inverted: dict[str, set[str]] = collections.defaultdict(set)
    for item in ids:
        for sign in signs[item] & motifs:
            inverted[sign].add(item)
    for item in ids:
        found: set[str] = set()
        for sign in signs[item] & motifs:
            found |= {
                other for other in inverted[sign] if corpus_of[other] != corpus_of[item]
            }
        if found:
            partners[item] = found

    labels = np.array([corpus_of[item] for item in ids])
    majority = collections.Counter(labels).most_common(1)[0][1] / len(labels)

    results = {}
    for name, matrix in (("pixels (DINOv2)", pixels), ("isotype (signes)", symbolic)):
        probe = LogisticRegression(max_iter=1500)
        linear = float(cross_val_score(probe, matrix, labels, cv=4, n_jobs=2).mean())
        knn = float(
            cross_val_score(
                KNeighborsClassifier(n_neighbors=args.k, metric="cosine"),
                matrix,
                labels,
                cv=4,
                n_jobs=2,
            ).mean()
        )
        ranks = []
        cross_in_top = 0
        for query, targets in partners.items():
            index = position[query]
            scores = matrix[index] @ matrix.T
            scores[index] = -np.inf
            order = np.argsort(-scores)
            rank_of = {ids[int(j)]: rank for rank, j in enumerate(order, start=1)}
            ranks.append(min(rank_of[target] for target in targets))
            cross_in_top += sum(
                1 for j in order[: args.k] if corpus_of[ids[int(j)]] != corpus_of[query]
            )
        ranks_array = np.asarray(ranks)
        results[name] = {
            "linear_probe_corpus": round(linear, 4),
            "knn_probe_corpus": round(knn, 4),
            "median_rank_of_best_cross_medium_partner": int(np.median(ranks_array)),
            "share_with_a_partner_in_top_10": round(float((ranks_array <= 10).mean()), 4),
            "share_with_a_partner_in_top_50": round(float((ranks_array <= 50).mean()), 4),
            "share_of_top_k_from_the_other_corpus": round(
                cross_in_top / max(len(partners) * args.k, 1), 4
            ),
        }
        entry = results[name]
        print(
            f"  {name:20s} sonde kNN {entry['knn_probe_corpus']:.1%}  "
            f"rang médian du partenaire {entry['median_rank_of_best_cross_medium_partner']:5d}  "
            f"top-10 {entry['share_with_a_partner_in_top_10']:.1%}  "
            f"voisins d'autre corpus {entry['share_of_top_k_from_the_other_corpus']:.1%}",
            flush=True,
        )

    report = {
        "corpora": {name: len(records) for name, records in corpora.items()},
        "pooled_items": len(ids),
        "items_with_a_cross_medium_partner": len(partners),
        "shared_motifs": len(motifs & set(frequency)),
        "majority_baseline": round(majority, 4),
        "k": args.k,
        "hub_size": args.hub_size,
        "results": results,
        "reading": (
            "The probe says how strongly the representation encodes the support; the median "
            "rank says whether a cross-medium relation is reachable at all. A representation "
            "that names the corpus perfectly and buries the partner has learned the material, "
            "not the motif."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
