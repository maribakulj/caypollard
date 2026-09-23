#!/usr/bin/env python3
"""Assemble a record from named nodes, whoever named them.

Two namers write the same thing -- a picture, a list of nodes, each a word from
a closed vocabulary and a cell of a coarse grid -- and they have to end in the
same space or they cannot be compared. One scores every segmented part against
the vocabulary and knows how large each part is; the other reads the picture
whole and does not. So the weight is a parameter rather than an assumption:
``area`` uses the part's share of the frame where that is recorded, ``count``
gives every node the same weight, and the choice is written into the metadata
beside the figures it produced.

The layout is nodes by place, which is the form the pooling comparison
selected: one block of vocabulary counts per cell, in cell order. Two records
built from the same vocabulary and grid are therefore comparable dimension by
dimension, whatever produced them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import save_embedding_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", help="jsonl of {id, nodes: [{name, cell, area?}]}")
    parser.add_argument("--labels", default="data/vocabularies/everyday-nouns.txt")
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--weight", choices=("area", "count"), default="count")
    parser.add_argument(
        "--max-nodes",
        type=int,
        help="Keep only the largest N nodes of each picture. The namers differ in "
             "how many they return, and whether that is what separates their records "
             "is a measurement rather than a guess.",
    )
    parser.add_argument("--namer", default="", help="Recorded in the metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shuffled-output",
                        help="The same places and counts with names drawn at random "
                             "from the same vocabulary -- what the arrangement alone is worth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    words = [
        line.strip()
        for line in Path(args.labels).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    vocabulary = list(dict.fromkeys(words))
    index = {word: position for position, word in enumerate(vocabulary)}
    cells = args.grid * args.grid
    width = cells * len(vocabulary)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    shuffled: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)
    empty = 0
    for line in Path(args.nodes).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        vector = np.zeros(width, dtype=np.float32)
        noise = np.zeros(width, dtype=np.float32)
        nodes = row.get("nodes") or []
        if args.max_nodes:
            nodes = sorted(nodes, key=lambda n: -float(n.get("area", 0.0)))[: args.max_nodes]
        for node in nodes:
            name = node.get("name")
            # A node's cell is expressed in the grid its namer used. Asking for a
            # single cell means "forget where it was", so the node moves to the
            # only cell there is -- dropping it instead would keep just the
            # nodes that happened to sit in the first cell, which is a biased
            # subset and not a record without position.
            cell = 0 if cells == 1 else int(node.get("cell", 0))
            if name not in index or not 0 <= cell < cells:
                continue
            weight = (
                float(node.get("area", 0.0)) + 1e-3 if args.weight == "area" else 1.0
            )
            vector[cell * len(vocabulary) + index[name]] += weight
            noise[cell * len(vocabulary) + int(rng.integers(len(vocabulary)))] += weight
        if not vector.any():
            # A picture the namer returned nothing for has no record to give,
            # and normalising a zero vector would invent one.
            empty += 1
            continue
        ids.append(str(row["id"]))
        vectors.append(vector)
        shuffled.append(noise)

    metadata = {
        "family": "symbolic",
        "method": "named nodes by grid cell",
        "namer": args.namer,
        "vocabulary": len(vocabulary),
        "grid": args.grid,
        "weight": args.weight,
        "max_nodes": args.max_nodes,
        "pictures_without_nodes": empty,
    }
    save_embedding_table(
        args.output, ids=tuple(ids), vectors=np.stack(vectors),
        metadata=metadata, normalize=True,
    )
    if args.shuffled_output:
        save_embedding_table(
            args.shuffled_output, ids=tuple(ids), vectors=np.stack(shuffled),
            metadata={**metadata, "family": "control",
                      "method": "the same places and counts, names drawn at random",
                      "seed": args.seed},
            normalize=True,
        )
    print(
        json.dumps(
            {"pictures": len(ids), "without_nodes": empty, "dimensions": width,
             "output": args.output},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
