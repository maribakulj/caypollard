#!/usr/bin/env python3
"""Create a transparent taxonomy-only graph embedding control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caypollard.embeddings.store import save_embedding_table
from caypollard.graphs.embeddings import (
    adjacency_svd_embeddings,
    aggregate_concept_embeddings_to_images,
)
from caypollard.graphs.iconclass import build_parent_index, child_edges, parse_notations
from caypollard.provenance import read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notations", help="Pinned Iconclass notations.txt")
    parser.add_argument("output", help="Output NPZ for concept embeddings")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        help="Optional benchmark manifest; also emits mean-pooled image graph embeddings",
    )
    parser.add_argument("--image-output")
    args = parser.parse_args()

    records = parse_notations(args.notations)
    edges = child_edges(records)
    parents = build_parent_index(edges)
    table = adjacency_svd_embeddings(edges, dimension=args.dimension, seed=args.seed)
    metadata = dict(table.metadata)
    metadata["notations_sha256"] = sha256_file(args.notations)
    save_embedding_table(
        args.output,
        ids=table.ids,
        vectors=table.vectors,
        metadata=metadata,
        normalize=False,
    )

    if args.manifest:
        manifest = read_jsonl(args.manifest)
        image_table = aggregate_concept_embeddings_to_images(manifest, table, parents)
        image_output = (
            Path(args.image_output)
            if args.image_output
            else Path(args.output).with_name(Path(args.output).stem + "-images.npz")
        )
        image_metadata = dict(image_table.metadata)
        image_metadata["concept_embedding_file"] = str(Path(args.output))
        save_embedding_table(
            image_output,
            ids=image_table.ids,
            vectors=image_table.vectors,
            metadata=image_metadata,
            normalize=False,
        )
        print(json.dumps(image_metadata, indent=2, sort_keys=True))
    else:
        print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
