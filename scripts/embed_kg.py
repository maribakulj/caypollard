#!/usr/bin/env python3
"""Embed a relation graph with transparent or PyKEEN KGE baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from caypollard.embeddings.store import save_embedding_table
from caypollard.graphs.embeddings import (
    node2vec_ppmi_embeddings,
    pykeen_kge_embeddings,
    rdf2vec_ppmi_embeddings,
)
from caypollard.graphs.triples import (
    ProjectionAudit,
    mask_target_relations,
    read_triples_tsv,
)


def _read_ids(path: str | Path) -> set[str]:
    return {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("triples", help="Three-column TSV: head, relation, tail")
    parser.add_argument("output", help="Output NPZ embedding table")
    parser.add_argument(
        "--method",
        choices=("node2vec-ppmi", "rdf2vec-ppmi", "complex", "rotate"),
        default="rdf2vec-ppmi",
    )
    parser.add_argument("--projection-id", default="G1")
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walks", type=int, default=10)
    parser.add_argument("--walk-length", type=int, default=8)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument(
        "--mask-entities",
        help="Optional newline-delimited held-out entity IDs whose target edges are masked",
    )
    parser.add_argument(
        "--mask-relations",
        nargs="+",
        default=["has_iconclass"],
        help="Relations removed for --mask-entities (default: has_iconclass)",
    )
    args = parser.parse_args()

    triples = read_triples_tsv(args.triples)
    masked = ()
    if args.mask_entities:
        triples, masked = mask_target_relations(
            triples,
            target_entities=_read_ids(args.mask_entities),
            relations=args.mask_relations,
        )

    if args.method == "node2vec-ppmi":
        table = node2vec_ppmi_embeddings(
            triples,
            dimension=args.dimension,
            walks_per_node=args.walks,
            walk_length=args.walk_length,
            window=args.window,
            p=args.p,
            q=args.q,
            seed=args.seed,
        )
    elif args.method == "rdf2vec-ppmi":
        table = rdf2vec_ppmi_embeddings(
            triples,
            dimension=args.dimension,
            walks_per_entity=args.walks,
            walk_length=args.walk_length,
            window=args.window,
            seed=args.seed,
        )
    else:
        table = pykeen_kge_embeddings(
            triples,
            model=args.method,
            dimension=args.dimension,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )

    audit = ProjectionAudit.from_triples(
        triples,
        projection_id=args.projection_id,
        masked_triples=len(masked),
    )
    metadata = dict(table.metadata)
    metadata["projection"] = asdict(audit)
    metadata["source_triples"] = str(Path(args.triples))
    metadata["masked_relations"] = list(args.mask_relations) if args.mask_entities else []
    metadata["mask_entities_file"] = args.mask_entities
    save_embedding_table(
        args.output,
        ids=table.ids,
        vectors=table.vectors,
        metadata=metadata,
        normalize=False,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
