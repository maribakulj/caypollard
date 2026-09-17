#!/usr/bin/env python3
"""Evaluate image embeddings against Iconclass hierarchy relevance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from caypollard.benchmarks.iconclass_retrieval import evaluate_iconclass_retrieval
from caypollard.embeddings.store import load_embedding_table
from caypollard.graphs.iconclass import (
    build_parent_index,
    child_edges,
    key_augmented_parents,
    parse_notations,
)
from caypollard.provenance import manifest_digest, read_jsonl, sha256_file
from caypollard.statistics import bootstrap_mean_ci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", help="NPZ embedding table produced by embed_images.py")
    parser.add_argument("manifest", help="Canonical benchmark manifest JSONL")
    parser.add_argument("notations", help="Pinned Iconclass notations.txt")
    parser.add_argument("--output-dir", default="results/visual-baseline")
    parser.add_argument("--query-split", default="test")
    parser.add_argument("--candidate-split", default="test")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--persist-top-k", type=int, default=10)
    parser.add_argument("--backend", choices=("numpy", "faiss"), default="numpy")
    parser.add_argument(
        "--key-policy",
        choices=("strip", "keep"),
        default="strip",
        help="Relevance treatment of bracketed Iconclass text keys: 'strip' folds "
             "86(MOTTO) onto 86 (protocol v0.1-v0.4); 'keep' attaches each observed "
             "key as a child of its base notation (protocol v0.5).",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    table = load_embedding_table(args.embeddings)
    records = read_jsonl(args.manifest)
    parents = build_parent_index(child_edges(parse_notations(args.notations)))
    if args.key_policy == "keep":
        parents = key_augmented_parents(
            parents, (label for row in records for label in row.get("iconclass", []))
        )
    summary, per_query = evaluate_iconclass_retrieval(
        table,
        records,
        parents,
        query_split=None if args.query_split == "all" else args.query_split,
        candidate_split=None if args.candidate_split == "all" else args.candidate_split,
        query_limit=args.query_limit,
        batch_size=args.batch_size,
        persist_top_k=args.persist_top_k,
        backend=args.backend,
    )

    # A headline metric without an interval invites reading a sampling wobble as
    # a method difference, so every reported mean carries one.
    intervals: dict[str, dict] = {}
    for name, values in (
        ("mean_ndcg_at_10", [row.ndcg_at_10 for row in per_query]),
        ("mrr", [row.reciprocal_rank for row in per_query]),
        ("map", [row.average_precision for row in per_query]),
        ("mean_recall_at_1", [row.recall_at_1 for row in per_query]),
        ("mean_recall_at_5", [row.recall_at_5 for row in per_query]),
        ("mean_recall_at_10", [row.recall_at_10 for row in per_query]),
    ):
        observed = [value for value in values if value is not None]
        if observed:
            intervals[name] = bootstrap_mean_ci(
                observed,
                confidence=args.bootstrap_confidence,
                n_resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
            )

    summary.update(
        {
            "bootstrap_intervals": intervals,
            "embedding_file": str(Path(args.embeddings)),
            "embedding_metadata": table.metadata,
            "manifest_file": str(Path(args.manifest)),
            "manifest_sha256": manifest_digest(records),
            "notations_file": str(Path(args.notations)),
            "notations_sha256": sha256_file(args.notations),
        }
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    with (output / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "query_id",
            "n_candidates",
            "n_exact_relevant",
            "ndcg_at_10",
            "reciprocal_rank",
            "average_precision",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "median_query_label_frequency",
            "median_query_hierarchy_depth",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_query:
            payload = row.to_dict()
            writer.writerow({key: payload[key] for key in fieldnames})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
