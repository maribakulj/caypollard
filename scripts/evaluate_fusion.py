#!/usr/bin/env python3
"""Tune transparent visual+graph fusion on validation data and evaluate once on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caypollard.benchmarks.iconclass_retrieval import evaluate_iconclass_retrieval
from caypollard.embeddings.store import load_embedding_table, save_embedding_table
from caypollard.fusion import fit_similarity_bounds, late_fusion_embedding_table
from caypollard.graphs.iconclass import build_parent_index, child_edges, parse_notations
from caypollard.provenance import read_jsonl


def _parse_alphas(value: str) -> list[float]:
    values = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not values or any(alpha < 0 or alpha > 1 for alpha in values):
        raise argparse.ArgumentTypeError("alphas must be comma-separated values in [0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("visual_embeddings")
    parser.add_argument("graph_embeddings")
    parser.add_argument("manifest")
    parser.add_argument("notations")
    parser.add_argument("output_dir")
    parser.add_argument("--alphas", type=_parse_alphas, default=_parse_alphas("0,0.25,0.5,0.75,1"))
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--sample-pairs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("numpy", "faiss"), default="numpy")
    args = parser.parse_args()

    visual = load_embedding_table(args.visual_embeddings)
    graph = load_embedding_table(args.graph_embeddings)
    records = read_jsonl(args.manifest)
    parents = build_parent_index(child_edges(parse_notations(args.notations)))
    common = set(visual.ids).intersection(graph.ids)
    validation_ids = sorted(
        str(record["id"])
        for record in records
        if str(record.get("split")) == args.validation_split and str(record["id"]) in common
    )
    if len(validation_ids) < 3:
        raise ValueError("at least three aligned validation items are required")

    visual_bounds, visual_calibration = fit_similarity_bounds(
        visual, validation_ids, sample_pairs=args.sample_pairs, seed=args.seed
    )
    graph_bounds, graph_calibration = fit_similarity_bounds(
        graph, validation_ids, sample_pairs=args.sample_pairs, seed=args.seed
    )

    validation_runs = []
    fused_tables = {}
    for alpha in args.alphas:
        fused = late_fusion_embedding_table(
            visual,
            graph,
            alpha=alpha,
            visual_bounds=visual_bounds,
            graph_bounds=graph_bounds,
        )
        summary, _ = evaluate_iconclass_retrieval(
            fused,
            records,
            parents,
            query_split=args.validation_split,
            candidate_split=args.validation_split,
            backend=args.backend,
        )
        validation_runs.append({"alpha": alpha, "summary": summary})
        fused_tables[alpha] = fused

    evaluable = [
        run for run in validation_runs if run["summary"].get("mean_ndcg_at_10") is not None
    ]
    if not evaluable:
        raise ValueError("no validation query has hierarchical relevance for alpha selection")
    # Conservative tie-break: prefer the larger visual weight when validation
    # nDCG is exactly equal, so graph complexity has to earn its influence.
    selected = max(
        evaluable,
        key=lambda run: (float(run["summary"]["mean_ndcg_at_10"]), float(run["alpha"])),
    )
    selected_alpha = float(selected["alpha"])
    fused = fused_tables[selected_alpha]
    test_summary, test_queries = evaluate_iconclass_retrieval(
        fused,
        records,
        parents,
        query_split=args.test_split,
        candidate_split=args.test_split,
        backend=args.backend,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_embedding_table(
        output_dir / "selected-fusion.npz",
        ids=fused.ids,
        vectors=fused.vectors,
        metadata={
            **fused.metadata,
            "protocol": "protocol-v0.3",
            "selected_on": args.validation_split,
        },
        normalize=False,
    )
    result = {
        "protocol": "protocol-v0.3",
        "selected_alpha": selected_alpha,
        "visual_calibration": visual_calibration,
        "graph_calibration": graph_calibration,
        "validation_runs": validation_runs,
        "test_summary": test_summary,
        "test_queries": [row.to_dict() for row in test_queries],
    }
    (output_dir / "fusion-evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "test_queries"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
