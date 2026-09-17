#!/usr/bin/env python3
"""Compare combination rules under the frozen protocol, on the same relevance path.

The weighted sum is one rule among several, and the hard-pair failure has a
shape -- a saturated graph score flooding the top of the list -- that other rules
are built not to have. This script runs each rule through the identical
calibration, alpha-selection and evaluation code as the frozen baseline, so the
only thing that differs between the rows of its output is the rule.

Alpha is selected on validation for every rule independently, with the same
conservative tie-break that prefers the larger visual weight.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from caypollard.benchmarks.iconclass_retrieval import evaluate_iconclass_retrieval
from caypollard.embeddings.store import l2_normalize, load_embedding_table
from caypollard.fusion import fit_similarity_bounds, minmax_scale
from caypollard.fusion_variants import RULES, combine
from caypollard.graphs.iconclass import (
    build_parent_index,
    child_edges,
    key_augmented_parents,
    parse_notations,
)
from caypollard.provenance import read_jsonl
from caypollard.statistics import bootstrap_mean_ci


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_alphas(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or any(alpha < 0 or alpha > 1 for alpha in values):
        raise argparse.ArgumentTypeError("alphas must be comma-separated values in [0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("visual", help="Visual embedding table")
    parser.add_argument("graph", help="Graph embedding table")
    parser.add_argument("manifest", help="Canonical benchmark manifest JSONL")
    parser.add_argument("notations", help="Pinned Iconclass notations.txt")
    parser.add_argument("--rules", type=_parse_list, default=list(RULES))
    parser.add_argument("--alphas", type=_parse_alphas, default=_parse_alphas("0,0.25,0.5,0.75,1"))
    parser.add_argument("--key-policy", choices=("strip", "keep"), default="keep")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--sample-pairs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    visual = load_embedding_table(args.visual)
    graph = load_embedding_table(args.graph)
    records = read_jsonl(args.manifest)
    parents = build_parent_index(child_edges(parse_notations(args.notations)))
    if args.key_policy == "keep":
        parents = key_augmented_parents(
            parents, (label for row in records for label in row.get("iconclass", []))
        )

    common = set(visual.ids).intersection(graph.ids)
    validation_ids = sorted(
        str(record["id"])
        for record in records
        if str(record.get("split")) == args.validation_split and str(record["id"]) in common
    )
    visual_bounds, visual_calibration = fit_similarity_bounds(
        visual, validation_ids, sample_pairs=args.sample_pairs, seed=args.seed
    )
    graph_bounds, graph_calibration = fit_similarity_bounds(
        graph, validation_ids, sample_pairs=args.sample_pairs, seed=args.seed
    )

    visual_unit = l2_normalize(visual.vectors)
    graph_unit = l2_normalize(graph.vectors)
    visual_row = {item_id: index for index, item_id in enumerate(visual.ids)}
    graph_row = {item_id: index for index, item_id in enumerate(graph.ids)}

    def make_scorer(rule: str, alpha: float):
        def score(query_ids: Sequence[str], candidate_ids: Sequence[str]) -> np.ndarray:
            q_visual = visual_unit[[visual_row[item] for item in query_ids]]
            c_visual = visual_unit[[visual_row[item] for item in candidate_ids]]
            q_graph = graph_unit[[graph_row[item] for item in query_ids]]
            c_graph = graph_unit[[graph_row[item] for item in candidate_ids]]
            v = minmax_scale(q_visual @ c_visual.T, low=visual_bounds.low, high=visual_bounds.high)
            g = minmax_scale(q_graph @ c_graph.T, low=graph_bounds.low, high=graph_bounds.high)
            return combine(v, g, rule=rule, alpha=alpha)

        return score

    # The fused table has to be restricted to items both modalities carry, or the
    # scorer is asked for a row that does not exist.
    aligned = [row for row in records if str(row["id"]) in common]
    restricted = visual.__class__(
        ids=tuple(item for item in visual.ids if item in common),
        vectors=np.stack([visual_unit[visual_row[item]] for item in visual.ids if item in common]),
        metadata={"family": "alignment-carrier"},
    )

    results = []
    for rule in args.rules:
        validation_runs = []
        for alpha in args.alphas:
            summary, _ = evaluate_iconclass_retrieval(
                restricted,
                aligned,
                parents,
                query_split=args.validation_split,
                candidate_split=args.validation_split,
                score_matrix_fn=make_scorer(rule, alpha),
            )
            validation_runs.append({"alpha": alpha, "ndcg": summary.get("mean_ndcg_at_10")})
        evaluable = [run for run in validation_runs if run["ndcg"] is not None]
        if not evaluable:
            raise ValueError(f"rule {rule}: no validation query has hierarchical relevance")
        selected = max(evaluable, key=lambda run: (float(run["ndcg"]), float(run["alpha"])))
        alpha = float(selected["alpha"])

        test_summary, test_queries = evaluate_iconclass_retrieval(
            restricted,
            aligned,
            parents,
            query_split=args.test_split,
            candidate_split=args.test_split,
            score_matrix_fn=make_scorer(rule, alpha),
        )
        ndcgs = [row.ndcg_at_10 for row in test_queries if row.ndcg_at_10 is not None]
        results.append(
            {
                "rule": rule,
                "selected_alpha": alpha,
                "validation_sweep": validation_runs,
                "test": {
                    "mean_ndcg_at_10": test_summary.get("mean_ndcg_at_10"),
                    "map": test_summary.get("map"),
                    "mrr": test_summary.get("mrr"),
                    "mean_recall_at_10": test_summary.get("mean_recall_at_10"),
                    "n_queries": test_summary.get("n_queries"),
                    "ndcg_ci95": bootstrap_mean_ci(ndcgs, seed=args.seed),
                },
            }
        )
        best = results[-1]["test"]["mean_ndcg_at_10"]
        print(f"{rule:12s} alpha={alpha:<5} nDCG@10={best:.4f}", flush=True)

    report = {
        "label": args.label,
        "protocol": "protocol-v0.5",
        "visual": args.visual,
        "graph": args.graph,
        "key_policy": args.key_policy,
        "visual_calibration": visual_calibration,
        "graph_calibration": graph_calibration,
        "rules": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
