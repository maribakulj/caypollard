#!/usr/bin/env python3
"""Does fusion help the hard pairs, and does the combination rule matter?

Every hard-pair measurement in this project so far scored the graph alone. The
preregistered hypothesis is about the retrieval system, so the question is
whether a *fused* score separates a hard positive from a candidate the visual
encoder finds equally distant, and whether a rule that refuses to let one
modality dominate beats the weighted sum.

Controls come from `benchmarks.matched_controls`, so visual similarity carries
no information by construction. Alphas are read from a validation-selected
sweep and never fitted here. Two AUCs are reported for every scorer: over all
matched pairs, and over the subset where neither the positive nor the control
sits in the query's own volume -- volume co-membership being the one thing the
context graph is already known to encode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from caypollard.benchmarks.matched_controls import (
    bootstrap_paired_auc,
    match_controls,
)
from caypollard.embeddings.store import l2_normalize, load_embedding_table
from caypollard.fusion import fit_similarity_bounds, minmax_scale
from caypollard.fusion_variants import RULES, combine
from caypollard.provenance import read_jsonl

MIN_SUBSET_PAIRS = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", help="Hard-pair JSONL, mined with a stratum")
    parser.add_argument("manifest", help="Manifest carrying the stratum key")
    parser.add_argument("visual")
    parser.add_argument("graph")
    parser.add_argument("--alpha-from", help="Fusion-variants report supplying each rule's alpha")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fallback when no report is given")
    parser.add_argument("--stratum-key", default="collection")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Repeat the control draw this many times from consecutive seeds. One draw "
             "can be lucky; the spread across draws is what makes the AUC defensible.",
    )
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--label", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pairs = [
        json.loads(line)
        for line in Path(args.pairs).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = [pair for pair in pairs if pair["pair_class"] == "hard_positive"]
    records = {str(row["id"]): row for row in read_jsonl(args.manifest)}
    visual = load_embedding_table(args.visual)
    graph = load_embedding_table(args.graph)
    common = set(visual.ids) & set(graph.ids) & set(records)

    visual_row = {item: i for i, item in enumerate(visual.ids)}
    graph_row = {item: i for i, item in enumerate(graph.ids)}
    v_unit = l2_normalize(visual.vectors)
    g_unit = l2_normalize(graph.vectors)

    validation_ids = sorted(
        item for item in common if str(records[item].get("split")) == args.validation_split
    )
    visual_bounds, _ = fit_similarity_bounds(visual, validation_ids, seed=args.seed)
    graph_bounds, _ = fit_similarity_bounds(graph, validation_ids, seed=args.seed)

    alphas = dict.fromkeys(RULES, args.alpha)
    if args.alpha_from:
        report = json.loads(Path(args.alpha_from).read_text(encoding="utf-8"))
        alphas.update({entry["rule"]: float(entry["selected_alpha"]) for entry in report["rules"]})

    pool = sorted(item for item in common if str(records[item].get("split")) == args.test_split)

    def visual_similarity(query: str, candidates: list[str]) -> np.ndarray:
        return v_unit[[visual_row[item] for item in candidates]] @ v_unit[visual_row[query]]

    position = {item: index for index, item in enumerate(pool)}
    pool_visual = v_unit[[visual_row[item] for item in pool]]
    pool_graph = g_unit[[graph_row[item] for item in pool]]

    def measure(seed: int) -> dict[str, Any]:
        matched, skipped = match_controls(
            pairs,
            records,
            visual_similarity,
            pool=pool,
            stratum_key=args.stratum_key,
            tolerance=args.tolerance,
            seed=seed,
        )
        if not matched:
            raise SystemExit("no hard positive could be matched to a control")

        queries = [triple.query_id for triple in matched]
        positives = [triple.positive_id for triple in matched]
        controls = [triple.control_id for triple in matched]

        # Every rule is applied to the query's full test candidate list: a rank
        # inside a two-item list carries almost no information, and the row
        # min-max the geometric mean needs would degenerate to {0, 1}.
        full_visual = minmax_scale(
            v_unit[[visual_row[item] for item in queries]] @ pool_visual.T,
            low=visual_bounds.low,
            high=visual_bounds.high,
        )
        full_graph = minmax_scale(
            g_unit[[graph_row[item] for item in queries]] @ pool_graph.T,
            low=graph_bounds.low,
            high=graph_bounds.high,
        )
        rows = np.arange(len(queries))
        positive_columns = np.asarray([position[item] for item in positives])
        control_columns = np.asarray([position[item] for item in controls])

        measurements = {
            "visuel seul": (
                full_visual[rows, positive_columns],
                full_visual[rows, control_columns],
            ),
            "graphe seul": (
                full_graph[rows, positive_columns],
                full_graph[rows, control_columns],
            ),
        }
        for rule in RULES:
            fused = combine(full_visual, full_graph, rule=rule, alpha=alphas[rule])
            measurements[f"fusion {rule} (a={alphas[rule]})"] = (
                fused[rows, positive_columns],
                fused[rows, control_columns],
            )

        def same_book(items: list[str]) -> np.ndarray:
            return np.asarray(
                [
                    records[q].get("book_id") is not None
                    and records[q].get("book_id") == records[i].get("book_id")
                    for q, i in zip(queries, items, strict=True)
                ]
            )

        positive_same_book = same_book(positives)
        control_same_book = same_book(controls)
        cross_volume = ~positive_same_book & ~control_same_book

        per_scorer = {}
        for name, (pos, ctl) in measurements.items():
            entry = {"toutes_paires": bootstrap_paired_auc(pos, ctl, seed=seed)}
            if cross_volume.sum() >= MIN_SUBSET_PAIRS:
                entry["hors_volume_commun"] = bootstrap_paired_auc(
                    pos[cross_volume], ctl[cross_volume], seed=seed
                )
            per_scorer[name] = entry
        return {
            "seed": seed,
            "n_matched": len(matched),
            "skipped": skipped,
            "mechanism": {
                "positive_shares_query_volume": round(float(positive_same_book.mean()), 4),
                "control_shares_query_volume": round(float(control_same_book.mean()), 4),
                "n_cross_volume_pairs": int(cross_volume.sum()),
            },
            "auc": per_scorer,
        }

    draws = [measure(args.seed + offset) for offset in range(max(1, args.seeds))]
    primary = draws[0]

    def across_draws(name: str, subset: str) -> dict[str, float] | None:
        values = [d["auc"][name][subset]["auc"] for d in draws if subset in d["auc"][name]]
        if not values:
            return None
        return {
            "mean": round(float(np.mean(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "n_draws": len(values),
        }

    results = {}
    for name, entry in primary["auc"].items():
        results[name] = dict(entry)
        for subset in ("toutes_paires", "hors_volume_commun"):
            spread = across_draws(name, subset)
            if spread is not None:
                results[name][f"{subset}_across_draws"] = spread

    skipped = primary["skipped"]
    matched = [None] * primary["n_matched"]
    positive_same_book = primary["mechanism"]["positive_shares_query_volume"]
    control_same_book = primary["mechanism"]["control_shares_query_volume"]
    cross_volume_count = primary["mechanism"]["n_cross_volume_pairs"]

    report = {
        "label": args.label,
        "protocol": "protocol-v0.5",
        "pairs_file": args.pairs,
        "visual": args.visual,
        "graph": args.graph,
        "n_hard_positives": len(pairs),
        "n_matched": len(matched),
        "skipped": skipped,
        "tolerance": args.tolerance,
        "stratum_key": args.stratum_key,
        "alphas": alphas,
        "n_control_draws": len(draws),
        "mechanism": {
            "positive_shares_query_volume": positive_same_book,
            "control_shares_query_volume": control_same_book,
            "n_cross_volume_pairs": cross_volume_count,
        },
        "auc": results,
        "draws": draws,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"{args.label}: {len(matched)} appariés, {cross_volume_count} hors volume commun, "
        f"{len(draws)} tirages de témoins"
    )
    for name, entry in results.items():
        overall = entry["toutes_paires"]
        subset = entry.get("hors_volume_commun")

        def interval(result: dict[str, float]) -> str:
            return f"{result['auc']:.3f} [{result['ci_low']:.3f},{result['ci_high']:.3f}]"

        spread = entry.get("toutes_paires_across_draws")
        line = f"  {name:30s} {interval(overall)}"
        if spread:
            line += f" ({spread['min']:.3f}-{spread['max']:.3f} selon tirage)"
        if subset:
            line += f"   hors volume {interval(subset)}"
        print(line)


if __name__ == "__main__":
    main()
