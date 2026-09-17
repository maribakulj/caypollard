#!/usr/bin/env python3
"""Measure what graph fusion actually does to a top-K ranking, query by query.

A mean nDCG gain of +0.03 is compatible with two very different mechanisms: a
small nudge on every query, or a large rescue of a few. The exit criterion for
phase 5 asks which, and asks for the promoted results to be explainable, so this
script reports the churn, the concentration of the gain, and -- for the queries
that moved most -- what each promoted item shares with the query.

It reads the two per-query artifacts as written (a retrieval `per_query.jsonl`
or a fusion `fusion-evaluation.json`), so it re-derives nothing and cannot
disagree with the published summary.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np

from caypollard.provenance import read_jsonl
from caypollard.statistics import bootstrap_mean_ci

SHARED_KEYS = ("book_id", "collection", "creator", "place", "date")
LABEL_KEYS = ("iconclass", "labels", "notations")


def load_queries(path: str) -> dict[str, dict[str, Any]]:
    """Accept either per-query JSONL or a fusion evaluation JSON."""
    file = Path(path)
    if file.suffix == ".jsonl":
        rows = list(read_jsonl(file))
    else:
        rows = json.loads(file.read_text(encoding="utf-8"))["test_queries"]
    return {str(row["query_id"]): row for row in rows}


def top_ids(row: dict[str, Any], k: int) -> list[str]:
    return [str(item["item_id"]) for item in row["top_results"][:k]]


def relevance_at(row: dict[str, Any], k: int) -> dict[str, float]:
    return {
        str(item["item_id"]): float(item.get("hierarchical_relevance", 0.0))
        for item in row["top_results"][:k]
    }


def labels_of(record: dict) -> set[str]:
    for key in LABEL_KEYS:
        value = record.get(key)
        if value:
            return {str(item) for item in value}
    return set()


def shared_attributes(query: dict, candidate: dict) -> list[str]:
    shared = []
    common = labels_of(query) & labels_of(candidate)
    if common:
        shared.append("notation " + ", ".join(sorted(common)[:3]))
    for key in SHARED_KEYS:
        value = query.get(key)
        if value is not None and value == candidate.get(key):
            shared.append(f"{key} {value}")
    return shared


def promotion_category(query: dict, candidate: dict) -> str:
    """Why an item entered the top K: the iconography, the volume, or neither.

    The distinction is the whole phase-4 question. An item promoted because it
    shares a notation is a semantic retrieval; one promoted because it is another
    plate of the same book is the generic contextual gain.
    """
    notation = bool(labels_of(query) & labels_of(candidate))
    context = any(
        query.get(key) is not None and query.get(key) == candidate.get(key)
        for key in SHARED_KEYS
    )
    if notation and context:
        return "notation et contexte"
    if notation:
        return "notation partagée seule"
    if context:
        return "contexte bibliographique seul"
    return "aucun attribut partagé"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", required=True, help="Visual-only per-query artifact")
    parser.add_argument("--fused", required=True, help="Fused per-query artifact")
    parser.add_argument("--manifest", help="Manifest, to explain promoted items")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--label", default="")
    parser.add_argument("--output")
    args = parser.parse_args()

    visual = load_queries(args.visual)
    fused = load_queries(args.fused)
    shared_queries = sorted(set(visual) & set(fused))
    if not shared_queries:
        raise SystemExit("the two artifacts share no query id")

    records = {str(row["id"]): row for row in read_jsonl(args.manifest)} if args.manifest else {}

    churn, deltas, changed_flags = [], [], []
    reordered = 0
    unscorable = 0
    per_query = []
    for query_id in shared_queries:
        before, after = top_ids(visual[query_id], args.k), top_ids(fused[query_id], args.k)
        entered = [item for item in after if item not in set(before)]
        left = [item for item in before if item not in set(after)]
        before_ndcg = visual[query_id].get("ndcg_at_10")
        after_ndcg = fused[query_id].get("ndcg_at_10")
        if before_ndcg is None or after_ndcg is None:
            # A query with no hierarchically relevant candidate has no nDCG to
            # move; it still moves items, so it counts in churn and not in gain.
            unscorable += 1
            continue
        delta = float(after_ndcg) - float(before_ndcg)
        churn.append(len(entered))
        deltas.append(delta)
        changed_flags.append(bool(entered))
        if before != after:
            reordered += 1
        per_query.append(
            {
                "query_id": query_id,
                "entered": entered,
                "left": left,
                "delta_ndcg": delta,
                "promoted_relevance": [
                    relevance_at(fused[query_id], args.k).get(item, 0.0) for item in entered
                ],
                "dropped_relevance": [
                    relevance_at(visual[query_id], args.k).get(item, 0.0) for item in left
                ],
            }
        )

    deltas_arr = np.array(deltas)
    churn_arr = np.array(churn)
    changed = np.array(changed_flags)
    improved = deltas_arr > 1e-9
    degraded = deltas_arr < -1e-9

    total_gain = float(deltas_arr[improved].sum())
    ranked_gain = np.sort(deltas_arr[improved])[::-1]
    decile = max(1, len(ranked_gain) // 10)
    interval = bootstrap_mean_ci(deltas_arr.tolist(), seed=42)

    report: dict[str, Any] = {
        "label": args.label,
        "visual": args.visual,
        "fused": args.fused,
        "k": args.k,
        "n_queries": len(shared_queries),
        "n_scorable": len(deltas),
        "n_without_relevant_candidate": unscorable,
        "movement": {
            "queries_with_any_top_k_change": int(changed.sum()),
            "fraction_with_any_top_k_change": float(changed.mean()),
            "queries_reordered_only": reordered - int(changed.sum()),
            "fraction_reordered_at_all": reordered / len(shared_queries),
            "mean_items_swapped": float(churn_arr.mean()),
            "median_items_swapped": float(np.median(churn_arr)),
            "max_items_swapped": int(churn_arr.max()),
        },
        "effect": {
            "mean_delta_ndcg": float(deltas_arr.mean()),
            "ci95": [interval["ci_low"], interval["ci_high"]],
            "queries_improved": int(improved.sum()),
            "queries_degraded": int(degraded.sum()),
            "queries_unchanged": int(len(deltas_arr) - improved.sum() - degraded.sum()),
            "mean_delta_where_changed": (
                float(deltas_arr[changed].mean()) if changed.any() else 0.0
            ),
            "mean_delta_where_unchanged": (
                float(deltas_arr[~changed].mean()) if (~changed).any() else 0.0
            ),
            "share_of_total_gain_from_top_decile": (
                float(ranked_gain[:decile].sum() / total_gain) if total_gain > 0 else 0.0
            ),
        },
        "promotion_quality": {
            "mean_relevance_of_promoted": float(
                np.mean([r for q in per_query for r in q["promoted_relevance"]] or [0.0])
            ),
            "mean_relevance_of_dropped": float(
                np.mean([r for q in per_query for r in q["dropped_relevance"]] or [0.0])
            ),
        },
    }

    if records:
        explained: collections.Counter[str] = collections.Counter()
        for entry in per_query:
            query = records.get(entry["query_id"])
            if query is None:
                continue
            for item in entry["entered"]:
                candidate = records.get(item)
                if candidate is None:
                    continue
                explained[promotion_category(query, candidate)] += 1
        report["why_promoted"] = dict(explained.most_common())

        def describe(entry: dict[str, Any]) -> dict[str, Any]:
            query = records.get(entry["query_id"], {})
            return {
                "query_id": entry["query_id"],
                "delta_ndcg": round(entry["delta_ndcg"], 4),
                "promoted": [
                    {
                        "item_id": item,
                        "relevance": rel,
                        "shares": shared_attributes(query, records.get(item, {})),
                    }
                    for item, rel in zip(
                        entry["entered"], entry["promoted_relevance"], strict=False
                    )
                ][:5],
                "dropped": [
                    {
                        "item_id": item,
                        "relevance": rel,
                        "shares": shared_attributes(query, records.get(item, {})),
                    }
                    for item, rel in zip(entry["left"], entry["dropped_relevance"], strict=False)
                ][:5],
            }

        ordered = sorted(per_query, key=lambda e: e["delta_ndcg"])
        # The qualitative sample phase 2 asks for: what the graph rescues, and
        # what it costs. Regressions are reported at the same length as gains.
        report["largest_gains"] = [describe(e) for e in reversed(ordered[-args.examples :])]
        report["largest_losses"] = [describe(e) for e in ordered[: args.examples]]

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
