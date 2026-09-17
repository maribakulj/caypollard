#!/usr/bin/env python3
"""Break a transfer result down by object type, date and annotation depth.

An aggregate figure on a corpus this heterogeneous hides more than it shows: a
print, a painting and a piece of furniture are not the same retrieval problem,
and a work carrying one crowd-assigned subject term is not the same as one
carrying six. Phase 8 asks for the breakdown, and for a failure analysis of
metadata and vocabulary mismatch, which is the same computation read the other
way round.

Strata are cut on properties of the *query*, never on its score.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

from caypollard.provenance import read_jsonl
from caypollard.statistics import compare_methods


def load(path: str) -> dict[str, dict[str, Any]]:
    file = Path(path)
    if file.suffix == ".jsonl":
        rows = list(read_jsonl(file))
    else:
        rows = json.loads(file.read_text(encoding="utf-8"))["test_queries"]
    return {str(row["query_id"]): row for row in rows}


def decade_of(value: str | None) -> str:
    if not value or not str(value)[:4].isdigit():
        return "date inconnue"
    return f"{str(value)[:2]}xx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("visual")
    parser.add_argument("fused")
    parser.add_argument("manifest")
    parser.add_argument("--metric", default="ndcg_at_10")
    parser.add_argument("--min-stratum", type=int, default=15)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    visual, fused = load(args.visual), load(args.fused)
    records = {str(row["id"]): row for row in read_jsonl(args.manifest)}
    shared = [q for q in sorted(set(visual) & set(fused)) if q in records]

    def strata_of(query_id: str) -> list[tuple[str, str]]:
        row = records[query_id]
        types = row.get("type") or []
        labels = row.get("iconclass") or []
        return [
            ("type d'objet", types[0] if types else "type inconnu"),
            ("siècle", decade_of(row.get("inception"))),
            (
                "notations par œuvre",
                "1 seule" if len(labels) <= 1 else ("2 à 3" if len(labels) <= 3 else "4 et plus"),
            ),
            ("créateur connu", "oui" if row.get("creator") else "non"),
        ]

    buckets: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for query_id in shared:
        for stratum in strata_of(query_id):
            buckets[stratum].append(query_id)

    breakdown: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for (dimension, value), ids in sorted(buckets.items()):
        usable = [
            q
            for q in ids
            if visual[q].get(args.metric) is not None and fused[q].get(args.metric) is not None
        ]
        if len(usable) < args.min_stratum:
            continue
        v = [float(visual[q][args.metric]) for q in usable]
        f = [float(fused[q][args.metric]) for q in usable]
        comparison = compare_methods(f, v, first_name="fusion", second_name="visuel", seed=42)
        breakdown[dimension].append(
            {
                "stratum": value,
                "n": len(usable),
                "visual": round(statistics.mean(v), 4),
                "fused": round(statistics.mean(f), 4),
                "difference": round(comparison["mean_difference"], 4),
                "ci95": [round(comparison["ci_low"], 4), round(comparison["ci_high"], 4)],
                "p_value": comparison["p_value"],
                "cohens_d": round(comparison["cohens_d"], 3),
            }
        )

    for rows in breakdown.values():
        rows.sort(key=lambda row: -row["difference"])

    helped = [r for rows in breakdown.values() for r in rows if r["difference"] > 0]
    hurt = [r for rows in breakdown.values() for r in rows if r["difference"] < 0]
    report = {
        "label": args.label,
        "metric": args.metric,
        "n_queries": len(shared),
        "min_stratum_size": args.min_stratum,
        "breakdown": dict(breakdown),
        "strata_helped": len(helped),
        "strata_hurt": len(hurt),
        "reading": (
            "Strata are cut on properties of the query, never on its score. A stratum whose "
            "interval spans zero is not evidence either way at this corpus size."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for dimension, rows in breakdown.items():
        print(f"\n=== {dimension} ===")
        for row in rows:
            flag = "*" if row["p_value"] < 0.05 else " "
            line = f"  {row['stratum']:22s} n={row['n']:4d}"
            line += f"  visuel {row['visual']:.3f} → fusion {row['fused']:.3f}"
            line += f"  Δ={row['difference']:+.4f} p={row['p_value']:.4f}{flag}"
            print(line)


if __name__ == "__main__":
    main()
