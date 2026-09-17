#!/usr/bin/env python3
"""Surface the cross-volume retrievals, and sort them by what explains them.

Phase 7 asks for interpretable cross-book relations, and for the discipline of
separating a retrieval that *suggests* a hypothesis from one that restates a
known bibliographic fact. The distinction is mechanical once the evidence is
attached to each pair:

``atelier``
    Different volume, same creator. The system has recovered an author's reuse of
    a motif across their own books -- interesting, but established by the
    catalogue, not discovered by the model.
``réseau``
    Different volume and creator, shared place or decade. A printing-house or
    regional connection; a historian would want the catalogue checked before
    calling it influence.
``circulation``
    Different volume, creator, place and decade, but a shared Iconclass notation.
    Nothing bibliographic links these books. This is the class that can suggest a
    hypothesis, and the class where the system is doing work no catalogue does.
``inexpliqué``
    Different volume and no shared attribute or notation at all. Either a failure
    or an iconographic resemblance the annotation does not record; both are worth
    looking at, and neither should be reported as a finding.
"""

from __future__ import annotations

import argparse
import collections
import json
from html import unescape
from pathlib import Path
from typing import Any

from caypollard.provenance import read_jsonl

# The predicates the `pure` projection actually carries, so the evidence attached
# to a pair is the evidence the graph itself saw rather than a parallel lookup.
PREDICATES = {
    "created_by": "créateur",
    "published_at": "lieu",
    "published_in": "date",
    "instance_of": "œuvre",
}
ATTRIBUTES = tuple(PREDICATES.values())


def load_queries(path: str) -> list[dict[str, Any]]:
    file = Path(path)
    if file.suffix == ".jsonl":
        return list(read_jsonl(file))
    return json.loads(file.read_text(encoding="utf-8"))["test_queries"]


def classify(query: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, list[str]]:
    shared: list[str] = []
    notations = {str(x) for x in query.get("iconclass", [])} & {
        str(x) for x in candidate.get("iconclass", [])
    }
    if notations:
        shared.append("notation " + ", ".join(sorted(notations)[:3]))

    def common(attribute: str) -> set[str]:
        return set(query.get(attribute) or ()) & set(candidate.get(attribute) or ())

    for attribute in ATTRIBUTES:
        values = common(attribute)
        if values:
            shared.append(f"{attribute} " + ", ".join(sorted(values)[:2]))

    if common("créateur"):
        return "atelier", shared
    if common("lieu") or common("date"):
        return "réseau", shared
    if notations:
        return "circulation", shared
    return "inexpliqué", shared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fused", help="Fused per-query artifact")
    parser.add_argument("manifest", help="Manifest carrying book and bibliographic fields")
    parser.add_argument(
        "--attribute-triples",
        help="The projection the graph was embedded from, e.g. variants/pure.tsv. Attaching "
             "the evidence the graph itself saw avoids explaining a retrieval with a fact "
             "the model never had.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-relevance", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = {str(row["id"]): dict(row) for row in read_jsonl(args.manifest)}
    if args.attribute_triples:
        with open(args.attribute_triples, encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3 or parts[1] not in PREDICATES:
                    continue
                subject, predicate, obj = parts
                row = records.get(subject)
                if row is not None:
                    row.setdefault(PREDICATES[predicate], []).append(obj)

    counts: collections.Counter[str] = collections.Counter()
    by_class: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for query_row in load_queries(args.fused):
        query = records.get(str(query_row["query_id"]))
        if query is None:
            continue
        for rank, item in enumerate(query_row["top_results"][: args.k], start=1):
            candidate = records.get(str(item["item_id"]))
            if candidate is None:
                continue
            if candidate.get("book_id") is None or candidate.get("book_id") == query.get("book_id"):
                continue
            label, shared = classify(query, candidate)
            counts[label] += 1
            relevance = float(item.get("hierarchical_relevance", 0.0))
            if relevance >= args.min_relevance:
                by_class[label].append(
                    {
                        "query_id": str(query_row["query_id"]),
                        "item_id": str(item["item_id"]),
                        "rank": rank,
                        "relevance": relevance,
                        "shares": shared,
                        "query_book": query.get("book_id"),
                        "candidate_book": candidate.get("book_id"),
                        "query_motto": unescape(query.get("motto") or "") or None,
                        "candidate_motto": unescape(candidate.get("motto") or "") or None,
                    }
                )

    total = sum(counts.values()) or 1
    report = {
        "label": args.label,
        "fused": args.fused,
        "k": args.k,
        "cross_volume_results": sum(counts.values()),
        "composition": {
            name: {"n": count, "share": round(count / total, 4)}
            for name, count in counts.most_common()
        },
        "reading": (
            "Only the 'circulation' class can suggest a hypothesis: the two volumes share no "
            "creator, place or decade, so nothing in the catalogue joins them and the shared "
            "notation is what the retrieval found. 'atelier' and 'réseau' restate bibliographic "
            "facts, and should be reported as recovery rather than discovery."
        ),
        "examples": {
            name: sorted(rows, key=lambda row: (-row["relevance"], row["rank"]))[: args.examples]
            for name, rows in by_class.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cross_volume_results", "composition")}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
