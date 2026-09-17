#!/usr/bin/env python3
"""Combine the filename-derived ``G1`` context graph with BSB bibliographic edges.

``G1`` alone links plates only inside one volume, so it holds no edge between the
visually distant but iconographically close pairs the hard-positive benchmark is
made of. This projection adds the cross-volume relations recovered from Munich
IIIF manifests — creator, printer, place, decade, work — and reports how much of
that gap it actually closes, which is the only number that decides whether the
enrichment was worth doing.

No Iconclass edge is emitted, so the result remains a context-only projection in
the sense of ``docs/GRAPH_PROJECTIONS.md``.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from caypollard.datasets.bsb_metadata import metadata_report, metadata_triples, parse_manifest
from caypollard.graphs.context import context_graph_report, context_triples
from caypollard.graphs.triples import triples_digest, write_triples_tsv
from caypollard.provenance import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Canonical benchmark manifest JSONL")
    parser.add_argument("--bsb-cache", default="data/raw/bsb-manifests")
    parser.add_argument("--output-dir", default="data/derived/graph-g3")
    parser.add_argument("--adjacency-window", type=int, default=1)
    parser.add_argument(
        "--no-decades",
        action="store_true",
        help="Ablation: drop date nodes and keep only person, place, and work relations.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    context = context_triples(records, adjacency_window=args.adjacency_window)

    cache = Path(args.bsb_cache)
    volumes = []
    for path in sorted(cache.glob("bsb*.json")):
        volumes.append(parse_manifest(json.loads(path.read_text(encoding="utf-8")),
                                      volume_id=path.stem))
    bibliographic = metadata_triples(volumes, decade_nodes=not args.no_decades)

    triples = tuple(sorted(set(context) | set(bibliographic)))

    report = context_graph_report(triples, records=records)
    report |= {
        "projection_id": "G3-context",
        "n_context_triples": len(context),
        "n_bibliographic_triples": len(bibliographic),
        "bsb_metadata": metadata_report(volumes),
        "adjacency_window": args.adjacency_window,
        "decade_nodes": not args.no_decades,
        "manifest_file": str(Path(args.manifest)),
        "triples_sha256": triples_digest(triples),
        "carries_target_labels": False,
    }

    # The decisive question is not how many triples exist but whether two images
    # in different volumes can now reach each other at all.
    books_by_node: dict[str, set[str]] = collections.defaultdict(set)
    for head, _relation, tail in bibliographic:
        books_by_node[tail].add(head)
    linked_books = {b for members in books_by_node.values() if len(members) > 1 for b in members}
    report["n_books_with_a_cross_volume_edge"] = len(linked_books)
    report["cross_volume_relations"] = dict(
        sorted(collections.Counter(t[1] for t in bibliographic).items())
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_triples_tsv(triples, output / "enriched_triples.tsv")
    (output / "graph_images.txt").write_text(
        "".join(f"{t[0]}\n" for t in triples if t[1] == "part_of"), encoding="utf-8"
    )
    (output / "volumes.jsonl").write_text(
        "".join(json.dumps(v.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for v in volumes),
        encoding="utf-8",
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
