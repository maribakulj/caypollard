#!/usr/bin/env python3
"""Emit the context-only ``G1`` projection from a benchmark manifest.

The manifest already carries the book identifiers recovered from filenames, so
this script only shapes them into triples and records what the resulting graph
does and does not contain. No Iconclass edge is written: see
``docs/GRAPH_PROJECTIONS.md`` for why a graph carrying the evaluation labels is
an oracle rather than evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caypollard.graphs.context import context_graph_report, context_triples
from caypollard.graphs.triples import triples_digest, write_triples_tsv
from caypollard.provenance import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Canonical benchmark manifest JSONL")
    parser.add_argument("--output-dir", default="data/derived/graph-g1")
    parser.add_argument(
        "--adjacency-window",
        type=int,
        default=1,
        help="Link each plate to its N following neighbours in the same volume; "
             "0 for membership only.",
    )
    parser.add_argument(
        "--split",
        help="Restrict the graph to one partition; omit to build over the whole manifest.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    if args.split:
        records = [row for row in records if row.get("split") == args.split]

    triples = context_triples(records, adjacency_window=args.adjacency_window)
    report = context_graph_report(triples, records=records)
    report |= {
        "projection_id": "G1",
        "manifest_file": str(Path(args.manifest)),
        "adjacency_window": args.adjacency_window,
        "split": args.split,
        "triples_sha256": triples_digest(triples),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_triples_tsv(triples, output / "context_triples.tsv")
    # The evaluable pool is the sub-corpus the graph actually reaches; comparing
    # a G1 score against a full-corpus visual baseline would compare two
    # different question sets.
    (output / "graph_images.txt").write_text(
        "".join(f"{triple[0]}\n" for triple in triples if triple[1] == "part_of"),
        encoding="utf-8",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
