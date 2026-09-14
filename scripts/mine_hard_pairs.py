#!/usr/bin/env python3
"""Freeze a pre-fusion hard-pair benchmark from validation-calibrated thresholds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from caypollard.benchmarks.hard_pairs import (
    calibrate_visual_thresholds,
    mine_hard_pairs,
    write_hard_pairs,
)
from caypollard.embeddings.store import load_embedding_table
from caypollard.graphs.iconclass import build_parent_index, child_edges, parse_notations
from caypollard.provenance import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("manifest")
    parser.add_argument("notations")
    parser.add_argument("output")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--sample-pairs", type=int, default=20_000)
    parser.add_argument("--visual-top-k", type=int, default=50)
    parser.add_argument("--random-distant", type=int, default=50)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--semantic-close-min", type=float, default=0.5)
    parser.add_argument("--semantic-distant-max", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("numpy", "faiss"), default="numpy")
    args = parser.parse_args()

    table = load_embedding_table(args.embeddings)
    records = read_jsonl(args.manifest)
    graph_records = parse_notations(args.notations)
    parents = build_parent_index(child_edges(graph_records))
    common = set(table.ids)
    validation_ids = sorted(
        str(record["id"])
        for record in records
        if str(record.get("split")) == args.validation_split and str(record["id"]) in common
    )
    test_ids = sorted(
        str(record["id"])
        for record in records
        if str(record.get("split")) == args.test_split and str(record["id"]) in common
    )
    thresholds, calibration = calibrate_visual_thresholds(
        table,
        validation_ids,
        sample_pairs=args.sample_pairs,
        seed=args.seed,
        semantic_close_min=args.semantic_close_min,
        semantic_distant_max=args.semantic_distant_max,
    )
    pairs, mining = mine_hard_pairs(
        table,
        records,
        parents,
        thresholds,
        query_ids=test_ids,
        candidate_ids=test_ids,
        visual_top_k=args.visual_top_k,
        random_distant_per_query=args.random_distant,
        max_per_class=args.max_per_class,
        seed=args.seed,
        backend=args.backend,
    )
    digest = write_hard_pairs(pairs, args.output)
    metadata = {
        "protocol": "protocol-v0.3",
        "thresholds": asdict(thresholds),
        "calibration": calibration,
        "mining": mining,
        "hard_pairs_sha256": digest,
        "hard_pairs_file": str(Path(args.output)),
    }
    metadata_path = Path(args.output).with_suffix(Path(args.output).suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
