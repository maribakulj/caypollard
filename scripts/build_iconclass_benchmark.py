#!/usr/bin/env python3
"""Build deterministic phase-1 artifacts from Iconclass source files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from caypollard.datasets.iconclass import (
    audit_annotations,
    build_manifest,
    load_testset_annotations,
)
from caypollard.graphs.iconclass import (
    build_parent_index,
    child_edges,
    hierarchy_depth,
    normalize_notation,
    parse_notations,
    to_skos_graph,
)
from caypollard.provenance import manifest_digest, sha256_file, write_jsonl
from caypollard.splitting import find_checksum_leakage, find_group_leakage, split_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", help="Path to Iconclass test-set data.json")
    parser.add_argument("--notations", help="Optional path to Iconclass notations.txt")
    parser.add_argument("--output-dir", default="data/derived/iconclass-v0.1")
    parser.add_argument(
        "--image-dir",
        help="Optional extracted image directory; enables SHA-256 exact-duplicate grouping.",
    )
    parser.add_argument("--seed", default="iconclass-v0.1")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    annotations = load_testset_annotations(args.annotations)
    audit = audit_annotations(annotations)
    manifest = build_manifest(annotations)

    split_group_key = None
    if args.image_dir:
        image_dir = Path(args.image_dir)
        missing = 0
        for record in manifest:
            image_path = image_dir / record["filename"]
            if image_path.is_file():
                record["sha256"] = sha256_file(image_path)
            else:
                missing += 1
        audit["missing_image_files"] = missing
        audit["images_with_sha256"] = sum(bool(row.get("sha256")) for row in manifest)
        split_group_key = "sha256"

    manifest = split_records(manifest, group_key=split_group_key, seed=args.seed)
    manifest_sha = write_jsonl(manifest, output / "manifest.jsonl")

    audit["manifest_sha256"] = manifest_sha
    audit["split_strategy"] = (
        "stable exact-byte duplicate groups"
        if split_group_key == "sha256"
        else "stable item-hash baseline; not final leakage-controlled split"
    )
    audit["group_leakage"] = {
        key: sorted(value) for key, value in find_group_leakage(manifest).items()
    }
    audit["checksum_leakage"] = {
        key: sorted(value) for key, value in find_checksum_leakage(manifest).items()
    }

    if args.notations:
        records = parse_notations(args.notations)
        edges = child_edges(records)
        parents = build_parent_index(edges)
        resolved = 0
        total = 0
        depths: list[int] = []
        for labels in annotations.values():
            for label in labels:
                total += 1
                if label in parents or normalize_notation(label) in parents:
                    resolved += 1
                    depth = hierarchy_depth(label, parents)
                    if depth is not None:
                        depths.append(depth)
        audit["hierarchy_nodes"] = len(parents)
        audit["hierarchy_edges"] = len(edges)
        audit["resolved_annotation_assignments"] = resolved
        audit["resolved_annotation_fraction"] = resolved / total if total else 0.0
        audit["resolved_hierarchy_depth"] = {
            "min": min(depths) if depths else None,
            "max": max(depths) if depths else None,
            "mean": mean(depths) if depths else None,
            "median": median(depths) if depths else None,
        }

        with (output / "iconclass_edges.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["parent", "child"])
            writer.writerows(edges)
        to_skos_graph(records).serialize(destination=output / "iconclass_skos.ttl", format="turtle")

    audit["benchmark_digest"] = manifest_digest(manifest)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
