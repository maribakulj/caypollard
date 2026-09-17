#!/usr/bin/env python3
"""Build the Emblematica benchmark: manifest, splits, and a context-only graph.

This is the second corpus, and it exists to answer the question the first one
could not. In the Iconclass AI sample every hard positive was a cross-volume pair
and the only available relation was volume membership, so the graph held no edge
between the images H2 is about. Here the catalogue supplies creator, place and
date per volume, and 74% of emblem books share a creator with another book, so
cross-volume paths exist before any modelling begins.

The manifest deliberately reuses the field names of the Iconclass pipeline, so
auditing, splitting, retrieval, fusion and hard-pair mining run unchanged.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from caypollard.datasets.bsb_metadata import metadata_report, metadata_triples
from caypollard.datasets.emblematica import (
    book_id_from_pictura,
    book_metadata,
    emblem_book_triples,
    iconclass_coverage,
    parse_emblem_xml,
)
from caypollard.graphs.context import ADJACENT_TO
from caypollard.graphs.triples import triples_digest, write_triples_tsv
from caypollard.provenance import manifest_digest, write_jsonl
from caypollard.splitting import combine_group_keys, find_group_leakage, split_records


def load_records(cache: Path) -> list:
    # The search index does not carry the holding library, but the book catalogue
    # does, and every emblem already resolves to a book. Looking it up there
    # avoids one detail request per emblem for a field we can simply join.
    collections_by_book = {
        str(row.get("bookID")): row.get("CollectionCode")
        for row in (
            json.loads(line)
            for line in (cache / "books.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    index = {
        row["emblemID"]: row
        for row in (
            json.loads(line)
            for line in (cache / "index.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    records = []
    for path in sorted((cache / "xml").glob("E*.xml")):
        row = index.get(path.stem, {})
        book = book_id_from_pictura(row.get("Pictura"))
        records.append(
            parse_emblem_xml(
                path.read_text(encoding="utf-8", errors="replace"),
                emblem_id=path.stem,
                book_id=book,
                collection=collections_by_book.get(book) or row.get("CollectionCode"),
                pictura_url=row.get("Pictura"),
            )
        )
    return records


def plate_adjacency(records, *, window: int) -> list[tuple[str, str, str]]:
    """Link consecutive emblems within one volume.

    Emblem identifiers are globally sequential and assigned in reading order
    within a book, so sorting by identifier inside a book recovers plate order
    without needing a page number the records do not carry.
    """
    by_book: dict[str, list[str]] = collections.defaultdict(list)
    for record in records:
        if record.book_id:
            by_book[record.book_id].append(f"emblematica:{record.emblem_id}")
    triples = []
    for members in by_book.values():
        ordered = sorted(members)
        for index, head in enumerate(ordered):
            for offset in range(1, window + 1):
                if index + offset < len(ordered):
                    triples.append((head, ADJACENT_TO, ordered[index + offset]))
    return triples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/raw/emblematica")
    parser.add_argument("--output-dir", default="data/derived/emblematica-v0.1")
    parser.add_argument("--adjacency-window", type=int, default=1)
    parser.add_argument("--seed", default="emblematica-v0.1")
    parser.add_argument(
        "--keep-unlabelled",
        action="store_true",
        help="Retain emblems without Iconclass notations; they cannot be evaluated.",
    )
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    records = load_records(cache)
    audit = {"ingestion": iconclass_coverage(records)}

    usable = records if args.keep_unlabelled else [r for r in records if r.iconclass]
    manifest = [record.as_manifest_record() for record in usable]

    # Volume membership is the only grouping signal here, and unlike the Iconclass
    # sample it covers essentially the whole corpus rather than 17% of it.
    manifest = combine_group_keys(manifest, keys=["book_id"])
    manifest = split_records(manifest, group_key="group_id", seed=args.seed)
    manifest_sha = write_jsonl(manifest, output / "manifest.jsonl")

    context = list(emblem_book_triples(usable)) + plate_adjacency(
        usable, window=args.adjacency_window
    )
    catalogue = [
        json.loads(line)
        for line in (cache / "books.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    present = {record.book_id for record in usable if record.book_id}
    volumes = [book_metadata(row) for row in catalogue if str(row.get("bookID")) in present]
    bibliographic = metadata_triples(volumes)
    triples = tuple(sorted(set(context) | set(bibliographic)))
    write_triples_tsv(triples, output / "context_triples.tsv")

    shared: dict[str, set[str]] = collections.defaultdict(set)
    for head, _relation, tail in bibliographic:
        shared[tail].add(head)
    linked = {book for members in shared.values() if len(members) > 1 for book in members}

    audit |= {
        "n_manifest_records": len(manifest),
        "n_books_in_manifest": len(present),
        "manifest_sha256": manifest_sha,
        "benchmark_digest": manifest_digest(manifest),
        "split_counts": dict(
            sorted(collections.Counter(row["split"] for row in manifest).items())
        ),
        "n_split_groups": len({row["group_id"] for row in manifest}),
        "group_leakage": {k: sorted(v) for k, v in find_group_leakage(manifest).items()},
        "graph": {
            "n_triples": len(triples),
            "n_context_triples": len(set(context)),
            "n_bibliographic_triples": len(bibliographic),
            "relations": dict(sorted(collections.Counter(t[1] for t in triples).items())),
            "n_books_with_a_cross_volume_edge": len(linked),
            "cross_volume_book_fraction": len(linked) / len(present) if present else 0.0,
            "triples_sha256": triples_digest(triples),
            "carries_target_labels": False,
        },
        "catalogue": metadata_report(volumes),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
