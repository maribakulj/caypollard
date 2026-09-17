#!/usr/bin/env python3
"""Measure what textual evidence the emblem corpus actually carries.

H3 -- images that mean the same thing without looking alike -- needs an
annotation that separates a motif from its allegorical reading. Iconclass cannot
supply it: the vocabulary is denotative, so two lions of opposite meaning share
25F23. The obvious remaining candidate is the emblem's own interpretive verse,
the subscriptio.

This script checks whether that candidate exists in usable quantity, by counting
how often a transcribed subscriptio and an Iconclass annotation occur on the
same record. If they rarely co-occur, the verse cannot be joined to the ground
truth and the route is closed whatever its literary interest.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from caypollard.datasets.emblematica import parse_emblem_xml

PICTURA = re.compile(r"<(?:\w+:)?pictura\b[^>]*?xlink:href=\"([^\"]+)\"", re.S)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    counts: collections.Counter[str] = collections.Counter()
    lengths: dict[str, list[int]] = {"motto": [], "subscriptio": [], "pictura_description": []}
    both: list[str] = []
    hosts: collections.Counter[str] = collections.Counter()

    paths = sorted(Path(args.xml_dir).glob("*.xml"))
    for path in paths:
        xml = path.read_text(encoding="utf-8", errors="replace")
        record = parse_emblem_xml(xml, emblem_id=path.stem)
        counts["records"] += 1
        has_iconclass = bool(record.iconclass)
        counts["with_iconclass"] += int(has_iconclass)
        for field in lengths:
            value = getattr(record, field)
            if value:
                counts[f"with_{field}"] += 1
                lengths[field].append(len(value))
        if record.subscriptio:
            counts["subscriptio_with_iconclass"] += int(has_iconclass)
            if has_iconclass:
                both.append(path.stem)
            else:
                url = PICTURA.search(xml)
                hosts[url.group(1).split("/")[2] if url else "aucune pictura"] += 1

    def median(values: list[int]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        return float(
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )

    report = {
        "xml_dir": args.xml_dir,
        "counts": dict(counts),
        "median_length_characters": {field: median(values) for field, values in lengths.items()},
        "subscriptio_and_iconclass_on_the_same_record": {
            "n": counts["subscriptio_with_iconclass"],
            "ids": both[:20],
        },
        "hosts_of_subscriptio_records_without_iconclass": dict(hosts.most_common(10)),
        "reading": (
            "A subscriptio can only be joined to graded relevance through an Iconclass "
            "annotation on the same record. Where that intersection is near-empty, the "
            "interpretive text cannot serve as ground truth for H3 in this corpus."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
