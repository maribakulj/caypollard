#!/usr/bin/env python3
"""Do two namers, given the same closed vocabulary, say the same thing?

The record's nodes carry names, and a name is only worth something if it is a
fact about the picture rather than a reading of it. The cheapest way to find out
is to name the same pictures twice by different means -- once by scoring each
segmented part against the vocabulary, once by asking a vision-language model to
read the picture whole -- and see how far the two agree.

Agreement is reported against a floor, because a number on its own says nothing:
the same comparison is run after pairing each picture's names with *another*
picture's, which is what two namers would agree by on a corpus where some words
are simply common. Two namers that share a vocabulary and a corpus will always
agree a little.

The agreed names are written out as well. If agreement marks reliability, a
record built from the intersection should be worth more per node than either
namer alone, and it will certainly be sparser; whether the trade is good is a
measurement and not an intuition.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def load(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = list(row.get("nodes") or [])
    return rows


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--first-name", default="premier")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agreed-output",
                        help="jsonl of the nodes both namers gave, placed where the "
                             "first namer put them")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    first, second = load(Path(args.first)), load(Path(args.second))
    shared = sorted(set(first) & set(second))
    if not shared:
        raise SystemExit("the two namers have no picture in common")

    left = {item: {node["name"] for node in first[item]} for item in shared}
    right = {item: {node["name"] for node in second[item]} for item in shared}

    scores = [jaccard(left[item], right[item]) for item in shared]
    overlaps = [len(left[item] & right[item]) for item in shared]

    shuffled = shared[:]
    random.Random(args.seed).shuffle(shuffled)
    floor = [
        jaccard(left[item], right[other])
        for item, other in zip(shared, shuffled, strict=True)
        if item != other
    ]

    report = {
        "pictures_compared": len(shared),
        "names_per_picture": {
            args.first_name: round(statistics.mean(len(left[i]) for i in shared), 2),
            args.second_name: round(statistics.mean(len(right[i]) for i in shared), 2),
        },
        "jaccard": round(statistics.mean(scores), 4),
        "jaccard_floor_against_another_picture": round(statistics.mean(floor), 4),
        "share_with_any_name_in_common": round(
            sum(1 for value in overlaps if value) / len(overlaps), 4
        ),
        "median_names_in_common": statistics.median(overlaps),
        "reading": (
            "Agreement above the floor means the two namers are seeing the same pictures "
            "rather than the same corpus. Agreement far below one means a name is a "
            "reading and not a fact, whatever the record built on it then achieves."
        ),
    }

    if args.agreed_output:
        path = Path(args.agreed_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as sink:
            for item in shared:
                agreed = left[item] & right[item]
                nodes = [node for node in first[item] if node["name"] in agreed]
                if nodes:
                    sink.write(
                        json.dumps({"id": item, "nodes": nodes}, ensure_ascii=False) + "\n"
                    )
        report["agreed_output"] = str(path)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
