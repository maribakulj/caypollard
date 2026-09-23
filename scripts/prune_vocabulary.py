#!/usr/bin/env python3
"""Cut the vocabulary down to the words two namers actually confirm.

Reliability is a property of the word, not of the namer: ``ship`` is confirmed
by the second namer 53% of the time against a 1.6% base rate, while ``chain``,
``drum`` and ``glove`` are proposed in the hundreds and confirmed never. A
closed vocabulary is supposed to be inspectable, and part of this one is noise.

Two decisions keep the cut from being self-fulfilling.

*Reliability is measured on the training split alone.* Choosing which words to
keep by how they behave on the pictures the result is then read from would be
the same leak as choosing notations by how often they occur in the target, and
that mistake is already recorded twice in this project.

*Confirmation is scored against the base rate, not raw.* A word both namers
apply to a third of the corpus will be confirmed often by coincidence; what
counts is confirmation over what chance alone would give, and it is required in
both directions, so a word one namer never proposes cannot survive on the
other's enthusiasm.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(path: Path) -> dict[str, set[str]]:
    rows: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = {n["name"] for n in row.get("nodes") or []}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--labels", default="data/vocabularies/everyday-nouns.txt")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-lift", type=float, default=2.0,
                        help="Confirmation over the base rate, required in both directions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    words = [
        line.strip()
        for line in Path(args.labels).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    vocabulary = list(dict.fromkeys(words))

    split: dict[str, str] = {}
    for path in args.manifest:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                split[str(row["id"])] = str(row.get(args.split_field, ""))

    first, second = load(Path(args.first)), load(Path(args.second))
    train = sorted(
        item for item in set(first) & set(second) if split.get(item) == args.train_split
    )
    if not train:
        raise SystemExit("no picture of the training split is named by both")

    def direction(source: dict[str, set[str]], target: dict[str, set[str]]) -> dict[str, tuple]:
        proposed: collections.Counter[str] = collections.Counter()
        confirmed: collections.Counter[str] = collections.Counter()
        for item in train:
            for word in source[item]:
                proposed[word] += 1
                if word in target[item]:
                    confirmed[word] += 1
        base = {
            word: sum(1 for item in train if word in target[item]) / len(train)
            for word in proposed
        }
        return {
            word: (
                proposed[word],
                confirmed[word] / proposed[word],
                base[word],
                (confirmed[word] / proposed[word]) / base[word] if base[word] else 0.0,
            )
            for word in proposed
        }

    forward, backward = direction(first, second), direction(second, first)
    kept, dropped = [], []
    for word in vocabulary:
        one, other = forward.get(word), backward.get(word)
        if not one or not other:
            dropped.append((word, "proposé par un seul nommeur", 0.0))
            continue
        if one[0] < args.min_support or other[0] < args.min_support:
            dropped.append(
                (word, f"support {min(one[0], other[0])}", round(min(one[3], other[3]), 2))
            )
            continue
        lift = min(one[3], other[3])
        if lift < args.min_lift:
            dropped.append((word, "confirmé au hasard", round(lift, 2)))
        else:
            kept.append((word, round(lift, 2), one[0], other[0]))

    kept.sort(key=lambda row: -row[1])
    header = [
        "# The vocabulary, cut to the words two namers confirm in each other.",
        "#",
        f"# Confirmation is measured on the {args.train_split} split alone "
        f"({len(train)} pictures named by both), never on the pictures a result",
        "# is read from, and it is scored against the base rate rather than raw: a",
        "# word both namers scatter over a third of the corpus is confirmed by",
        "# coincidence. Both directions must clear the bar, so a word one namer",
        "# never proposes cannot survive on the other's enthusiasm.",
        "#",
        f"# kept {len(kept)} of {len(vocabulary)} at lift >= {args.min_lift}, "
        f"support >= {args.min_support}",
        "#",
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        "\n".join(header + [word for word, *_ in kept]) + "\n", encoding="utf-8"
    )

    report = {
        "training_pictures": len(train),
        "kept": len(kept),
        "dropped": len(dropped),
        "kept_words": [
            {"word": w, "lift": lift, "proposed_first": a, "proposed_second": b}
            for w, lift, a, b in kept[:20]
        ],
        "dropped_words": [
            {"word": w, "why": why, "lift": lift} for w, why, lift in dropped[:20]
        ],
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({k: v for k, v in report.items() if k != "kept_words"},
                     indent=2, ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()
