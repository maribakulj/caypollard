#!/usr/bin/env python3
"""What a name is worth, measured four ways instead of one.

Set overlap between two namers was the first thing tried and the wrong thing to
report. It counts ``man`` against ``woman`` as the same failure as ``man``
against ``vase``; it punishes a namer for returning more names than the other,
and the two here return 5.0 and 2.4; it credits agreement that comes from both
namers saying whatever the corpus says most often; it throws away the position,
which the record uses; and above all it answers a question nobody asked.
Agreement is a proxy. What matters is whether a name carries information about
what the picture is catalogued as, and that is measurable directly.

So four measurements, each with the floor that makes it readable.

``soft agreement``
    The two name sets matched one to one, maximising summed similarity in an
    encoder neither namer used, and normalised by the smaller set. Near
    synonyms stop counting as failures, and returning extra names stops being
    punished. The floor is the same computation against another picture's names.

``per-name reliability``
    For each word, how often the other namer also gives it, against how often
    that namer gives it at all. A word whose reliability is at its base rate is
    confirmed by coincidence.

``lift against the catalogue``
    For each word and each notation, how much more likely that notation is on a
    picture carrying the word than on a picture at large. This is the question
    the record is built to answer, asked of one node at a time.

``does agreement mark reliability``
    The same lift, computed separately over names both namers gave and names
    only one gave. If agreement is worth anything, the first should be higher,
    and a record built on intersections should then be worth its sparseness.
    If it is not higher, agreement is a property of the namers and not of the
    pictures, and the sparser record will lose.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

BASE = re.compile(r"[(\[]")


def load_nodes(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = list(row.get("nodes") or [])
    return rows


def load_notations(paths: list[str]) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                labels[str(row["id"])] = {
                    BASE.split(str(value))[0].strip()
                    for value in row.get("iconclass", [])
                }
    return labels


def word_vectors(words: list[str], preset: str) -> np.ndarray:
    from caypollard.text.encoders import HuggingFaceTextEncoder, resolve_text_spec

    encoder = HuggingFaceTextEncoder(resolve_text_spec(preset, None))
    return encoder.encode(words, normalize=True)


def soft_score(left: list[int], right: list[int], similarity: np.ndarray) -> float:
    """One-to-one matching of two name sets, normalised by the smaller."""
    if not left or not right:
        return 0.0
    block = similarity[np.ix_(left, right)]
    rows, columns = linear_sum_assignment(-block)
    return float(block[rows, columns].sum() / min(len(left), len(right)))


def interval(values: list[float], rng: np.random.Generator) -> tuple[float, float]:
    draws = np.asarray(values, dtype=np.float64)
    n = len(draws)
    resampled = draws[rng.integers(0, n, size=(2000, n))].mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--labels", default="data/vocabularies/everyday-nouns.txt")
    parser.add_argument("--encoder", default="labse",
                        help="Used only to judge similarity between words, and "
                             "deliberately not an encoder either namer used")
    parser.add_argument("--first-name", default="premier")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    words = [
        line.strip()
        for line in Path(args.labels).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    vocabulary = list(dict.fromkeys(words))
    index = {word: position for position, word in enumerate(vocabulary)}

    first, second = load_nodes(Path(args.first)), load_nodes(Path(args.second))
    notations = load_notations(args.manifest)
    shared = sorted(set(first) & set(second))
    if not shared:
        raise SystemExit("the two namers have no picture in common")

    left = {i: [index[n["name"]] for n in first[i] if n["name"] in index] for i in shared}
    right = {i: [index[n["name"]] for n in second[i] if n["name"] in index] for i in shared}

    vectors = word_vectors(vocabulary, args.encoder)
    similarity = vectors @ vectors.T

    rng = np.random.default_rng(args.seed)
    shuffled = shared[:]
    random.Random(args.seed).shuffle(shuffled)

    def jaccard(a: list[int], b: list[int]) -> float:
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

    hard = [jaccard(left[i], right[i]) for i in shared]
    soft = [soft_score(left[i], right[i], similarity) for i in shared]
    pairs = [(i, o) for i, o in zip(shared, shuffled, strict=True) if i != o]
    hard_floor = [jaccard(left[i], right[o]) for i, o in pairs]
    soft_floor = [soft_score(left[i], right[o], similarity) for i, o in pairs]

    # Reliability per word: how often the other namer confirms it, against how
    # often that namer says it at all.
    confirmed: collections.Counter[int] = collections.Counter()
    proposed: collections.Counter[int] = collections.Counter()
    for item in shared:
        other = set(right[item])
        for word in set(left[item]):
            proposed[word] += 1
            if word in other:
                confirmed[word] += 1
    base_rate = {
        word: sum(1 for i in shared if word in set(right[i])) / len(shared)
        for word in set(proposed)
    }
    reliability = sorted(
        (
            (
                vocabulary[word],
                proposed[word],
                round(confirmed[word] / proposed[word], 3),
                round(base_rate[word], 3),
                round(
                    (confirmed[word] / proposed[word]) / base_rate[word], 2
                ) if base_rate[word] else None,
            )
            for word in proposed
            if proposed[word] >= args.min_support
        ),
        key=lambda row: -(row[4] or 0),
    )

    # Lift against the catalogue, per assigned name, for one namer.
    def lifts(assigned: dict[str, list[int]]) -> dict[str, list[float]]:
        notation_rate = collections.Counter()
        covered = [i for i in shared if notations.get(i)]
        for item in covered:
            for notation in notations[item]:
                notation_rate[notation] += 1
        total = max(len(covered), 1)
        by_word: dict[int, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        seen: collections.Counter[int] = collections.Counter()
        for item in covered:
            for word in set(assigned[item]):
                seen[word] += 1
                for notation in notations[item]:
                    by_word[word][notation] += 1
        out: dict[str, list[float]] = {}
        for word, counts in by_word.items():
            if seen[word] < args.min_support:
                continue
            best = max(
                (counts[n] / seen[word]) / (notation_rate[n] / total)
                for n in counts
                if notation_rate[n] >= args.min_support
            ) if any(notation_rate[n] >= args.min_support for n in counts) else None
            if best is not None:
                out[vocabulary[word]] = [round(best, 2), seen[word]]
        return out

    lift_first, lift_second = lifts(left), lifts(right)

    # The decisive one: is a name both namers gave worth more than a name only
    # one gave? Computed on the same pictures, the same words, the same
    # notations -- only the agreement differs.
    agreed_lift: list[float] = []
    lone_lift: list[float] = []
    covered = [i for i in shared if notations.get(i)]
    notation_rate = collections.Counter(n for i in covered for n in notations[i])
    total = max(len(covered), 1)
    word_notation: dict[tuple[int, str], int] = collections.Counter()
    word_seen: collections.Counter[int] = collections.Counter()
    for item in covered:
        for word in set(left[item]) | set(right[item]):
            word_seen[word] += 1
            for notation in notations[item]:
                word_notation[(word, notation)] += 1
    for item in covered:
        both = set(left[item]) & set(right[item])
        lone = (set(left[item]) | set(right[item])) - both
        for group, sink in ((both, agreed_lift), (lone, lone_lift)):
            for word in group:
                if word_seen[word] < args.min_support:
                    continue
                for notation in notations[item]:
                    if notation_rate[notation] < args.min_support:
                        continue
                    observed = word_notation[(word, notation)] / word_seen[word]
                    expected = notation_rate[notation] / total
                    sink.append(observed / expected)

    report = {
        "pictures_compared": len(shared),
        "names_per_picture": {
            args.first_name: round(statistics.mean(len(set(left[i])) for i in shared), 2),
            args.second_name: round(statistics.mean(len(set(right[i])) for i in shared), 2),
        },
        "agreement": {
            "hard_jaccard": round(statistics.mean(hard), 4),
            "hard_jaccard_floor": round(statistics.mean(hard_floor), 4),
            "soft_matched": round(statistics.mean(soft), 4),
            "soft_matched_floor": round(statistics.mean(soft_floor), 4),
            "soft_interval": interval(soft, rng),
            "soft_floor_interval": interval(soft_floor, rng),
            "encoder": args.encoder,
        },
        "reliability_per_name": [
            {"name": name, "proposed": n, "confirmed": rate,
             "base_rate": base, "lift": lift}
            for name, n, rate, base, lift in reliability[:15]
        ],
        "reliability_worst": [
            {"name": name, "proposed": n, "confirmed": rate,
             "base_rate": base, "lift": lift}
            for name, n, rate, base, lift in reliability[-10:]
        ],
        "catalogue_lift": {
            args.first_name: dict(
                sorted(lift_first.items(), key=lambda kv: -kv[1][0])[:12]
            ),
            args.second_name: dict(
                sorted(lift_second.items(), key=lambda kv: -kv[1][0])[:12]
            ),
        },
        "does_agreement_mark_reliability": {
            "names_both_namers_gave": {
                "n": len(agreed_lift),
                "mean_lift": round(statistics.mean(agreed_lift), 3) if agreed_lift else None,
                "interval": interval(agreed_lift, rng) if agreed_lift else None,
            },
            "names_only_one_gave": {
                "n": len(lone_lift),
                "mean_lift": round(statistics.mean(lone_lift), 3) if lone_lift else None,
                "interval": interval(lone_lift, rng) if lone_lift else None,
            },
        },
        "reading": (
            "A lift of 1.0 is a name that says nothing about the catalogue. Agreement above "
            "its floor means the namers see the same pictures; agreement far below one means "
            "a name is a reading. The last block decides whether the intersection is worth "
            "building: if agreed names do not lift more, agreement is a property of the "
            "namers and not of the pictures."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
