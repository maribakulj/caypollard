#!/usr/bin/env python3
"""Name a region, not a picture: multiple-instance learning over shape descriptors.

Every naming attempt so far has been at the level of the whole picture. A sign
was said to *lean* towards a notation -- three to five per cent of the pictures
carrying it -- and a historian shown that has nothing to contest, because no
claim was made about any particular patch. The system becomes arguable only when
it says *this* shape is why the picture is thought to show an anchor.

The supervision for that does not exist. Nobody has labelled regions. What exists
is a picture labelled with notations and a segmentation of it into regions, which
is exactly the setting multiple-instance learning was invented for: a bag is
positive if at least one instance in it is, and which instance is unknown.

The procedure is the standard alternating one, kept deliberately simple.

1. Regions of pictures carrying the notation are provisional positives; regions
   of pictures not carrying it are certain negatives, which is the only firm
   supervision in the problem.
2. Fit a region classifier.
3. Re-label: within each positive bag keep only the highest-scoring region as
   positive and drop the rest, which is the step that turns "somewhere in this
   picture" into "here".
4. Repeat.

Evaluation is at the bag level because that is where ground truth lives: a
picture's notations are predicted by taking, for each notation, the best score
over its regions. Beating the frequency prior there means the regions carry the
signal, since nothing else was given. The artifact that makes it contestable is
the per-notation table of highest-scoring regions with their pictures.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from caypollard.provenance import read_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_shape_vocabulary import descriptor

BASE = re.compile(r"[(\[]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", action="append", required=True, metavar="CORPUS=PATH")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--min-support", type=int, default=60)
    parser.add_argument("--max-notations", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels: dict[str, set[str]] = {}
    split: dict[str, str] = {}
    for path in args.manifest:
        for row in read_jsonl(path):
            item = str(row["id"])
            labels[item] = {BASE.split(str(x))[0].strip() for x in row.get("iconclass", [])}
            split[item] = str(row.get(args.split_field, "test"))

    bags: dict[str, np.ndarray] = {}
    for spec in args.regions:
        _name, _, path = spec.partition("=")
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            item = str(row["id"])
            if item in labels and row["regions"]:
                bags[item] = np.stack([descriptor(region) for region in row["regions"]])

    train = [item for item in bags if split.get(item) == args.train_split]
    test = [item for item in bags if split.get(item) == args.test_split]
    if not train or not test:
        raise SystemExit("the split fields leave no train or test bag")

    stacked = np.concatenate([bags[item] for item in train])
    centre, scale = stacked.mean(axis=0), stacked.std(axis=0)
    scale[scale == 0] = 1.0
    normalised = {item: (bags[item] - centre) / scale for item in bags}

    counts = collections.Counter(
        notation for item in train for notation in labels[item]
    )
    vocabulary = [
        notation
        for notation, count in counts.most_common(args.max_notations)
        if count >= args.min_support
    ]
    if not vocabulary:
        raise SystemExit("no notation has enough support in the training split")

    models: dict[str, LogisticRegression] = {}
    witnesses: dict[str, list[dict]] = {}
    for notation in vocabulary:
        positive_bags = [item for item in train if notation in labels[item]]
        negative_bags = [item for item in train if notation not in labels[item]]
        if len(positive_bags) < 10 or len(negative_bags) < 10:
            continue
        # Negatives are certain: no region of a picture without the notation can
        # be the reason for it. Positives start as every region of every positive
        # bag and are narrowed by the loop below.
        negatives = np.concatenate([normalised[item] for item in negative_bags])
        if negatives.shape[0] > 20000:
            step = negatives.shape[0] // 20000 + 1
            negatives = negatives[::step]
        chosen = dict.fromkeys(positive_bags)

        model = None
        for _iteration in range(max(args.iterations, 1)):
            if all(index is None for index in chosen.values()):
                positives = np.concatenate([normalised[item] for item in positive_bags])
            else:
                positives = np.stack(
                    [
                        normalised[item][index if index is not None else 0]
                        for item, index in chosen.items()
                    ]
                )
            features = np.concatenate([positives, negatives])
            target = np.concatenate(
                [np.ones(len(positives), dtype=int), np.zeros(len(negatives), dtype=int)]
            )
            model = LogisticRegression(max_iter=800, class_weight="balanced").fit(
                features, target
            )
            for item in positive_bags:
                scores = model.decision_function(normalised[item])
                chosen[item] = int(np.argmax(scores))

        models[notation] = model
        ranked = sorted(
            (
                (float(model.decision_function(normalised[item])[chosen[item]]), item, chosen[item])
                for item in positive_bags
            ),
            reverse=True,
        )
        witnesses[notation] = [
            {"picture": item, "region_index": index, "score": round(score, 3)}
            for score, item, index in ranked[: args.examples]
        ]

    # Bag-level evaluation: a picture's notations are predicted by the best score
    # any of its regions achieves, which is the only honest way to score a
    # region model against picture-level ground truth.
    hits1 = hits5 = 0
    prior_hits1 = prior_hits5 = 0
    evaluated = 0
    prior_order = [n for n, _ in counts.most_common() if n in models]
    for item in test:
        truth = labels[item] & set(models)
        if not truth:
            continue
        evaluated += 1
        scored = sorted(
            (
                (float(models[notation].decision_function(normalised[item]).max()), notation)
                for notation in models
            ),
            reverse=True,
        )
        ranked = [notation for _score, notation in scored]
        hits1 += int(ranked[0] in truth)
        hits5 += int(bool(set(ranked[:5]) & truth))
        prior_hits1 += int(prior_order[0] in truth)
        prior_hits5 += int(bool(set(prior_order[:5]) & truth))

    report = {
        "notations_modelled": len(models),
        "train_bags": len(train),
        "test_bags_evaluated": evaluated,
        "iterations": args.iterations,
        "bag_level": {
            "hits_at_1": round(hits1 / max(evaluated, 1), 4),
            "hits_at_5": round(hits5 / max(evaluated, 1), 4),
        },
        "frequency_prior": {
            "hits_at_1": round(prior_hits1 / max(evaluated, 1), 4),
            "hits_at_5": round(prior_hits5 / max(evaluated, 1), 4),
        },
        "witnesses": witnesses,
        "reading": (
            "Beating the prior at bag level means the regions carry the signal, since nothing "
            "but regions was given. The witnesses are the contestable part: each names a "
            "picture and the index of the region the model holds responsible, so a historian "
            "can look and say no."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("witnesses", "reading")},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
