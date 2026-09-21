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

CONTEXT = (
    "ctx_above",
    "ctx_below",
    "ctx_left",
    "ctx_right",
    "ctx_inside",
    "ctx_contains",
    "ctx_touching",
    "ctx_count",
)


def with_context(region: dict, *, use_context: bool) -> np.ndarray:
    """A region described by itself, and optionally by what surrounds it.

    The first attempt at naming a region described it alone and failed; the
    pooled control showed that a picture's notation is better predicted by all
    its regions together than by any one. So the region now carries its
    neighbourhood -- how many parts sit above it, below it, beside it, whether
    anything encloses it or it encloses anything -- which is the relation
    channel moved down from the picture to the region.
    """
    base = descriptor(region)
    if not use_context:
        return base
    counts = np.asarray([float(region.get(key, 0)) for key in CONTEXT], dtype=np.float32)
    total = max(counts[-1], 1.0)
    # Proportions, not counts: a region flanked by three of twelve parts is in a
    # different situation from one flanked by three of four.
    return np.concatenate([base, np.log1p(counts), counts[:-1] / total])

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
    parser.add_argument(
        "--max-prevalence",
        type=float,
        default=0.25,
        help="Drop notations carried by more than this fraction of training pictures. "
             "Taking simply the most frequent notations makes the frequency prior "
             "unbeatable by construction -- on the emblems one label covered three "
             "quarters of the test set -- and a task with one dominant answer is not a "
             "naming task.",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Describe each region alone, as the first attempt did, so the contribution "
             "of context is measured rather than assumed.",
    )
    parser.add_argument(
        "--record",
        action="append",
        help="An .npz of whole-image vectors (ids, vectors) used as a ceiling control; "
             "repeatable, which turns the control into a ladder across renderings: "
             "the same notations, the same split, the same classifier, but the picture "
             "described as a whole. Without it a failure at region level cannot be told "
             "apart from a task no representation wins.",
    )
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
                bags[item] = np.stack(
                    [
                        with_context(region, use_context=not args.no_context)
                        for region in row["regions"]
                    ]
                )

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
    ceiling = args.max_prevalence * len(train)
    vocabulary = [
        notation
        for notation, count in counts.most_common()
        if args.min_support <= count <= ceiling
    ][: args.max_notations]
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
        # Margins, not probabilities: for logistic regression the sigmoid is a
        # monotone map applied identically to every model, so it reorders
        # nothing. Cross-model calibration would need held-out fitting per
        # notation, which is only worth doing if the localised hypothesis
        # survives the pooled control below.
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

    # The control that decides what the failure means. The same features and
    # split, but each picture represented by the mean and max of its regions
    # rather than by one region: if pooling works where narrowing does not, the
    # notation is carried by the configuration and localising it is the mistake.
    def pooled(item: str) -> np.ndarray:
        block = normalised[item]
        return np.concatenate([block.mean(axis=0), block.max(axis=0)])

    pooled_train = np.stack([pooled(item) for item in train])
    pooled_models = {}
    for notation in models:
        target = np.array([notation in labels[item] for item in train])
        if target.sum() >= 10 and (~target).sum() >= 10:
            pooled_models[notation] = LogisticRegression(
                max_iter=800, class_weight="balanced"
            ).fit(pooled_train, target)
    pooled_hits1 = pooled_hits5 = 0
    for item in test:
        truth = labels[item] & set(pooled_models)
        if not (labels[item] & set(models)):
            continue
        block = pooled(item).reshape(1, -1)
        ranked = [
            notation
            for _score, notation in sorted(
                (
                    (float(model.decision_function(block)[0]), notation)
                    for notation, model in pooled_models.items()
                ),
                reverse=True,
            )
        ]
        pooled_hits1 += int(bool(ranked) and ranked[0] in truth)
        pooled_hits5 += int(bool(set(ranked[:5]) & truth))

    # The ceiling. The pooled control says whether localising is the mistake;
    # this says whether the shapes are. Same notations, same pictures, same
    # classifier, but the picture described by the record instead of by its
    # regions. If the record clears the prior where the shapes do not, the
    # failure belongs to the descriptor. If it does not clear it either, the
    # protocol is asking a question no representation answers and the region
    # result says nothing about regions.
    record_report = {}
    for record_path in args.record or []:
        loaded = np.load(record_path, allow_pickle=True)
        vectors = {
            str(key): row for key, row in zip(loaded["ids"], loaded["vectors"], strict=True)
        }
        record_train = [item for item in train if item in vectors]
        record_test = [
            item for item in test if item in vectors and (labels[item] & set(models))
        ]
        if len(record_train) >= 50 and record_test:
            block = np.stack([vectors[item] for item in record_train])
            record_centre = block.mean(axis=0)
            record_scale = block.std(axis=0)
            record_scale[record_scale == 0] = 1.0
            block = (block - record_centre) / record_scale
            record_models = {}
            for notation in models:
                target = np.array([notation in labels[item] for item in record_train])
                if target.sum() >= 10 and (~target).sum() >= 10:
                    record_models[notation] = LogisticRegression(
                        max_iter=800, class_weight="balanced"
                    ).fit(block, target)
            record_hits1 = record_hits5 = 0
            subset_prior1 = subset_prior5 = 0
            for item in record_test:
                truth = labels[item] & set(record_models)
                row = ((vectors[item] - record_centre) / record_scale).reshape(1, -1)
                ranked = [
                    notation
                    for _score, notation in sorted(
                        (
                            (float(model.decision_function(row)[0]), notation)
                            for notation, model in record_models.items()
                        ),
                        reverse=True,
                    )
                ]
                record_hits1 += int(bool(ranked) and ranked[0] in truth)
                record_hits5 += int(bool(set(ranked[:5]) & truth))
                subset_prior1 += int(prior_order[0] in labels[item])
                subset_prior5 += int(bool(set(prior_order[:5]) & labels[item]))
            total = len(record_test)
            record_report[Path(record_path).stem] = {
                "pictures": total,
                "hits_at_1": round(record_hits1 / total, 4),
                "hits_at_5": round(record_hits5 / total, 4),
                # The prior recomputed on exactly these pictures, since the
                # record covers a subset of the test bags.
                "prior_at_1": round(subset_prior1 / total, 4),
                "prior_at_5": round(subset_prior5 / total, 4),
            }

    report = {
        "notations_modelled": len(models),
        "train_bags": len(train),
        "test_bags_evaluated": evaluated,
        "iterations": args.iterations,
        "context": not args.no_context,
        "bag_level": {
            "hits_at_1": round(hits1 / max(evaluated, 1), 4),
            "hits_at_5": round(hits5 / max(evaluated, 1), 4),
        },
        "pooled_control": {
            "hits_at_1": round(pooled_hits1 / max(evaluated, 1), 4),
            "hits_at_5": round(pooled_hits5 / max(evaluated, 1), 4),
        },
        "frequency_prior": {
            "hits_at_1": round(prior_hits1 / max(evaluated, 1), 4),
            "hits_at_5": round(prior_hits5 / max(evaluated, 1), 4),
        },
        "record_ceiling": record_report or None,
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
