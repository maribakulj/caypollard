"""Utilities for the Iconclass AI Test Set.

The official dataset stores annotations in ``data.json`` as a mapping from image
filenames to lists of Iconclass notations. This module turns that compact source
format into deterministic, row-oriented records suitable for later enrichment.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

ICONCLASS_TESTSET_URL = "https://iconclass.org/testset/"
ICONCLASS_TESTSET_ARCHIVE_URL = (
    "https://iconclass.org/testset/779ba2ca9e977c58d818e3823a676973.zip"
)
ICONCLASS_TESTSET_MD5 = "779ba2ca9e977c58d818e3823a676973"


def load_testset_annotations(path: str | Path) -> dict[str, list[str]]:
    """Load and validate the official ``data.json`` annotation mapping."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Iconclass data.json must contain a JSON object")

    cleaned: dict[str, list[str]] = {}
    for filename, notations in payload.items():
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Every annotation key must be a non-empty filename")
        if not isinstance(notations, list) or not all(isinstance(x, str) for x in notations):
            raise ValueError(f"Annotations for {filename!r} must be a list of strings")
        normalized = sorted({notation.strip() for notation in notations if notation.strip()})
        cleaned[filename.strip()] = normalized
    return dict(sorted(cleaned.items()))


def build_manifest(annotations: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Convert the filename-to-label map into deterministic canonical records.

    The source test set currently does not expose book/edition identifiers in
    ``data.json``. Consequently ``group_id`` is intentionally left null rather
    than fabricating provenance that the source does not provide.
    """
    records: list[dict[str, Any]] = []
    for filename in sorted(annotations):
        records.append(
            {
                "id": f"iconclass-ai:{filename}",
                "filename": filename,
                "iconclass": list(annotations[filename]),
                "source_dataset": "Iconclass AI Test Set",
                "source_url": ICONCLASS_TESTSET_URL,
                "group_id": None,
                "sha256": None,
                "split": None,
            }
        )
    return records


def _root_bucket(notation: str) -> str:
    """Return the broad top-level Iconclass bucket when it is explicit."""
    return notation[0] if notation and notation[0].isdigit() else "unknown"


def audit_annotations(annotations: dict[str, list[str]], *, top_n: int = 20) -> dict[str, Any]:
    """Compute corpus statistics that do not require downloading image bytes."""
    label_counts = Counter(label for labels in annotations.values() for label in labels)
    cardinalities = [len(labels) for labels in annotations.values()]
    roots = Counter(_root_bucket(label) for label in label_counts.elements())
    cooccurrences = Counter()
    for labels in annotations.values():
        for left, right in combinations(sorted(set(labels)), 2):
            cooccurrences[(left, right)] += 1

    n_images = len(annotations)
    n_assignments = sum(cardinalities)
    return {
        "n_images": n_images,
        "n_assignments": n_assignments,
        "n_images_without_annotations": sum(not labels for labels in annotations.values()),
        "n_unique_notations": len(label_counts),
        "n_singleton_notations": sum(count == 1 for count in label_counts.values()),
        "mean_labels_per_image": (n_assignments / n_images) if n_images else 0.0,
        "median_labels_per_image": median(cardinalities) if cardinalities else 0.0,
        "max_labels_per_image": max(cardinalities, default=0),
        "top_notations": label_counts.most_common(top_n),
        "top_cooccurrences": [
            {"notations": list(pair), "count": count}
            for pair, count in cooccurrences.most_common(top_n)
        ],
        "top_level_assignment_counts": dict(sorted(roots.items())),
    }
