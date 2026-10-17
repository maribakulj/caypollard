#!/usr/bin/env python3
"""Build the ``relations`` bench from the Iconclass test set (87,744 pictures on disk).

Sixteen subjects, fixed before any run, in two families where the same figures come back
and mostly their relations tell the subjects apart:

- childhood of Christ: Annunciation, Visitation, Nativity, adoration by the shepherds, by
  the kings, Presentation in the temple, flight into Egypt, Holy Family;
- Passion and after: Gethsemane, Ecce Homo, carrying of the cross, the crucified Christ
  with persons under the cross, descent from the cross, mourning, entombment,
  Resurrection.

A candidate has exactly one of the sixteen notations (or a notation below it) among its
codes; 48 candidates per subject are drawn at random (seed 0) and looked at, and the first
30 where the scene is the main content are kept (``scripts/relations_kept.json``). Left
out: book pages where the scene is a small vignette in the text, sheets of several
scenes, pictures that do not show the subject (a portrait, a page of text, a landscape
where the flight into Egypt is a speck, Mary or Simeon alone), and later copies of a print
already kept. Some subjects keep fewer than 30. Many paintings are museum photographs on
the same beige mount (RKD), often of the Holy Family and the shepherds: a whole-picture
channel can match the mount rather than the scene. Each picture searches all the others;
a hit has the same subject. Images are linked, not copied.

Run: ``uv run python scripts/build_relations_bench.py``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SOURCE = Path("data/raw/iconclass-testset")
OUT = Path("data/derived/relations")
KEPT = Path(__file__).with_name("relations_kept.json")
SUBJECTS = {
    "73A52": ("annonciation", "enfance"),
    "73A6": ("visitation", "enfance"),
    "73B13": ("nativite", "enfance"),
    "73B25": ("bergers", "enfance"),
    "73B57": ("mages", "enfance"),
    "73B4": ("presentation", "enfance"),
    "73B64": ("fuite-egypte", "enfance"),
    "73B81": ("sainte-famille", "enfance"),
    "73D31": ("gethsemani", "passion"),
    "73D36": ("ecce-homo", "passion"),
    "73D41": ("portement", "passion"),
    "73D64": ("crucifixion", "passion"),
    "73D71": ("descente", "passion"),
    "73D72": ("deploration", "passion"),
    "73D76": ("mise-au-tombeau", "passion"),
    "73E12": ("resurrection", "passion"),
}


def candidates() -> dict[str, list[str]]:
    """The 48 pictures per subject that were looked at, in the order they were looked at."""
    codes = json.loads((SOURCE / "data.json").read_text())
    by_subject: dict[str, list[str]] = {s: [] for s in SUBJECTS}
    for name, notations in sorted(codes.items()):
        hit = {s for s in SUBJECTS for c in notations if c.startswith(s)}
        if len(hit) == 1:
            by_subject[hit.pop()].append(name)
    return {
        label: random.Random(0).sample(by_subject[n], min(48, len(by_subject[n])))
        for n, (label, _) in SUBJECTS.items()
    }


def main() -> None:
    kept = json.loads(KEPT.read_text())
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for notation, (label, family) in SUBJECTS.items():
        for name in kept[label]:
            item = name.removesuffix(".jpg")
            link = OUT / "images" / f"{item}.jpg"
            if not link.exists():
                link.symlink_to((SOURCE / name).resolve())
            rows.append({"id": item, "label": label, "medium": family, "iconclass": notation})
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} pictures, {len(SUBJECTS)} subjects")


if __name__ == "__main__":
    main()
