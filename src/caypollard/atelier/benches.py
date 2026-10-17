"""Test benches: sets of pictures whose right answers are known in advance.

A bench lists its pictures (``id``, a ``label`` that defines what counts as a right
answer, and optionally a ``group``) and how queries are asked:

- ``all``: each picture searches all the others; a hit shares its label;
- ``cross``: each picture searches the other group only (an emblem among museum works,
  and the reverse); a hit shares its label. Each direction is scored on its own.

Scores: ``top1`` (the nearest answer is right), ``mAP`` (average precision over all
right answers), the median rank of the first right answer, and ``top1`` per label.
Chance is the mean over 500 random rankings of the same bench.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path("data/derived")


@dataclass(frozen=True)
class Bench:
    name: str
    root: Path
    rule: str  # "all" or "cross"
    description: str
    queries: tuple[str, ...] | None = None  # for "cross": the groups that ask (default: all)

    def items(self) -> list[dict]:
        rows = json.loads((self.root / "manifest.json").read_text())
        key = next(k for k in ("label", "scene", "subject") if k in rows[0])
        return [
            {
                "id": r["id"],
                "label": r[key],
                "group": r.get("corpus") or r.get("edition"),
                "medium": r.get("medium"),
            }
            for r in rows
        ]

    def image(self, item_id: str) -> Path:
        return self.root / "images" / f"{item_id}.jpg"

    def directions(self, items: list[dict]) -> list[str | None]:
        if self.rule == "all":
            return [None]
        return sorted(self.queries or {it["group"] for it in items})

    def score(self, dist: np.ndarray, items: list[dict]) -> dict:
        out = {}
        for direction in self.directions(items):
            key = "all" if direction is None else f"{direction}_queries"
            out[key] = _score(dist, items, direction)
        return out

    def chance(self, items: list[dict]) -> dict:
        rng = np.random.default_rng(0)
        out = {}
        for direction in self.directions(items):
            repeats = 500 if len(items) < 300 else 20
            runs = [
                _score(rng.random((len(items), len(items))), items, direction)
                for _ in range(repeats)
            ]
            key = "all" if direction is None else f"{direction}_queries"
            out[key] = {k: round(float(np.mean([r[k] for r in runs])), 3) for k in ("top1", "mAP")}
        return out


def _score(dist: np.ndarray, items: list[dict], direction: str | None) -> dict:
    top1, aps, firsts, by_label, by_medium = [], [], [], {}, {}
    for i, it in enumerate(items):
        if direction is not None and it["group"] != direction:
            continue
        gallery = [
            j
            for j, other in enumerate(items)
            if j != i and (direction is None or other["group"] != direction)
        ]
        hits = [items[j]["label"] == it["label"] for j in sorted(gallery, key=lambda j: dist[i, j])]
        if not any(hits):
            continue
        top1.append(hits[0])
        firsts.append(hits.index(True) + 1)
        found, precisions = 0, []
        for rank, hit in enumerate(hits, 1):
            if hit:
                found += 1
                precisions.append(found / rank)
        aps.append(sum(precisions) / len(precisions))
        by_label.setdefault(it["label"], []).append(hits[0])
        if it.get("medium"):
            by_medium.setdefault(it["medium"], []).append((hits[0], aps[-1]))
    out = {
        "queries": len(top1),
        "top1": round(float(np.mean(top1)), 3),
        "mAP": round(float(np.mean(aps)), 3),
        "median_first_hit": float(np.median(firsts)),
        "top1_by_label": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_label.items())},
    }
    if by_medium:
        out["by_medium"] = {
            m: {
                "queries": len(v),
                "top1": round(float(np.mean([t for t, _ in v])), 3),
                "mAP": round(float(np.mean([a for _, a in v])), 3),
            }
            for m, v in sorted(by_medium.items())
        }
        # One label per source painting says nothing on its own: drop it.
        if len(by_label) > 50:
            del out["top1_by_label"]
    return out


BENCHES = {
    b.name: b
    for b in (
        Bench(
            "copies",
            ROOT / "copies",
            "all",
            "Eight scenes of Holbein's Dance of Death in five editions (1526-1816), copies often "
            "mirrored, plus two scenes without copies: 36 pictures. A hit is the same scene.",
        ),
        Bench(
            "sujets",
            ROOT / "subjects",
            "cross",
            "Same subject across media: 12 emblems and 68 museum works (Crucifixion, Carrying of "
            "the Cross, Temptation, Visitation, Baptism, Flagellation). Each side searches the "
            "other; a hit shares the Iconclass subject.",
        ),
        Bench(
            "derivees",
            ROOT / "based-on",
            "cross",
            "Wikidata's documented derivations ('based on' a painting): every derived work in "
            "another medium (prints, drawings, tapestries, sculpture) and as many painted copies "
            "at random. A derived work searches the source paintings; a hit is its own source. "
            "Scores per medium.",
            queries=("derived",),
        ),
        Bench(
            "relations",
            ROOT / "relations",
            "all",
            "Sixteen Christian subjects from the Iconclass test set, 30 pictures each, in two "
            "families where the same figures come back (childhood of Christ, Passion) and "
            "mostly their relations tell the subjects apart. A hit is the same subject; "
            "scores per family.",
        ),
    )
}
