#!/usr/bin/env python3
"""Iconclass heads and VLMs compared branch by branch (results/iconclass-tete-pe/PREINSCRIPTION.md, « Comparaison par usage »).

Usage: iconclass_par_usage.py [--pe]   (light, local: reads saved predictions only)
For each benchmark (ICAI 788; Emblematica test of results/iconclass-externe/gel2, chapter 86 left out of truth and predictions) and
each branch b (first digit 1..9, plus 31A and 33): images whose truth holds a code of b; hierarchical F1 between the ancestor sets
restricted to codes of b. Paired T-PE - T difference with a bootstrap over images (ICAI) or books (Emblematica). Without --pe, only
the systems already computed. Writes results/iconclass-tete-pe/par_usage.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "scripts/remote_iconclass_head.py").read_text()
ns: dict = {"re": re}
exec(src[src.index("PART = re.compile"):src.index("def hier_f1")], ns)  # the same hierarchy as the heads
ancestors = ns["ancestors"]
BRANCHES = [str(d) for d in range(1, 10)] + ["31A", "33"]


def terms(code: str) -> list[str]:
    """The parts of a compound notation, split on the colons outside brackets only."""
    return [t.strip() for t in re.split(r":(?![^(]*\))", code) if t.strip()]


def branch_terms(codes: list[str], b: str) -> list[str]:
    """Strict projection on a branch: the terms of b, compounds cut into their parts (the context of a compound is dropped)."""
    return [t for c in codes for t in terms(c) if t.startswith(b)]


def branch_f1(gt: list[str], pred: list[str], b: str) -> float | None:
    g = branch_terms(gt, b)
    if not g:
        return None
    p = branch_terms(pred, b)
    if not p:
        return 0.0
    G = {a for t in g for a in ancestors(t) if a.startswith(b)}  # ancestors above the (sub-)branch are left out too
    P = {a for t in p for a in ancestors(t) if a.startswith(b)}
    i = len(G & P)
    return 0.0 if i == 0 else 2 * i / (len(G) + len(P))


def hier_f1(gt: list[str], pred: list[str]) -> float:
    """The global measure of the heads, unchanged (scripts/remote_iconclass_head.py)."""
    if not pred or not gt:
        return 0.0
    G, P = set().union(*map(ancestors, gt)), set().union(*map(ancestors, pred))
    i = len(G & P)
    return 0.0 if i == 0 else 2 * i / (len(G) + len(P))


def sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_jsonl_unique(path: Path, expected: list[str]) -> dict:
    rows = [json.loads(line) for line in open(path)]
    names = [r["filename"] for r in rows]
    assert len(names) == len(set(names)) and sorted(names) == sorted(expected), f"{path}: duplicates or missing images"
    return {r["filename"]: r for r in rows}


def boot(a: np.ndarray, b: np.ndarray, groups: list[str] | None) -> list[float]:
    rng = np.random.default_rng(0)
    if groups is None:
        idx = [rng.integers(0, len(a), len(a)) for _ in range(2000)]
    else:
        g = np.array(groups)
        ug = sorted(set(groups))
        where = {x: np.where(g == x)[0] for x in ug}
        idx = [np.concatenate([where[x] for x in rng.choice(ug, len(ug))]) for _ in range(2000)]
    d = [a[i].mean() - b[i].mean() for i in idx]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def icai(with_pe: bool):
    pub = {s: sorted(json.load(open(ROOT / f"data/raw/iconclass-eval-preds/passk/sft{s}_1ep.json"))["records"],
                     key=lambda r: r["index"]) for s in ("4b", "9b")}
    gt = [r["gt_codes"] for r in pub["4b"]]
    sys_ = {"T": json.load(open(ROOT / "results/iconclass-tete/full/predictions.json"))["T_head"]}
    if with_pe:
        run = json.load(open(ROOT / "results/iconclass-tete-pe/full/run.json"))
        assert run["smoke"] == 0 and run["feature_config"].startswith("timm/vit_pe_spatial_large_patch14_448.fb@"), run
        assert run["dataset"] == ["davanstrien/iconclass-vlm-brillfull", "8045366ff092a06e8e1c4e47f1eb43cb2a54d732"]
        sys_["T-PE"] = json.load(open(ROOT / "results/iconclass-tete-pe/full/predictions.json"))["T_head"]
    assert all(len(v) == len(gt) == 788 for v in sys_.values())
    for s in pub:
        sys_[s.upper()] = [r["greedy"]["codes"] if r["greedy"].get("valid_json") else [] for r in pub[s]]
    return gt, sys_, None


def emblematica(with_pe: bool):
    gel = ROOT / "results/iconclass-externe/gel2"
    rows = {json.loads(line)["filename"]: json.loads(line) for line in open(ROOT / "data/derived/external-iconclass/candidates.jsonl")}
    test = json.load(open(gel / "test.json"))
    proj = lambda cs: [c for c in cs if not c.strip().startswith("86")]  # noqa: E731
    gt = [proj(rows[t["filename"]]["iconclass"]) for t in test]
    sys_ = {"T": [proj(p) for p in json.load(open(gel / "predictions.json"))["T_head_test"]]}
    if with_pe:
        ref = json.load(open(ROOT / "results/iconclass-tete-pe/externe/reference.json"))
        assert ref["reference_frozen_sha256"] == sha(gel / "frozen.json"), "T-PE was not scored on the gel2 draw"
        assert json.load(open(gel / "vu.json"))["frozen_sha256"] == sha(gel / "frozen.json")
        sys_["T-PE"] = [proj(p) for p in json.load(open(ROOT / "results/iconclass-tete-pe/externe/predictions.json"))["T_head_test"]]
    for s in ("4b", "9b"):
        d = load_jsonl_unique(gel / f"vlm_{s}.jsonl", [t["filename"] for t in test])
        sys_[s.upper()] = [proj(d[t["filename"]]["codes"]) if d[t["filename"]]["valid_json"] else [] for t in test]
    assert all(len(v) == len(gt) == 800 for v in sys_.values())
    return gt, sys_, [t["book_id"] for t in test]


def main() -> None:
    with_pe = "--pe" in sys.argv
    out: dict = {}
    for bench, loader in (("icai_788", icai), ("emblematica_test", emblematica)):
        gt, systems, books = loader(with_pe)
        out[bench] = {}
        # Global measure first: the default-version rule reads only this (non-inferiority of T-PE at -2 points).
        keep = [i for i, g in enumerate(gt) if g]
        glob: dict = {"images": len(keep)}
        gs = {n: np.array([hier_f1(gt[i], p[i]) for i in keep]) for n, p in systems.items()}
        glob.update({n: round(float(v.mean()), 4) for n, v in gs.items()})
        if with_pe:
            glob["T-PE_minus_T"] = round(float(gs["T-PE"].mean() - gs["T"].mean()), 4)
            glob["ci95"] = boot(gs["T-PE"], gs["T"], [books[i] for i in keep] if books else None)
        out[bench]["global"] = glob
        # Branches: descriptive; an interval only with at least 20 images (and the number of books is given).
        for b in BRANCHES:
            keep = [i for i, g in enumerate(gt) if branch_terms(g, b)]
            row: dict = {"images": len(keep), "books": len({books[i] for i in keep}) if books else None}
            scores = {}
            for name, preds in systems.items():
                scores[name] = np.array([branch_f1(gt[i], preds[i], b) for i in keep], dtype=float)
                row[name] = round(float(scores[name].mean()), 4) if keep else None
            if with_pe and len(keep) >= 20:
                row["T-PE_minus_T"] = round(float(scores["T-PE"].mean() - scores["T"].mean()), 4)
                row["ci95"] = boot(scores["T-PE"], scores["T"], [books[i] for i in keep] if books else None)
            out[bench][b] = row
    dest = ROOT / "results/iconclass-tete-pe/par_usage.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
